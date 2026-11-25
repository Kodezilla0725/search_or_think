"""Token-level multi-turn rollout for GRPO. Same protocol as agent.run_agent, but it keeps the sampled token ids and
log probs, and marks retrieved <info> tokens with env_mask = 0 so they get no loss (TRL rollout_func contract)."""
from .agent import LIMIT_MSG, SEARCH_RE, format_info
from .scoring import norm_query

KEYS = ("prompt_ids", "completion_ids", "logprobs", "env_mask", "n_search", "fake_info", "forced", "n_repeat")
FORCE_TEXT = {"search": "\n<search>", "answer": "\n<answer>"}
FORCE_CODE = {"search": 1, "answer": 2}
OPEN_TAGS = ("<search>", "<answer>")


def _keep_before(tok, ids, text, tag):
    """How many of a turn's tokens to keep so that the kept text stops before the last `tag` in it."""
    pos = text.rfind(tag)
    for k in range(len(ids), -1, -1):
        if len(tok.decode(ids[:k])) <= pos:
            return k
    return 0


def multi_turn_rollout(llm, tok, retrieve, prompts, sampling_params, max_searches=4, max_len=4096, force=None,
                       force_params=None, strict=False):
    """llm: a vLLM LLM; prompts: chat-formatted strings; sampling_params must stop at agent.STOP
    (</search>, </answer>, <info>) with include_stop_str_in_output=True and logprobs=0.

    force: per rollout None, "search" or "answer". A forced rollout thinks freely; if its first turn ends at
    </think>, the environment writes the opening <search> or <answer> tag (env tokens: no loss) and the model
    carries on from there, so the search decision is the only thing taken out of its hands. force_params are the
    sampling params of that first turn: sampling_params plus the stop string "</think>". If the model gets to a
    search or an answer without closing its <think>, the rollout simply stays free (forced = 0).

    strict: the policy can learn to skip </think> and so escape the forcing (it did in stage A2). With strict=True
    force_params must also stop at "<search>" and "<answer>". A forced rollout that opens the other action is cut
    back to just before that tag, and the environment writes "</think>" plus the forced tag (env tokens). One that
    opens the forced action itself keeps it. Either way it counts as forced."""
    def enc(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    n = len(prompts)
    P = [enc(p) for p in prompts]
    C, L, M = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    searches, over, done, fake = [0] * n, [0] * n, [False] * n, [0] * n
    force = force or [None] * n
    forced, opened = [0] * n, [""] * n  # opened: a tag the environment wrote, needed to parse the model's next turn
    seen, repeats = [set() for _ in range(n)], [0] * n  # repeats: searches whose query the episode already issued

    def env(i, text):  # tokens the model did not generate: no loss, no sampling log prob
        ids = enc(text)
        C[i] += ids
        L[i] += [None] * len(ids)
        M[i] += [0] * len(ids)

    for turn in range(max_searches + 3):
        for i in range(n):  # stop anything that could overflow the context window
            done[i] = done[i] or len(P[i]) + len(C[i]) + sampling_params.max_tokens > max_len
        act = [i for i in range(n) if not done[i]]
        if not act:
            break
        first = {i for i in act if turn == 0 and force[i]}  # forced rollouts stop their first turn at </think>
        outs = {}
        for idx, params in (([i for i in act if i in first], force_params), ([i for i in act if i not in first],
                                                                             sampling_params)):
            if idx:
                reqs = [{"prompt_token_ids": P[i] + C[i]} for i in idx]
                outs.update(zip(idx, llm.generate(reqs, params, use_tqdm=False)))
        need, queries = [], []
        for i in act:
            g = outs[i].outputs[0]
            ids = list(g.token_ids)
            C[i] += ids
            L[i] += [lp[t].logprob for lp, t in zip(g.logprobs, ids)]
            M[i] += [1] * len(ids)
            text, opened[i] = opened[i] + g.text, ""
            tail = text.rstrip()
            if i in first and tail.endswith("</think>"):
                env(i, FORCE_TEXT[force[i]])
                opened[i], forced[i] = FORCE_TEXT[force[i]].strip(), FORCE_CODE[force[i]]
            elif i in first and strict and tail.endswith(OPEN_TAGS):
                tag, want = next(t for t in OPEN_TAGS if tail.endswith(t)), FORCE_TEXT[force[i]].strip()
                forced[i], opened[i] = FORCE_CODE[force[i]], want
                if tag != want:  # take the model's own choice back and write the forced one
                    drop = len(ids) - _keep_before(tok, ids, g.text, tag)
                    if drop:
                        del C[i][-drop:], L[i][-drop:], M[i][-drop:]
                    env(i, "</think>" + FORCE_TEXT[force[i]])
            elif not tail.endswith("</search>"):
                done[i] = True  # answered, wrote its own <info>, or stopped without a tag
                fake[i] = int(tail.endswith("<info>"))
            elif searches[i] < max_searches:
                m = SEARCH_RE.findall(text)
                need.append(i)
                queries.append(m[-1].strip() if m else "")
            else:
                over[i] += 1
                done[i] = over[i] >= 2
                if not done[i]:
                    env(i, LIMIT_MSG)
        for i, q, res in zip(need, queries, retrieve(queries) if queries else []):
            searches[i] += 1
            repeats[i] += norm_query(q) in seen[i]
            seen[i].add(norm_query(q))
            env(i, format_info(res))
    return {"prompt_ids": P, "completion_ids": C, "logprobs": L, "env_mask": M, "n_search": searches,
            "fake_info": fake, "forced": forced, "n_repeat": repeats}


def resample_flat_groups(out, recs, roll, rewards_of, reserve, G, rounds=1):
    """DAPO-style dynamic sampling. A question group whose G rollouts all earn the same reward gives GRPO no
    gradient, so it is swapped for a fresh question from `reserve`, up to `rounds` times.

    out: rollout dict over consecutive groups of G; recs: one question record per rollout (edited in place, like out)
    roll(records) -> rollout dict for those records; rewards_of(out, recs) -> one reward per rollout
    Returns the number of groups that were swapped."""
    swapped = 0
    for _ in range(rounds):
        r = rewards_of(out, recs)
        flat = [g for g in range(0, len(r), G) if max(r[g:g + G]) - min(r[g:g + G]) < 1e-8]
        if not flat:
            break
        new_recs = [rec for rec in (next(reserve) for _ in flat) for _ in range(G)]
        new = roll(new_recs)
        for j, g in enumerate(flat):
            for k in KEYS:
                out[k][g:g + G] = new[k][j * G:(j + 1) * G]
            recs[g:g + G] = new_recs[j * G:(j + 1) * G]
        swapped += len(flat)
    return swapped