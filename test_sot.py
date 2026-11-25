"""Unit tests: python -m pytest -q test_sot.py"""
import math

import pytest

from sot.agent import extract_answer, run_agent
from sot.analysis import (boundary_signal, boundary_table, enriched_sample, failure_modes, p_by_id,
                          paired_bootstrap, qor, sor)
from sot.rewards import (OLD_TRIGGER, Boundary, Warmup, WarmupTrigger, compute_rewards, estimate_boundary,
                         group_stats, replay_trigger, reward_fixed, reward_plain, reward_saas, reward_soft_saas,
                         saas_label, tighten_n_min)
from sot.scoring import em, f1, has_answer, knows, n_distinct, norm_query

A = 0.1  # alpha


# scoring
def test_scoring():
    assert em("The Eiffel Tower!", ["eiffel tower"]) == 1.0
    assert f1("Paris France", ["Paris"]) == pytest.approx(2 / 3)
    assert f1("yes", ["no"]) == 0.0 and f1("no", ["no"]) == 1.0
    assert has_answer("Built in 1889 by Eiffel.", ["1889"]) and not has_answer("in 18890", ["1889"])


# rewards
def test_plain_and_fixed():
    assert reward_plain(0.7, 3) == 0.7
    assert reward_fixed(1.0, 2, 0.1) == pytest.approx(0.8)


def test_wrong_answers_never_penalised():
    for b in [Boundary(1.0, 0, "NoSearch"), Boundary(0.0, 1, "NeedSearch"), Boundary(0.5, 1, "Undetermined")]:
        assert reward_saas(0.5, 4, b, A) == 0.5
        assert reward_soft_saas(0.5, 4, b, A) == 0.5


def test_saas():
    assert reward_saas(1.0, 2, Boundary(1.0, 0, "NoSearch"), A) == pytest.approx(0.8)
    assert reward_saas(1.0, 3, Boundary(0.0, 1, "NeedSearch"), A) == pytest.approx(0.8)
    assert reward_saas(1.0, 1, Boundary(0.0, 1, "NeedSearch"), A) == 1.0
    assert reward_saas(1.0, 4, Boundary(0.5, 1, "Undetermined"), A) == 1.0  # no signal


def test_soft_saas_matches_saas_at_extremes_and_mixes_between():
    assert reward_soft_saas(1.0, 2, Boundary(1.0, 0, "NoSearch"), A) == pytest.approx(0.8)
    assert reward_soft_saas(1.0, 3, Boundary(0.0, 1, "NeedSearch"), A) == pytest.approx(0.8)
    # p = 0.5, N = 3, N_min = 1: 0.5*3 + 0.5*2 = 2.5 searches of penalty
    assert reward_soft_saas(1.0, 3, Boundary(0.5, 1, "Undetermined"), A) == pytest.approx(0.75)
    # never answerable and no correct searching rollout: only searches beyond the cap count
    assert reward_soft_saas(1.0, 4, Boundary(0.0, None, "Undetermined"), A) == 1.0
    assert reward_soft_saas(1.0, 2, Boundary(1.0, 0, "NoSearch"), A, w=0.0) == 1.0  # warm up off


def test_estimate_boundary():
    assert estimate_boundary([True] * 4, [(True, 1)]).label == "NoSearch"
    b = estimate_boundary([False] * 4, [(False, 4), (True, 2), (True, 1)])
    assert (b.label, b.n_min, b.p) == ("NeedSearch", 1, 0.0)
    assert estimate_boundary([False] * 4, [(False, 3)]).label == "Undetermined"


def test_saas_threshold_is_a_ratio():  # gap 1
    # NoSearch from delta / K = 2 / 4 = 0.5 of the search disabled rollouts, at any K
    assert estimate_boundary([True, True, False, False], [(True, 1)]).label == "NoSearch"
    assert estimate_boundary([True] * 4 + [False] * 4, []).label == "NoSearch"  # 4 of 8
    assert estimate_boundary([True] * 3 + [False] * 5, []).label == "Undetermined"  # 3 of 8, nothing correct with search
    # 1 of 4 correct without search and a correct searching rollout: the released code says NeedSearch (fewer than
    # delta correct), the paper says Undetermined (it wants none correct)
    assert estimate_boundary([True, False, False, False], [(True, 2)]).label == "NeedSearch"
    assert estimate_boundary([True, False, False, False], [(True, 2)], rule="paper").label == "Undetermined"
    assert saas_label(0.0, 1, "paper") == saas_label(0.0, 1, "code") == "NeedSearch"
    with pytest.raises(ValueError):
        saas_label(0.2, 1, "other")


def test_compute_rewards_all_kinds():  # gap 8
    f1s, ns = [1.0, 1.0, 0.5, 0.0], [2, 0, 3, 1]
    bs = [Boundary(0.5, 1, "NoSearch")] * 4
    assert compute_rewards("plain", f1s, ns, bs) == ([1.0, 1.0, 0.5, 0.0], [False] * 4)
    r, pen = compute_rewards("fixed", f1s, ns, bs, lam=0.1)
    assert r == pytest.approx([0.8, 1.0, 0.2, -0.1]) and pen == [True, False, True, True]  # fixed is not gated
    r, pen = compute_rewards("saas", f1s, ns, bs, alpha=A)
    assert r == pytest.approx([0.8, 1.0, 0.5, 0.0]) and pen == [True, False, False, False]
    r, pen = compute_rewards("soft", f1s, ns, bs, alpha=A)  # 0.5 * 2 + 0.5 * max(0, 2 - 1) = 1.5 searches
    assert r == pytest.approx([0.85, 1.0, 0.5, 0.0]) and pen == [True, False, False, False]
    for kind in ("fixed", "saas", "soft"):  # w = 0: every reward is the plain one
        assert compute_rewards(kind, f1s, ns, bs, w=0.0) == ([1.0, 1.0, 0.5, 0.0], [False] * 4)
    r, _ = compute_rewards("plain", f1s, ns, bs, format_bonus=0.1, format_ok=[True, True, True, True])
    assert r == pytest.approx([1.0, 1.0, 0.5, 0.1])  # bonus only for a well formed wrong answer
    assert compute_rewards("plain", f1s, ns, bs, format_bonus=0.1, format_ok=[False] * 4)[0][3] == 0.0
    with pytest.raises(ValueError):
        compute_rewards("other", f1s, ns, bs)


def test_tighten_n_min():
    bs = [Boundary(0.0, None, "Undetermined")] * 2 + [Boundary(0.0, 1, "NeedSearch")] * 2
    out = tighten_n_min(bs, [1.0, 0.0, 1.0, 1.0], [3, 4, 2, 3], G=2)
    assert (out[0].n_min, out[0].label, out[1].n_min) == (3, "NeedSearch", 3)  # the group's own best
    assert out[2].n_min == 1  # never raised above the stored value


def test_group_stats():  # gap 7
    s = group_stats([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0], 4)
    assert s["zero_var_share"] == pytest.approx(2 / 3) and s["zero_var_all_zero"] == pytest.approx(1 / 3)
    assert s["flat_groups"] == [0, 1]


def test_warmup_trigger():
    w = WarmupTrigger(window=5, eps=0.01, ramp=10, ma=3)
    ws = [w.update(min(0.05 * t, 0.6)) for t in range(60)]  # rises for 12 steps, then flat
    assert w.trigger_step is not None and 12 < w.trigger_step < 30
    assert all(x == 0.0 for x in ws[:w.trigger_step + 1])
    assert ws[-1] == 1.0 and ws == sorted(ws)


def test_warmup_updates_once_per_optimizer_step():  # gap 4
    kw = dict(window=5, eps=0.01, ma=3)
    ref = WarmupTrigger(ramp=10, **kw)
    w = Warmup("self", ramp=10, **kw)
    ws = []
    for step in range(60):
        r = min(0.05 * step, 0.6)
        ref.update(r)
        for part in (r - 0.1, r + 0.1, r):  # several reward calls inside one optimizer step
            ws.append(w.weight(step))
            w.observe(step, part)
        assert len(set(ws[-3:])) == 1  # w is constant within the step
        assert len(w.trigger.rewards) == step  # one committed update per finished step, never more
    w.weight(60)
    assert w.trigger.rewards == pytest.approx(ref.rewards) and w.trigger_step == ref.trigger_step
    assert ws[::3] == [ref.weight(t) for t in range(60)] and 0.0 < ws[3 * (ref.trigger_step + 1)] < 1.0 == ws[-1]
    # resume at step 40: the state file may already hold later steps, which must be dropped
    w2 = Warmup("self", ramp=10, **kw)
    w2.restore(w.state(), upto=40)
    assert len(w2.trigger.rewards) == 40 and w2.trigger_step == ref.trigger_step
    assert w2.weight(40) == ref.weight(40)


def test_trigger_rule_ignores_step_noise():
    import random
    rng = random.Random(0)
    rising = [0.10 + 0.0017 * t + rng.gauss(0, 0.04) for t in range(150)]  # the smoke test's slope and noise
    assert replay_trigger(rising) is None                      # new rule: no trigger while reward still rises
    assert replay_trigger(rising, **OLD_TRIGGER) is not None   # the interim report's rule fires on the noise
    plateau = rising + [0.35 + rng.gauss(0, 0.04) for _ in range(150)]
    assert 150 < replay_trigger(plateau) <= 260                # and it does fire once reward is flat
    assert replay_trigger([0.3] * 61) == 61 and replay_trigger([0.3] * 60) is None  # earliest possible step


def test_warmup_specs():
    assert [Warmup("off").weight(t) for t in (0, 5)] == [1.0, 1.0]
    hard = Warmup("manual_step:3")  # SAAS: a hard switch
    assert [hard.weight(t) for t in range(5)] == [0.0, 0.0, 0.0, 1.0, 1.0]
    ramp = Warmup("manual_step:2", ramp=4)
    assert [ramp.weight(t) for t in range(8)] == [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert Warmup("self").ramp == 50 and Warmup("self").weight(0) == 0.0
    with pytest.raises(ValueError):
        Warmup("sometimes")
    w = Warmup("off")  # the trigger is tracked for every spec, so a plain run logs when it would have fired
    for t in range(80):
        w.weight(t)
        w.observe(t, 0.3)
    assert w.trigger_step is not None and w.state()["rewards"] == [0.3] * 80


# agent loop with a fake model and retriever
def fake_retrieve(queries):
    return [[{"title": f"T_{q}", "text": f"text about {q}"}] for q in queries]


def make_generate(script):
    """script: prompt keyword -> list of outputs, one per turn."""
    calls = {}

    def gen(prompts, stop, max_tokens, temperature):
        outs = []
        for p in prompts:
            key = next(k for k in script if k in p)
            turn = calls.get(p.split("|")[0], 0)
            calls[p.split("|")[0]] = turn + 1
            outs.append(script[key][turn] if turn < len(script[key]) else "")
        return outs
    return gen


def test_run_agent_paths():
    script = {
        "Q_direct": ["<think>know it</think><answer> Paris </answer>"],
        "Q_search": ["<think>?</think><search> eiffel </search>", "<think>ok</think><answer>1889</answer>"],
        "Q_greedy": ["<search> a </search>"] * 6 + ["<answer>x</answer>"],
        "Q_rambles": ["no tags here just text\nfinal guess Rome"],
        "Q_fakes": ["<think>?</think><search> x </search>", "<think>hm</think>\n<info>"],
    }
    prompts = [f"{k}|" for k in script]
    st = run_agent(make_generate(script), fake_retrieve, prompts, max_searches=2)
    d, s, g, r, k = st
    assert (k["stop_reason"], k["n_search"], k["format_ok"]) == ("fake_info", 1, False)  # gap 5: own <info> ends it
    assert (d["answer"], d["n_search"], d["stop_reason"]) == ("Paris", 0, "answer")
    assert (s["answer"], s["n_search"]) == ("1889", 1) and "<info>" in s["text"] and "T_eiffel" in s["text"]
    assert g["n_search"] == 2 and g["stop_reason"] == "search_limit" and "Search limit reached" in g["text"]
    assert r["format_ok"] is False and r["answer"] == "final guess Rome" and r["stop_reason"] == "no_tag"


def test_closed_book_and_prefill():
    gen = make_generate({"Q": ["<think>x</think><answer>a</answer>"]})
    st = run_agent(gen, None, ["Q|"], allow_search=False,
                   prefill=[{"text": "<search> Q </search>", "searches": [{"query": "Q", "passages": []}]}])
    assert st[0]["answer"] == "a" and st[0]["n_search"] == 1
    assert extract_answer("<answer>a</answer> then <answer> b </answer>") == "b"
    assert extract_answer("<answer><answer> Paris </answer>") == "Paris"  # model repeats the prefilled tag
    assert extract_answer("<answer> Paris, France\nmore text", lenient=True) == "Paris, France"  # never closed
    assert extract_answer("<answer> Paris, France\nmore text") is None


# metrics
def test_metrics():
    recs = {"h1": {"dataset": "hotpotqa", "sup_titles": ["A", "B"], "golds": ["x"]},
            "n1": {"dataset": "nq", "sup_titles": [], "golds": ["1889"]}}
    agent = [
        {"id": "h1", "dataset": "hotpotqa", "n_search": 3, "f1": 1.0, "searches": [
            {"titles": ["A"], "texts": [""]}, {"titles": ['"B"'], "texts": [""]}, {"titles": ["C"], "texts": [""]}]},
        {"id": "n1", "dataset": "nq", "n_search": 2, "f1": 0.0, "searches": [
            {"titles": ["E"], "texts": ["built in 1889"]}, {"titles": ["E"], "texts": [""]}]},
    ]
    s = sor(agent, recs)
    assert s["hotpotqa"]["sor"] == pytest.approx(1 / 3) and s["nq"]["sor"] == pytest.approx(1 / 2)
    assert s["hotpotqa"]["complete"] == 1.0 and s["nq"]["complete"] == 1.0
    closed = [{"id": "h1", "dataset": "hotpotqa", "em": e} for e in (1, 1, 0, 0)] + \
             [{"id": "n1", "dataset": "nq", "em": 0}] * 4
    pm = p_by_id(closed)
    assert pm == {"h1": 0.5, "n1": 0.0}
    q = qor(agent, pm)["hotpotqa"]
    assert (q["eligible"], q["qor"], q["f1_eligible"]) == (1, 1.0, 1.0) and math.isnan(q["search_rate_rest"])
    assert qor(agent, pm)["nq"]["search_rate_rest"] == 1.0 and qor(agent, pm)["nq"]["eligible"] == 0
    assert p_by_id([{"id": "a", "em": 0.0, "cover": 1.0}, {"id": "a", "em": 0.0, "cover": 0.0}], "cover") == {"a": 0.5}


def test_one_definition_per_quantity():  # gap 2
    golds = {"h1": ["Paris"], "n1": ["1889"], "u1": ["Rome"]}
    closed = ([{"id": "h1", "answer": a, "em": e} for a, e in
               [("Paris", 1), ("It is Paris, France", 0), ("Lyon", 0), ("Paris", 1)]]  # cover 3/4, EM 2/4
              + [{"id": "n1", "answer": "1900", "em": 0}] * 4
              + [{"id": "u1", "answer": "Rome I think", "em": 0}] + [{"id": "u1", "answer": "Milan", "em": 0}] * 3)
    row = lambda i, ds, ok, f, n: {"id": i, "dataset": ds, "format_ok": ok, "f1": f, "n_search": n}  # noqa: E731
    agent = [row("h1", "hotpotqa", True, 1.0, 2),
             row("n1", "nq", False, 1.0, 1),   # right answer outside <answer> tags: earns no reward, so no N_min
             row("n1", "nq", True, 0.8, 1),    # partial F1 is not correct
             row("n1", "nq", True, 1.0, 3), row("n1", "nq", True, 1.0, 2),
             row("u1", "nq", True, 0.0, 1)]
    t = boundary_table(closed, agent, golds)
    assert (t["h1"]["p"], t["h1"]["p_em"], t["h1"]["label"]) == (0.75, 0.5, "NoSearch")  # p(q) is the cover match
    assert (t["n1"]["p"], t["n1"]["n_min"], t["n1"]["label"]) == (0.0, 2, "NeedSearch")  # N_min: strict F1 = 1
    assert (t["u1"]["p"], t["u1"]["n_min"], t["u1"]["label"]) == (0.25, None, "Undetermined")
    assert knows("It is Paris, France", ["Paris"]) and not knows(None, ["Paris"])
    sig = boundary_signal(closed, agent, golds)
    assert sig["hotpotqa"]["signal_saas"] == 1 and sig["hotpotqa"]["penalised_saas"] == 1.0
    assert sig["nq"]["questions"] == 2 and sig["nq"]["labels"] == {"NeedSearch": 1, "Undetermined": 1}
    assert sig["nq"]["signal_saas"] == 0.5 and sig["nq"]["signal_soft"] == 1.0  # u1: only Soft-SAAS has a signal
    assert sig["nq"]["correct_searching_rollouts"] == 2 and sig["nq"]["penalised_saas"] == 0.5  # 3 > N_min, 2 is not
    # the paper's stricter NeedSearch leaves a question with p = 0.25 and a correct searching rollout Undetermined
    closed2 = [{"id": "x", "answer": "Rome", "em": 1}] + [{"id": "x", "answer": "no", "em": 0}] * 3
    agent2 = [row("x", "nq", True, 1.0, 1)]
    assert boundary_table(closed2, agent2, {"x": ["Rome"]})["x"]["label"] == "NeedSearch"
    assert boundary_table(closed2, agent2, {"x": ["Rome"]}, rule="paper")["x"]["label"] == "Undetermined"


def test_enriched_sample():
    recs = [{"id": str(i)} for i in range(100)]
    pmap = {str(i): 0.5 if i < 10 else 0.0 if i < 90 else 1.0 for i in range(100)}
    out = enriched_sample(recs, pmap, 30)
    assert len(out) == 30 and sum(0 < pmap[r["id"]] < 1 for r in out) == 10
    assert len({r["id"] for r in out}) == 30
    out = enriched_sample(recs, pmap, 60)  # only 10 in-between questions exist: take all, fill with the rest
    assert len(out) == 60 and sum(0 < pmap[r["id"]] < 1 for r in out) == 10
    flip = {k: 0.5 if v == 0.0 else v for k, v in pmap.items()}  # 90 in between, 10 at p = 1
    out = enriched_sample(recs, flip, 60)
    assert len(out) == 60 and sum(flip[r["id"]] == 1.0 for r in out) == 10


# token-level rollout for GRPO, with a fake vLLM and a character tokenizer
class _Tok:
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in s]}


class _LP:
    logprob = -0.5


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeLLM:
    def __init__(self, script):
        self.script, self.turn = script, {}

    def generate(self, reqs, sp, use_tqdm=False):
        outs = []
        for r in reqs:
            text = "".join(map(chr, r["prompt_token_ids"]))
            key = next(k for k in self.script if text.startswith(k))
            t = self.turn[key] = self.turn.get(key, -1) + 1
            out = self.script[key][min(t, len(self.script[key]) - 1)]
            ids = [ord(c) for c in out]
            outs.append(_NS(outputs=[_NS(text=out, token_ids=ids, logprobs=[{i: _LP()} for i in ids])]))
        return outs


def test_multi_turn_rollout():
    from sot.rollout import multi_turn_rollout
    script = {"A:": ["<search> q1 </search>", "<answer> x </answer>"],
              "B:": ["<answer> y </answer>"],
              "C:": ["<search> a </search>"],
              "D:": ["<search> d </search>", "so the results are\n<info>"]}
    sp = _NS(max_tokens=50)
    out = multi_turn_rollout(_FakeLLM(script), _Tok(), fake_retrieve, ["A:", "B:", "C:", "D:"], sp, max_searches=2)
    assert out["n_search"] == [1, 0, 2, 1]
    assert out["fake_info"] == [0, 0, 0, 1]  # gap 5: D writes its own <info> and the episode ends there
    d = "".join(map(chr, out["completion_ids"][3]))
    assert d.endswith("<info>") and d.count("<info>") == 2 and out["env_mask"][3][-1] == 1  # one real, one invented
    for c, lp, m in zip(out["completion_ids"], out["logprobs"], out["env_mask"]):
        assert len(c) == len(lp) == len(m)
        assert all((x is None) == (k == 0) for x, k in zip(lp, m))  # env tokens: no log prob, no loss
    a = "".join(map(chr, out["completion_ids"][0]))
    masked = "".join(ch for ch, k in zip(a, out["env_mask"][0]) if k == 0)
    assert a.startswith("<search> q1 </search>") and a.endswith("<answer> x </answer>")
    assert "<info>" in masked and "T_q1" in masked and "<answer>" not in masked
    c = "".join(map(chr, out["completion_ids"][2]))
    assert c.count("Search limit reached") == 1  # third search refused once, then the episode ends
    short = multi_turn_rollout(_FakeLLM(script), _Tok(), fake_retrieve, ["A:"], sp, max_len=60)
    assert short["n_search"] == [1] and short["completion_ids"][0][-1] == ord("\n")  # stopped before overflowing


def test_resample_flat_groups():  # gap 7: zero-variance groups are swapped for fresh questions
    from sot.rollout import KEYS, resample_flat_groups
    G = 2

    def roll(recs):  # the "rollout" just carries the question id in every field
        return {k: [r["id"] for r in recs] for k in KEYS}

    def rewards_of(out, recs):  # questions a* give equal rewards within the group, questions b* do not
        return [0.0 if r["id"].startswith("a") else float(i % G) for i, r in enumerate(recs)]

    recs = [{"id": i} for i in ("a1", "a1", "b1", "b1", "a2", "a2")]
    out = roll(recs)
    reserve = iter([{"id": i} for i in ("b2", "a3", "b3", "b4")])
    n = resample_flat_groups(out, recs, roll, rewards_of, reserve, G, rounds=2)
    assert n == 3  # round 1 swaps a1 and a2 for b2 and a3; round 2 swaps a3 for b3
    assert [r["id"] for r in recs] == ["b2", "b2", "b1", "b1", "b3", "b3"]
    assert all(out[k] == ["b2", "b2", "b1", "b1", "b3", "b3"] for k in KEYS)  # rollouts and records stay aligned
    assert resample_flat_groups(out, recs, roll, rewards_of, reserve, G, rounds=2) == 0 and next(reserve)["id"] == "b4"


# trainer plumbing (no GPU, no TRL)
def _write(path, rows):
    import json
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def test_training_questions_carry_their_boundary(tmp_path):  # gap 3
    from train_grpo import load_questions, parse_args
    (tmp_path / "data").mkdir()
    _write(tmp_path / "data/train_enriched.jsonl",
           [{"id": "a", "dataset": "nq", "question": "qa", "golds": ["x"], "p_pool": 0.5},
            {"id": "b", "dataset": "hotpotqa", "question": "qb", "golds": ["y"], "p_pool": 0.25},
            {"id": "c", "dataset": "nq", "question": "qc", "golds": ["z"], "p_pool": 0.0}])
    _write(tmp_path / "relabel.jsonl", [{"id": "b", "p": 0.125, "n_min": 2}, {"id": "c", "p": 0.0, "n_min": None}])
    args = parse_args(["--project", str(tmp_path), "--reward", "soft", "--boundary", str(tmp_path / "relabel.jsonl")])
    _, rows = load_questions(args)
    assert [set(r) for r in rows] == [{"prompt", "id", "dataset", "golds", "p", "n_min", "label"}] * 3
    a, b, c = rows
    assert (a["p"], a["n_min"], a["label"]) == (0.5, -1, "NoSearch")       # not relabelled: base model p(q), no N_min
    assert (b["p"], b["n_min"], b["label"]) == (0.125, 2, "NeedSearch")    # relabelled values win
    assert (c["p"], c["n_min"], c["label"]) == (0.0, -1, "Undetermined")
    args = parse_args(["--project", str(tmp_path), "--boundary", str(tmp_path / "relabel.jsonl"), "--saas_rule", "paper"])
    assert load_questions(args)[1][1]["label"] == "Undetermined"
    assert args.tag == "plain_s0" and parse_args(["--project", "p", "--reward", "saas", "--seed", "2"]).tag == "saas_s2"


def test_reward_function_logs_and_warms_up_once_per_step(tmp_path):  # gaps 4, 5, 7, 8
    import json
    from train_grpo import make_reward_fn, parse_args
    args = parse_args(["--project", str(tmp_path), "--reward", "soft", "--alpha", "0.1", "--G", "4",
                       "--warmup", "manual_step:1"])
    warm = Warmup(args.warmup, args.ramp)
    fn = make_reward_fn(args, warm, str(tmp_path))
    batch = dict(
        completions=["<search> q </search> <info> d </info> <answer> Paris </answer>", "<answer> Paris </answer>",
                     "Paris", "<search> q </search>\n<info>",
                     "<answer> no </answer>"] + ["<answer> nope </answer>"] * 3,
        golds=[["Paris"]] * 4 + [["yes"]] * 4, n_search=[1, 0, 0, 1, 0, 0, 0, 0], fake_info=[0, 0, 0, 1, 0, 0, 0, 0],
        p=[0.5] * 8, n_min=[1] * 4 + [-1] * 4, label=["NoSearch"] * 8, id=list("aaaabbbb"), resampled=[1] * 8)
    logged = {}

    def call(step):
        logged.clear()
        return fn(prompts=["q"] * 8, completion_ids=[[0]] * 8, trainer_state=_NS(global_step=step),
                  log_metric=lambda k, v: logged.__setitem__(k, v), **batch)

    assert call(0) == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # w = 0 before the switch: plain F1, strict tags
    assert logged["w"] == 0.0 and logged["penalised_share"] == 0.0
    call(0)  # a second reward call in the same optimizer step must not advance the warm-up
    r = call(1)
    assert r == pytest.approx([0.95, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.1 * (0.5 * 1 + 0.5 * max(0, 1 - 1))
    assert set(logged) == {"f1", "searches_mean", "w", "trigger_step", "penalised_share", "zero_var_share",
                           "zero_var_all_zero", "fake_info_rate", "format_ok", "resampled_groups", "n0_share",
                           "n1_share", "n2plus_share", "forced_share", "repeat_share"}
    assert logged["repeat_share"] == 0.0  # the rollout reported no repeated query
    batch["n_repeat"] = [1, 0, 0, 0, 0, 0, 0, 0]
    call(1)
    assert logged["repeat_share"] == 0.5  # 2 searches in the batch, one of them a repeat
    del batch["n_repeat"]
    r = call(1)
    assert (logged["n0_share"], logged["n1_share"], logged["n2plus_share"]) == (0.75, 0.25, 0.0)
    assert logged["w"] == 1.0 and logged["penalised_share"] == 1 / 8 and logged["fake_info_rate"] == 1 / 8
    assert logged["zero_var_share"] == 0.5 and logged["zero_var_all_zero"] == 0.5 and logged["format_ok"] == 6 / 8
    assert logged["f1"] == 0.25 and logged["searches_mean"] == 0.25 and logged["trigger_step"] == -1
    assert logged["resampled_groups"] == 1
    # forced arms: rollout 0 was forced to search, rollout 1 forced to answer without search
    batch["forced"] = [1, 2, 0, 0, 0, 0, 0, 0]
    r = call(2)
    assert r[0] == 1.0  # the forced search is not charged: 0 chosen searches
    assert logged["penalised_share"] == 0.0 and logged["forced_share"] == 0.25
    assert logged["f1_forced_search"] == 1.0 and logged["f1_forced_answer"] == 1.0
    assert logged["f1"] == 0.0 and logged["searches_mean"] == pytest.approx(1 / 6)  # free rollouts only
    assert logged["n0_share"] == pytest.approx(5 / 6)
    assert json.load(open(tmp_path / "warmup.json"))["rewards"] == [0.25, 0.25, 0.0]  # one entry per optimizer step
    assert len(open(tmp_path / "samples.jsonl").readlines()) == 12  # two rollouts per reward call


# evaluation helpers
def test_paired_bootstrap():
    a = [0.0] * 50 + [1.0] * 50
    same = paired_bootstrap(a, a)
    assert same == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    better = paired_bootstrap(a, [min(1.0, x + 0.5) for x in a])  # +0.5 on half of the items
    assert better["diff"] == pytest.approx(0.25) and 0.15 < better["lo"] < 0.25 < better["hi"] < 0.35
    assert paired_bootstrap(a, [1 - x for x in a])["lo"] < 0 < paired_bootstrap(a, [1 - x for x in a])["hi"]


def test_failure_modes():
    row = lambda n, f, stop, qs: {"dataset": "nq", "n_search": n, "f1": f, "stop_reason": stop,  # noqa: E731
                                  "searches": [{"query": q} for q in qs]}
    m = failure_modes([row(0, 1.0, "answer", []), row(2, 0.0, "answer", ["Eiffel  tower", "eiffel tower"]),
                       row(1, 0.5, "fake_info", [""]), row(1, 0.0, "answer", ["x"])])["nq"]
    assert m["stop"] == {"answer": 0.75, "fake_info": 0.25}
    assert m["n_search_share"] == {"0": 0.25, "1": 0.5, "2+": 0.25}
    assert m["f1_by_n_search"] == {"0": 1.0, "1": 0.25, "2+": 0.0}
    assert m["empty_query"] == 0.25 and m["repeat_query"] == 0.25  # 4 queries: one empty, one repeated
    assert m["distinct_queries"] == 0.75  # 0, 1, 1 and 1 distinct queries


def test_distinct_queries_and_think_closed():
    from sot.analysis import think_closed
    assert norm_query("  Eiffel   TOWER ") == "eiffel tower"
    assert n_distinct([]) == 0 and n_distinct(["a", "A ", "b"]) == 2 and n_distinct(["q", "q", "q"]) == 1
    assert think_closed("<think> hm </think>\n<search> q </search>")
    assert not think_closed("<think> hm <search> q </search> </think>")  # reached a tag with the block still open
    assert not think_closed("<think><answer> x </answer>") and think_closed("<think> a </think> b")


def test_n_min_counts_distinct_queries():
    """A policy that repeats one query never shows fewer than 2 searches; its distinct queries show the real need."""
    golds = {"q1": ["Paris"], "q2": ["Rome"]}
    closed = [{"id": q, "answer": "no idea", "em": 0.0} for q in golds for _ in range(4)]
    row = lambda q, f, n, d: {"id": q, "dataset": "nq", "format_ok": True, "f1": f, "n_search": n,  # noqa: E731
                              "n_distinct": d}
    agent = [row("q1", 1.0, 3, 1), row("q1", 1.0, 2, 2), row("q1", 0.0, 1, 1), row("q2", 1.0, 2, 2)]
    raw, dis = boundary_table(closed, agent, golds), boundary_table(closed, agent, golds, count="n_distinct")
    assert (raw["q1"]["n_min"], dis["q1"]["n_min"]) == (2, 1) and raw["q2"]["n_min"] == dis["q2"]["n_min"] == 2
    assert raw["q1"]["label"] == dis["q1"]["label"] == "NeedSearch"
    # with N_min = 1 the repeated second search of a correct rollout is charged; with the raw N_min = 2 it is free
    b_raw, b_dis = (Boundary(t["q1"]["p"], t["q1"]["n_min"], t["q1"]["label"]) for t in (raw, dis))
    assert reward_saas(1.0, 2, b_raw, A) == 1.0 and reward_saas(1.0, 2, b_dis, A) == pytest.approx(0.9)
    assert reward_soft_saas(1.0, 2, b_raw, A) == 1.0 and reward_soft_saas(1.0, 2, b_dis, A) == pytest.approx(0.9)
    sig = boundary_signal(closed, agent, golds, count="n_distinct")["nq"]
    assert sig["penalised_saas"] == pytest.approx(2 / 3) and sig["mean_pen_saas"] == pytest.approx(1.0)  # 2, 1, 0
    # a one-search budget probe: q2 answered correctly with a single search lowers its N_min for the same rollouts
    probe = boundary_table(closed, agent + [row("q2", 1.0, 1, 1)], golds, count="n_distinct")
    assert probe["q2"]["n_min"] == 1
    sig = boundary_signal(closed, agent, golds, count="n_distinct", table=probe)["nq"]
    assert sig["penalised_saas"] == 1.0 and sig["mean_pen_saas"] == pytest.approx(4 / 3)  # 2, 1, 1


def test_forced_search_decision():
    """The environment takes the search decision for some rollouts: it writes the opening tag after </think>."""
    from sot.rollout import multi_turn_rollout
    script = {"S:": [" I may know this. </think>", " capital of france </search>", "<think> ok </think>\n<answer> x </answer>"],
              "A:": [" not sure </think>", " Lyon </answer>"],
              "N:": [" hm </think>\n<search> q </search>", "<answer> y </answer>"],   # free rollout
              "X:": ["<answer> z </answer>"]}                                          # never closes <think>: stays free
    llm, sp = _FakeLLM(script), _NS(max_tokens=50)
    out = multi_turn_rollout(llm, _Tok(), fake_retrieve, ["S:", "A:", "N:", "X:"], sp,
                             force=["search", "answer", None, "search"], force_params=_NS(max_tokens=50))
    assert out["forced"] == [1, 2, 0, 0] and out["n_search"] == [1, 0, 1, 0]
    text = ["".join(map(chr, c)) for c in out["completion_ids"]]
    env = ["".join(ch for ch, k in zip(t, m) if k == 0) for t, m in zip(text, out["env_mask"])]
    assert text[0].startswith(" I may know this. </think>\n<search> capital of france </search>")
    assert "T_capital of france" in env[0] and env[0].startswith("\n<search>")  # the forced tag is an env token
    assert text[1] == " not sure </think>\n<answer> Lyon </answer>" and env[1] == "\n<answer>"
    assert extract_answer(text[1]) == "Lyon" and extract_answer(text[0]) == "x"
    assert "<search>" not in env[2] and env[3] == ""  # free rollouts: only retrieved text is masked
    for lp, m in zip(out["logprobs"], out["env_mask"]):
        assert all((x is None) == (k == 0) for x, k in zip(lp, m))


def test_rollout_counts_repeated_queries():
    from sot.rollout import multi_turn_rollout
    script = {"R:": [" x </think>\n<search> Eiffel tower </search>", "<search> eiffel  tower </search>",
                     "<search> Paris </search>", "<answer> a </answer>"],
              "O:": [" y </think>\n<search> q </search>", "<answer> b </answer>"]}
    out = multi_turn_rollout(_FakeLLM(script), _Tok(), fake_retrieve, ["R:", "O:"], _NS(max_tokens=50))
    assert out["n_search"] == [3, 1] and out["n_repeat"] == [1, 0]  # same query up to case and spacing


class _TokD(_Tok):
    def decode(self, ids):
        return "".join(map(chr, ids))


def test_strict_forcing_cannot_be_dodged():
    """A forced answer rollout that opens <search> without closing <think> is cut back and made to answer."""
    from sot.rollout import multi_turn_rollout
    script = {"S:": [" pick a query <search>", " Lyon </answer>"],      # opens a search, never closed <think>
              "A:": [" I know it <answer>", " Paris </answer>"],        # opens the forced action itself
              "T:": [" sure </think>", " Rome </answer>"],               # closes <think>: the old path
              "F:": [" hm <search> q </search>", "<answer> y </answer>"]}  # free: untouched
    sp, spf = _NS(max_tokens=50), _NS(max_tokens=50)
    loose = multi_turn_rollout(_FakeLLM(dict(script)), _TokD(), fake_retrieve, ["T:"], sp, force=["answer"],
                               force_params=spf)
    assert loose["forced"] == [2]
    out = multi_turn_rollout(_FakeLLM(script), _TokD(), fake_retrieve, ["S:", "A:", "T:", "F:"], sp,
                             force=["answer", "answer", "answer", None], force_params=spf, strict=True)
    assert out["forced"] == [2, 2, 2, 0] and out["n_search"] == [0, 0, 0, 1]
    text = ["".join(map(chr, c)) for c in out["completion_ids"]]
    env = ["".join(ch for ch, k in zip(t, m) if k == 0) for t, m in zip(text, out["env_mask"])]
    assert text[0] == " pick a query </think>\n<answer> Lyon </answer>" and env[0] == "</think>\n<answer>"
    assert extract_answer(text[0]) == "Lyon" and "<search>" not in text[0]
    assert text[1] == " I know it <answer> Paris </answer>" and env[1] == ""   # its own choice, kept
    assert text[2] == " sure </think>\n<answer> Rome </answer>" and env[2] == "\n<answer>"
    assert "<info>" in env[3] and "<search>" not in env[3] and out["forced"][3] == 0
    for c, lp, m in zip(out["completion_ids"], out["logprobs"], out["env_mask"]):
        assert len(c) == len(lp) == len(m) and all((x is None) == (k == 0) for x, k in zip(lp, m))


def test_keep_before_cuts_at_the_tag():
    from sot.rollout import _keep_before
    tok = _TokD()
    ids = [ord(c) for c in "abc <search>"]
    assert _keep_before(tok, ids, "abc <search>", "<search>") == 4

    class Chunky(_TokD):  # tokens that straddle the tag: "c <" is one token
        def decode(self, ids):
            return "".join({1: "ab", 2: "c <", 3: "search>"}[i] for i in ids)
    assert _keep_before(Chunky(), [1, 2, 3], "abc <search>", "<search>") == 1


def test_seed_averaged_paired_and_ood_records():
    from sot.analysis import seed_averaged_paired
    from sot.data import flashrag_record, sup_titles
    a = [[{"id": "q1", "f1": 0.0}, {"id": "q2", "f1": 1.0}], [{"id": "q1", "f1": 1.0}, {"id": "q2", "f1": 1.0}]]
    b = [[{"id": "q1", "f1": 1.0}, {"id": "q2", "f1": 1.0}, {"id": "q3", "f1": 1.0}]]
    r = seed_averaged_paired(a, b, "f1")
    assert r["n"] == 2 and r["diff"] == pytest.approx(0.25)  # q1: 0.5 -> 1, q2: 1 -> 1; q3 is not in a
    assert sup_titles({"supporting_facts": {"title": ["B", "A", "B"], "sent_id": [0, 1, 2]}}) == ["A", "B"]
    assert sup_titles({"supporting_facts": [["X", 0], ["Y", 2]]}) == ["X", "Y"]
    assert sup_titles({"paragraphs": [{"title": "P", "is_supporting": True}, {"title": "Q", "is_supporting": False}]}) == ["P"]
    assert sup_titles({"other": 1}) == [] and sup_titles(None) == []
    rec = flashrag_record("2wiki", 7, {"id": "abc", "question": "Who?", "golden_answers": ["X", ""],
                                       "metadata": {"supporting_facts": [["T", 0]]}})
    assert rec == {"id": "2wiki_abc", "dataset": "2wiki", "split": "test", "question": "Who?", "golds": ["X"],
                   "sup_titles": ["T"]}
    assert flashrag_record("popqa", 3, {"question": "Q", "golden_answers": "Y"})["id"] == "popqa_3"
    # SOR on a set with supporting titles uses them; without, the answer string
    recs = {"w": {"dataset": "2wiki", "sup_titles": ["T"], "golds": ["zzz"]},
            "p": {"dataset": "popqa", "sup_titles": [], "golds": ["zzz"]}}
    rows = [{"id": "w", "dataset": "2wiki", "searches": [{"titles": ["T"], "texts": [""]}, {"titles": ["U"], "texts": [""]}]},
            {"id": "p", "dataset": "popqa", "searches": [{"titles": ["T"], "texts": ["zzz"]}, {"titles": ["U"], "texts": [""]}]}]
    s = sor(rows, recs)
    assert s["2wiki"] == {"searches": 2, "sor": 0.5, "complete": 1.0} and s["popqa"]["sor"] == 0.5


def test_forced_arms_in_args(tmp_path):
    from train_grpo import parse_args
    a = parse_args(["--project", str(tmp_path), "--force_search", "2", "--force_answer", "2"])
    assert (a.force_search, a.force_answer, a.G, a.force_strict) == (2, 2, 8, False)
    assert parse_args(["--project", str(tmp_path), "--force_answer", "2", "--force_strict"]).force_strict
    with pytest.raises(SystemExit):
        parse_args(["--project", str(tmp_path), "--force_search", "4", "--force_answer", "4"])