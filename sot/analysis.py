"""Metrics over saved runs: accuracy tables, p(q) distribution, QOR, SOR, and the SAAS vs Soft-SAAS signal check."""
import collections
import json
import math
import random

from .rewards import SAAS_RULE, Boundary, estimate_boundary, is_correct, penalty_saas, penalty_soft
from .scoring import has_answer, knows, norm_query, norm_title


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def by_dataset(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["dataset"]].append(r)
    return dict(sorted(g.items()))


def summarize(rows):
    return {ds: {"n": len(v), "em": mean(r["em"] for r in v), "f1": mean(r["f1"] for r in v),
                 "searches": mean(r["n_search"] for r in v), "format_ok": mean(r["format_ok"] for r in v)}
            for ds, v in by_dataset(rows).items()}


def p_by_id(closed_rows, col="em"):
    """col: "em" (strict) or "cover" (gold answer contained in the prediction)."""
    d = collections.defaultdict(list)
    for r in closed_rows:
        d[r["id"]].append(r[col] >= 1.0)
    return {k: sum(v) / len(v) for k, v in d.items()}


def enriched_sample(recs, pmap, n, mid_frac=1 / 3, seed=0):
    """n questions with about mid_frac at 0 < p(q) < 1 (fewer if the pool has fewer); the rest keep the
    pool's own mix of p(q) = 0 and p(q) = 1."""
    rng = random.Random(seed)
    mid = [r for r in recs if 0 < pmap[r["id"]] < 1]
    rest = [r for r in recs if not 0 < pmap[r["id"]] < 1]
    rng.shuffle(mid)
    rng.shuffle(rest)
    k = min(len(mid), round(n * mid_frac))
    out = mid[:k] + rest[:n - k]
    out += mid[k:k + n - len(out)]  # too few p(q) = 0 or 1 questions: top up with in-between ones
    rng.shuffle(out)
    return out


def p_distribution(pmap, recs):
    out = {}
    for ds, v in by_dataset(recs).items():
        ps = [pmap[r["id"]] for r in v if r["id"] in pmap]
        out[ds] = {"n": len(ps), "hist": dict(sorted(collections.Counter(round(p, 2) for p in ps).items())),
                   "p0": mean(p == 0 for p in ps), "mid": mean(0 < p < 1 for p in ps), "p1": mean(p == 1 for p in ps)}
    return out


def qor(agent_rows, pmap, thr=0.5):
    """Question level over search: share of questions the base model can answer without search
    (p(q) >= thr) on which the policy still searches. Next to it, the search rate on the other questions: a policy
    that knows when to search has qor well below search_rate_rest; one with a fixed habit has the two equal."""
    out = {}
    for ds, v in by_dataset(agent_rows).items():
        elig = [r for r in v if pmap.get(r["id"], 0.0) >= thr]
        rest = [r for r in v if r["id"] in pmap and pmap[r["id"]] < thr]
        out[ds] = {"eligible": len(elig), "qor": mean(r["n_search"] > 0 for r in elig),
                   "search_rate_rest": mean(r["n_search"] > 0 for r in rest),
                   "f1_eligible": mean(r["f1"] for r in elig)}
    return out


def _evidence_complete(rec, titles, text):
    """All gold evidence retrieved: every supporting page title when the set has them (HotpotQA, and 2Wiki and
    MuSiQue when their metadata carries them), otherwise the answer string (NQ, PopQA)."""
    if rec.get("sup_titles"):
        return {norm_title(t) for t in rec["sup_titles"]} <= titles
    if rec["dataset"] == "hotpotqa":
        return False
    return has_answer(text, rec["golds"])


def sor(agent_rows, recs_by_id):
    """Step level over search: share of searches issued after all gold evidence was already retrieved.
    complete: share of episodes whose searches retrieved all gold evidence at some point (a sanity check: near 0 on
    a set with supporting titles means those titles do not match the retrieval corpus)."""
    out = {}
    for ds, v in by_dataset(agent_rows).items():
        over = total = complete = 0
        for r in v:
            rec, titles, text = recs_by_id[r["id"]], set(), ""
            for s in r["searches"]:
                over += _evidence_complete(rec, titles, text)
                total += 1
                titles |= {norm_title(t) for t in s["titles"]}
                text += " " + " ".join(s["texts"])
            complete += bool(r["searches"]) and _evidence_complete(rec, titles, text)
        out[ds] = {"searches": total, "sor": over / total if total else math.nan,
                   "complete": complete / len(v) if v else math.nan}
    return out


def boundary_table(closed_rows, agent_rows, golds_by_id, rule=SAAS_RULE, count="n_search"):
    """Per question knowledge boundary from saved rollouts, with the definitions fixed in rewards.py:
    p(q) = share of search disabled rows whose answer covers a gold answer (p_em is the strict EM version, kept
    for comparison); N_min = fewest searches among search enabled rows that are well formed and have F1 = 1.
    count: the row field that counts an episode's searches. "n_search" counts every call; "n_distinct" counts
    distinct queries, so an episode that repeats one query three times shows the question needs one retrieval.
    Rows come from run_inference.py. Returns {id: {p, p_em, n_min, label, k_closed, k_agent, dataset}}."""
    disabled, disabled_em, enabled, ds_of = (collections.defaultdict(list) for _ in range(4))
    for r in closed_rows:
        disabled[r["id"]].append(knows(r["answer"], golds_by_id[r["id"]]))
        disabled_em[r["id"]].append(r["em"] >= 1.0)
    for r in agent_rows:
        enabled[r["id"]].append((bool(r["format_ok"]) and is_correct(r["f1"]), r[count]))
        ds_of[r["id"]] = r["dataset"]
    out = {}
    for qid in disabled:
        b = estimate_boundary(disabled[qid], enabled.get(qid, []), rule)
        out[qid] = {"p": b.p, "p_em": mean(disabled_em[qid]), "n_min": b.n_min, "label": b.label,
                    "k_closed": len(disabled[qid]), "k_agent": len(enabled.get(qid, [])), "dataset": ds_of.get(qid)}
    return out


def boundary_signal(closed_rows, sampled_agent_rows, golds_by_id, rule=SAAS_RULE, count="n_search", table=None):
    """How many questions can each reward penalise at all, and how often correct searching rollouts get penalised.
    count picks the N_min estimate (see boundary_table); the penalty itself always counts every search call.
    table: a boundary table built elsewhere (e.g. with budget probe rows added), applied to these rollouts."""
    table = table or boundary_table(closed_rows, sampled_agent_rows, golds_by_id, rule, count)
    out = collections.defaultdict(lambda: {"questions": 0, "labels": collections.Counter(), "signal_saas": 0,
                                           "signal_soft": 0, "correct_searching": 0, "pen_saas": [], "pen_soft": []})
    seen = set()
    for r in sampled_agent_rows:
        if r["id"] not in table:
            continue
        t = table[r["id"]]
        b = Boundary(t["p"], t["n_min"], t["label"])
        o = out[r["dataset"]]
        if r["id"] not in seen:
            seen.add(r["id"])
            o["questions"] += 1
            o["labels"][b.label] += 1
            o["signal_saas"] += b.label != "Undetermined"
            o["signal_soft"] += b.p > 0 or b.n_min is not None
        if r["format_ok"] and is_correct(r["f1"]) and r["n_search"] > 0:
            o["correct_searching"] += 1
            o["pen_saas"].append(penalty_saas(r["n_search"], b))
            o["pen_soft"].append(penalty_soft(r["n_search"], b))
    return {ds: {"questions": o["questions"], "labels": dict(o["labels"]),
                 "signal_saas": o["signal_saas"] / max(o["questions"], 1),
                 "signal_soft": o["signal_soft"] / max(o["questions"], 1),
                 "correct_searching_rollouts": o["correct_searching"],
                 "penalised_saas": mean(p > 0 for p in o["pen_saas"]),
                 "penalised_soft": mean(p > 0 for p in o["pen_soft"]),
                 "mean_pen_saas": mean(o["pen_saas"]), "mean_pen_soft": mean(o["pen_soft"])}
            for ds, o in sorted(out.items())}


def seed_averaged_paired(a_runs, b_runs, metric, n_boot=2000):
    """b minus a over several seeds: each question's metric is first averaged over the runs of each side (only
    questions present in every run), then a paired bootstrap over questions. a_runs, b_runs: lists of row lists."""
    def avg(runs):
        d = collections.defaultdict(list)
        for rows in runs:
            for r in rows:
                d[r["id"]].append(r[metric])
        return {k: mean(v) for k, v in d.items() if len(v) == len(runs)}
    a, b = avg(a_runs), avg(b_runs)
    keys = sorted(set(a) & set(b))
    return {**paired_bootstrap([a[k] for k in keys], [b[k] for k in keys], n_boot), "n": len(keys)}


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Mean of b - a over paired items (the same questions), with a 95% bootstrap interval over items."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    return {"diff": mean(d), "lo": means[int(0.025 * n_boot)], "hi": means[int(0.975 * n_boot) - 1]}


def think_closed(text):
    """Did the episode close its first <think> block before its first <search> or <answer> tag?"""
    cut = min([i for i in (text.find("<search>"), text.find("<answer>")) if i >= 0], default=len(text))
    return "</think>" in text[:cut]


def failure_modes(agent_rows):
    """Tallies for the failure mode summary, per dataset: how episodes end, how many searches they use, how often
    a query is empty or repeats an earlier query of the same episode (repeat_query: the redundant search rate),
    how many distinct queries an episode uses, whether it reasons before its first tag, and F1 by search count."""
    out = {}
    for ds, v in by_dataset(agent_rows).items():
        queries = empty = repeat = 0
        distinct = []
        for r in v:
            seen = set()
            for s in r["searches"]:
                q = norm_query(s["query"])
                queries += 1
                empty += not q
                repeat += bool(q) and q in seen
                seen.add(q)
            distinct.append(len(seen))
        buckets = {"0": [r for r in v if r["n_search"] == 0], "1": [r for r in v if r["n_search"] == 1],
                   "2+": [r for r in v if r["n_search"] >= 2]}
        out[ds] = {"n": len(v), "stop": {k: c / len(v) for k, c in sorted(collections.Counter(
                       r["stop_reason"] for r in v).items())},
                   "n_search_share": {k: len(b) / len(v) for k, b in buckets.items()},
                   "f1_by_n_search": {k: mean(r["f1"] for r in b) for k, b in buckets.items() if b},
                   "empty_query": empty / max(queries, 1), "repeat_query": repeat / max(queries, 1),
                   "distinct_queries": mean(distinct),
                   "think_closed": mean(think_closed(r["text"]) for r in v if "text" in r)}
    return out