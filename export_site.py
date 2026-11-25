"""Exports the recorded evaluation trajectories for the static demo page (docs/ on GitHub Pages).

Reads only: data/<set>_test.jsonl and outputs/<folder>/<task>.jsonl (with outputs/<folder>_ood/ for the three out
of domain sets), exactly the files the evaluation notebook read. Nothing is rerun. Writes:
  <out>/meta.json        policies, per policy metrics on every test set, the check against the report, showcase
  <out>/q_<set>.json     one file per test set: questions, a shared passage table, and every policy's trajectory

usage: python export_site.py --project P --out DIR [--limit N]
--limit keeps the first N questions per set in the q_ files (dry runs); metrics and checks always use every row.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sot.analysis import by_dataset, failure_modes, load_jsonl, mean, qor, summarize
from sot.scoring import knows, norm_query

SETS = ["hotpotqa", "2wiki", "musique", "nq", "popqa"]
SET_LABELS = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMultihopQA", "musique": "MuSiQue", "nq": "NQ", "popqa": "PopQA"}
IN_DOMAIN = {"hotpotqa", "nq"}

# key, outputs folder, task file, label, description. The four trained rewards all branch from the start policy.
POLICIES = [
    ("base_agent", "base_0.5B", "agent", "Untrained agent",
     "Qwen2.5-0.5B-Instruct prompted to use the search tags, before any RL"),
    ("start_agent", "stageA2", "agent", "Start policy",
     "Plain GRPO with forced exploration for 200 steps (stage A2). Every run below branches from it"),
    ("start_rag", "stageA2", "rag", "One search (RAG)",
     "The start policy given exactly one search, with the question itself as the query"),
    ("plain", "B_plain_s0", "agent", "Plain GRPO",
     "Reward: answer F1. 100 more steps from the start policy, seed 0"),
    ("fixed", "B_fixed05_s0", "agent", "Fixed penalty",
     "Reward: F1 minus 0.05 per search, on every rollout"),
    ("saas", "B_saas_s0", "agent", "SAAS",
     "Penalises searches its hard knowledge boundary labels call unnecessary, only when the answer is correct"),
    ("soft", "B_soft_s0", "agent", "Soft-SAAS (ours)",
     "SAAS with the boundary weighted by p(q), the start policy's closed book accuracy, seed 0"),
]
CLOSED = ("stageA2", "no_search")  # the start policy without search: shown as "closed book" on each question
PQ_FOLDER = "stageA2_pq"           # 8 closed book samples per test question: p(q), as in the report
KNOWN = 0.5                        # p(q) >= 0.5 counts as known, as for QOR in the report

# Report appendix (Table: every run), EM / F1 in % and SC, per set in SETS order. The export must reproduce these.
REPORT = {
    "base_agent": [(6.6, 13.9, 0.79), (7.4, 18.0, 0.96), (1.2, 4.8, 0.85), (10.6, 16.6, 0.75), (16.6, 21.2, 0.96)],
    "start_agent": [(15.1, 22.6, 2.54), (18.2, 23.1, 2.64), (2.2, 7.5, 2.61), (18.2, 25.0, 2.58), (24.0, 28.0, 2.57)],
    "start_rag": [(20.0, 28.7, 1.00), (21.8, 27.7, 1.00), (3.4, 9.1, 1.00), (17.2, 24.1, 1.00), (27.8, 30.9, 1.00)],
    "plain": [(19.3, 26.5, 3.00), (23.8, 27.8, 3.00), (3.6, 7.8, 3.00), (19.4, 28.1, 2.99), (25.4, 28.0, 3.00)],
    "fixed": [(10.6, 18.5, 0.00), (18.7, 23.8, 0.00), (1.8, 7.4, 0.00), (5.4, 12.0, 0.00), (11.0, 14.2, 0.00)],
    "saas": [(18.9, 26.9, 1.00), (24.5, 29.2, 1.00), (2.4, 7.4, 1.00), (18.8, 26.9, 1.00), (27.6, 31.3, 1.00)],
    "soft": [(19.6, 27.5, 1.00), (21.6, 27.2, 1.00), (2.2, 7.7, 1.00), (17.2, 25.1, 1.00), (24.4, 27.1, 1.00)],
    "closed": [(10.0, 17.7, 0.00), (16.8, 22.0, 0.00), (0.8, 6.5, 0.00), (7.6, 13.9, 0.00), (8.2, 14.3, 0.00)],
}
REPORT_KNOWN = {"hotpotqa": 102, "2wiki": 180, "nq": 31, "popqa": 57}  # report Table 4 (MuSiQue not reported)

INFO_RE = re.compile(r"<info>.*?</info>", re.S)


def rows_of(out, folder, task):
    rows = []
    for f in (folder, folder + "_ood"):
        path = f"{out}/{f}/{task}.jsonl"
        if os.path.exists(path):
            rows += load_jsonl(path)
    return rows


def model_text(text, cap):
    """What the model wrote: the environment's <info> blocks are cut (their passages are shown separately)."""
    t = INFO_RE.sub("<info>…</info>", text or "")
    return t if len(t) <= cap else t[:cap] + " …[cut]"


def repeats(queries):
    """How many queries repeat an earlier one of the same episode (case and spacing ignored)."""
    seen, n = set(), 0
    for q in map(norm_query, queries):
        n += bool(q) and q in seen
        seen.add(q)
    return n


def check_against_report(metrics):
    out, ok = {}, True
    for key, exp in REPORT.items():
        for ds, (e_em, e_f1, e_sc) in zip(SETS, exp):
            m = metrics.get(key, {}).get(ds)
            got = None if m is None else (100 * m["em"], 100 * m["f1"], m["sc"])
            # unrounded values against the report's rounded ones: within rounding (0.06 allows double rounding,
            # e.g. 22.545 -> 0.2255 -> 22.6 in the report)
            match = got is not None and abs(got[0] - e_em) <= 0.06 and abs(got[1] - e_f1) <= 0.06 \
                and abs(got[2] - e_sc) <= 0.006
            got = got and (round(got[0], 2), round(got[1], 2), round(got[2], 3))
            ok &= match
            if not match:
                out[f"{key}/{ds}"] = {"got": got, "report": (e_em, e_f1, e_sc)}
    return ok, out


def pick_showcase(qs, runs, per_kind=3):
    """Questions that show the report's findings, chosen by fixed rules from the recorded rows (test file order)."""
    picks, used = [], set()

    def take(kind, why, cond, sets=None, k=per_kind):
        n = 0
        for ds in (sets or SETS):
            for i, q in enumerate(qs[ds]):
                if n >= k:
                    return
                r = {key: runs[key][ds][i] for key in runs}
                if q["id"] in used or any(v is None for v in r.values()):
                    continue
                if cond(q, r):
                    picks.append({"id": q["id"], "set": ds, "kind": kind, "why": why})
                    used.add(q["id"])
                    n += 1
                    if sets is None:  # spread over the sets
                        break

    trained = ("plain", "fixed", "saas", "soft")
    take("repeats", "Plain GRPO repeats a query; SAAS searches once and gets it right",
         lambda q, r: repeats([x[0] for x in r["plain"]["s"]]) >= 1 and r["saas"]["n"] == 1 and r["saas"]["f1"] == 1.0,
         sets=["hotpotqa", "nq", "2wiki"], k=1)
    take("known", "The start policy knows this closed book, yet every trained policy still searches",
         lambda q, r: (q["p"] or 0) >= 0.75 and q["cb_ok"] and all(r[k]["n"] >= 1 for k in ("plain", "saas", "soft"))
         and min(r[k]["f1"] for k in ("plain", "saas", "soft")) < 0.5)
    take("fixed", "The fixed penalty never searches and misses; SAAS finds it with one search",
         lambda q, r: r["fixed"]["n"] == 0 and r["fixed"]["f1"] == 0 and r["saas"]["n"] == 1 and r["saas"]["f1"] == 1.0,
         sets=["hotpotqa", "popqa"], k=1)
    take("comparison", "A comparison question: one search for the whole question, often naming the wrong one",
         lambda q, r: bool(re.search(r"\bor\b", q["q"])) and bool(re.search(
             r"\b(first|earlier|older|younger|more recently|longer|larger)\b", q["q"])) and r["plain"]["n"] == 3,
         sets=["hotpotqa"], k=2)
    take("rag", "One search with the question beats every trained agent",
         lambda q, r: r["start_rag"]["f1"] == 1.0 and all(r[k]["f1"] == 0 for k in trained),
         sets=["hotpotqa", "popqa"], k=1)
    return picks


def build(project, out_dir, limit=None, passage_cap=500, text_cap=1500):
    data, out = f"{project}/data", f"{project}/outputs"
    recs = {}
    order = {}
    for ds in SETS:
        path = f"{data}/{ds}_test.jsonl"
        order[ds] = []
        for r in (load_jsonl(path) if os.path.exists(path) else []):
            recs[r["id"]] = r
            order[ds].append(r["id"])

    all_rows = {key: rows_of(out, folder, task) for key, folder, task, _, _ in POLICIES}
    all_rows["closed"] = rows_of(out, *CLOSED)
    by_id = {key: {r["id"]: r for r in rows} for key, rows in all_rows.items()}

    pq = {}
    for r in rows_of(out, PQ_FOLDER, "closed_test"):
        if r["id"] in recs:
            pq.setdefault(r["id"], []).append(knows(r["answer"], recs[r["id"]]["golds"]))
    pmap = {k: mean(v) for k, v in pq.items()}

    # Per policy metrics on every row (as in the report), and the check that they reproduce the report.
    metrics = {}
    for key, rows in all_rows.items():
        if not rows:
            continue
        fm = failure_modes(rows) if key != "closed" else {}
        qo = qor(rows, pmap, KNOWN) if pmap else {}
        metrics[key] = {ds: {"n": m["n"], "em": m["em"], "f1": m["f1"], "sc": m["searches"],
                             "format": m["format_ok"],
                             "repeat": fm.get(ds, {}).get("repeat_query"),
                             "distinct": fm.get(ds, {}).get("distinct_queries"),
                             "search_known": qo.get(ds, {}).get("qor"),
                             "search_rest": qo.get(ds, {}).get("search_rate_rest")}
                        for ds, m in summarize(rows).items()}
    report_ok, mismatches = check_against_report(metrics)
    known = {ds: sum(pmap.get(i, 0) >= KNOWN for i in order[ds] if i in pmap) for ds in SETS}
    known_ok = all(known.get(ds) == n for ds, n in REPORT_KNOWN.items())

    # Questions in test file order that every shown policy answered (the reference is the start policy).
    keys = [p[0] for p in POLICIES]
    qs, runs, passages, missing = {}, {k: {} for k in keys}, {}, {}
    os.makedirs(out_dir, exist_ok=True)
    for ds in SETS:
        ids = [i for i in order[ds] if i in by_id["start_agent"]]
        missing[ds] = {k: sum(i not in by_id[k] for i in ids) for k in keys + ["closed"]}
        missing[ds]["no_p"] = sum(i not in pmap for i in ids)
        table, index = [], {}

        def pid(title, text):
            k = (title, text[:passage_cap])
            if k not in index:
                index[k] = len(table)
                table.append([k[0], k[1] + ("…" if len(text) > passage_cap else "")])
            return index[k]

        qs[ds] = []
        for i in ids:
            rec, cb = recs[i], by_id["closed"].get(i)
            qs[ds].append({"id": i, "q": rec["question"], "golds": rec["golds"],
                           "p": round(pmap[i], 3) if i in pmap else None,
                           "cb": cb["answer"] if cb else None, "cb_ok": bool(cb) and knows(cb["answer"], rec["golds"]),
                           "cb_f1": round(cb["f1"], 3) if cb else None})
        for k in keys:
            runs[k][ds] = []
            for i in ids:
                r = by_id[k].get(i)
                if r is None:
                    runs[k][ds].append(None)
                    continue
                runs[k][ds].append({
                    "a": r["answer"], "f1": round(r["f1"], 3), "em": r["em"], "n": r["n_search"],
                    "stop": r["stop_reason"], "ok": r["format_ok"],
                    "s": [[s["query"], [pid(t, x) for t, x in zip(s.get("titles", []), s.get("texts", []))]]
                          for s in r.get("searches", [])],
                    "t": model_text(r.get("text", ""), text_cap)})
        passages[ds] = table

    showcase = pick_showcase(qs, runs)
    sizes = {}
    for ds in SETS:
        n = len(qs[ds]) if limit is None else min(limit, len(qs[ds]))
        payload = {"set": ds, "questions": qs[ds][:n], "runs": {k: runs[k][ds][:n] for k in keys}}
        if limit is not None:  # keep only the passages the cut questions use
            used = sorted({p for k in keys for r in payload["runs"][k] if r for s in r["s"] for p in s[1]})
            remap = {p: j for j, p in enumerate(used)}
            for k in keys:
                for r in payload["runs"][k]:
                    if r:
                        r["s"] = [[s[0], [remap[p] for p in s[1]]] for s in r["s"]]
            payload["passages"] = [passages[ds][p] for p in used]
        else:
            payload["passages"] = passages[ds]
        path = f"{out_dir}/q_{ds}.json"
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        sizes[ds] = round(os.path.getsize(path) / 1e6, 2)

    if limit is not None:  # showcase must point at questions that are in the cut files
        kept = {q["id"] for ds in SETS for q in qs[ds][:limit]}
        showcase = [s for s in showcase if s["id"] in kept]
    meta = {
        "title": "Search or Think? Soft Knowledge Boundaries for Cost-Aware Retrieval in Small Language Models",
        "model": "Qwen2.5-0.5B-Instruct",
        "sets": [{"key": ds, "label": SET_LABELS[ds], "in_domain": ds in IN_DOMAIN,
                  "n": len(qs[ds]) if limit is None else min(limit, len(qs[ds])), "known": known.get(ds)} for ds in SETS],
        "policies": [{"key": k, "label": lab, "desc": desc, "folder": folder, "task": task}
                     for k, folder, task, lab, desc in POLICIES],
        "metrics": {k: {ds: {m: (round(v, 4) if isinstance(v, float) else v) for m, v in d.items()}
                        for ds, d in v.items()} for k, v in metrics.items()},
        "known_threshold": KNOWN,
        "showcase": showcase,
        "check": {"report_match": report_ok, "known_counts_match": known_ok},
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    sizes["meta"] = round(os.path.getsize(f"{out_dir}/meta.json") / 1e6, 3)

    return {"questions": {ds: len(qs[ds]) for ds in SETS}, "missing": missing, "known": known,
            "report_match": report_ok, "mismatches": mismatches, "known_counts_match": known_ok,
            "passages": {ds: len(passages[ds]) for ds in SETS}, "mb": sizes,
            "showcase": [(s["id"], s["kind"]) for s in showcase]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--passage_cap", type=int, default=500, help="characters kept per passage")
    ap.add_argument("--text_cap", type=int, default=1500, help="characters kept of what the model wrote")
    a = ap.parse_args()
    print(json.dumps(build(a.project, a.out, a.limit, a.passage_cap, a.text_cap)))


if __name__ == "__main__":
    main()
