"""Tests for export_site.py on a tiny fake project with the same folder layout as the real Drive folder."""
import json
import os

import pytest

import export_site as ex

SETS = ex.SETS


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def agent_row(rid, ds, answer, f1, queries, text=None, stop="answer"):
    searches = [{"query": q, "titles": [f"T {q}", "Shared"], "texts": [f"text for {q} " * 3, "same passage"]}
                for q in queries]
    return {"id": rid, "dataset": ds, "split": "test", "sample": 0, "answer": answer, "format_ok": True,
            "em": float(f1 == 1.0), "f1": f1, "n_search": len(queries), "stop_reason": stop,
            "n_distinct": len(set(queries)), "searches": searches,
            "text": text or "<think> x </think>" + "".join(
                f"<search> {q} </search>\n\n<info>\nDoc 1 (Title: T) long passage\n</info>\n\n" for q in queries)
            + f"<answer> {answer} </answer>"}


@pytest.fixture()
def project(tmp_path):
    p = tmp_path / "proj"
    recs = {ds: [{"id": f"{ds}_{i}", "dataset": ds, "split": "test", "question": f"Q{i} of {ds}: who or what?",
                  "golds": [f"gold{i}"]} for i in range(4)] for ds in SETS}
    recs["hotpotqa"][3]["question"] = "Who was born first, Ann Lee or Bo Ray?"
    for ds in SETS:
        write_jsonl(f"{p}/data/{ds}_test.jsonl", recs[ds] + [{"id": f"{ds}_extra", "dataset": ds, "split": "test",
                                                               "question": "not evaluated", "golds": ["x"]}])

    def run(folder, task, fn):  # in domain sets in outputs/<folder>/, the other three in outputs/<folder>_ood/
        by_sub = {}
        for ds in SETS:
            by_sub.setdefault(folder if ds in ex.IN_DOMAIN else folder + "_ood", []).extend(fn(ds, i) for i in range(4))
        for sub, rows in by_sub.items():
            write_jsonl(f"{p}/outputs/{sub}/{task}.jsonl", rows)

    run("stageA2", "agent", lambda ds, i: agent_row(f"{ds}_{i}", ds, f"gold{i}", 1.0, ["q a", "q a"]))
    run("stageA2", "rag", lambda ds, i: agent_row(f"{ds}_{i}", ds, f"gold{i}" if i == 1 else "no", float(i == 1),
                                                  [recs[ds][i]["question"]]))
    run("stageA2", "no_search", lambda ds, i: agent_row(f"{ds}_{i}", ds, f"gold{i}" if i == 0 else "no",
                                                        float(i == 0), []))
    run("base_0.5B", "agent", lambda ds, i: agent_row(f"{ds}_{i}", ds, "no", 0.0, ["b"]))
    run("B_plain_s0", "agent", lambda ds, i: agent_row(f"{ds}_{i}", ds, "no" if i else "gold0", float(i == 0),
                                                       ["Q", "q", "Q"] if i != 2 else ["Q", "r", "s"]))
    run("B_fixed05_s0", "agent", lambda ds, i: agent_row(f"{ds}_{i}", ds, "no", 0.0, []))
    run("B_saas_s0", "agent", lambda ds, i: agent_row(f"{ds}_{i}", ds, f"gold{i}" if i != 1 else "no",
                                                      float(i != 1), ["one"]))
    run("B_soft_s0", "agent", lambda ds, i: agent_row(f"{ds}_{i}", ds, "no", 0.0, ["one"]))
    # p(q): 8 samples per question; question 0 known (6 of 8), the rest unknown
    write_jsonl(f"{p}/outputs/stageA2_pq/closed_test.jsonl",
                [{"id": f"{ds}_{i}", "dataset": ds, "answer": f"gold{i}" if (i == 0 and k < 6) else "no",
                  "em": 0.0, "f1": 0.0, "n_search": 0} for ds in SETS for i in range(4) for k in range(8)])
    return str(p)


def test_model_text_cuts_info_and_caps():
    t = "<think> a </think><search> q </search>\n\n<info>\nDoc 1 long\n</info>\n\n<answer> x </answer>"
    out = ex.model_text(t, 1000)
    assert "Doc 1" not in out and "<info>…</info>" in out and out.endswith("<answer> x </answer>")
    assert ex.model_text("abcdef", 3) == "abc …[cut]"


def test_repeats_counts_normalised_duplicates():
    assert ex.repeats(["A b", "a  B", "c", "A B"]) == 2
    assert ex.repeats(["", ""]) == 0  # empty queries are not repeats


def test_check_against_report_passes_on_exact_values_and_flags_others():
    metrics = {k: {ds: {"em": e / 100, "f1": f / 100, "sc": s} for ds, (e, f, s) in zip(SETS, v)}
               for k, v in ex.REPORT.items()}
    ok, bad = ex.check_against_report(metrics)
    assert ok and bad == {}
    metrics["saas"]["hotpotqa"]["f1"] += 0.002  # 0.2 points off
    ok, bad = ex.check_against_report(metrics)
    assert not ok and list(bad) == ["saas/hotpotqa"]
    del metrics["plain"]["nq"]
    ok, bad = ex.check_against_report(metrics)
    assert "plain/nq" in bad and bad["plain/nq"]["got"] is None


def test_build_writes_consistent_files(project, tmp_path):
    out = str(tmp_path / "site")
    rep = ex.build(project, out)
    assert rep["questions"] == {ds: 4 for ds in SETS}  # the not evaluated extra question is dropped
    assert all(v == 0 for m in rep["missing"].values() for v in m.values())
    assert rep["known"] == {ds: 1 for ds in SETS}
    assert rep["report_match"] is False and rep["known_counts_match"] is False  # fake numbers
    meta = json.load(open(f"{out}/meta.json"))
    assert [p["key"] for p in meta["policies"]] == [p[0] for p in ex.POLICIES]
    m = meta["metrics"]
    assert m["plain"]["hotpotqa"]["sc"] == 3 and m["fixed"]["nq"]["sc"] == 0
    assert m["start_agent"]["hotpotqa"]["repeat"] == 0.5  # every second query repeats the first
    assert m["saas"]["hotpotqa"]["f1"] == 0.75 and m["closed"]["hotpotqa"]["f1"] == 0.25
    assert m["saas"]["hotpotqa"]["search_known"] == 1.0  # searches the known question
    for ds in SETS:
        d = json.load(open(f"{out}/q_{ds}.json"))
        assert [q["id"] for q in d["questions"]] == [f"{ds}_{i}" for i in range(4)]
        q0 = d["questions"][0]
        assert q0["p"] == 0.75 and q0["cb"] == "gold0" and q0["cb_ok"] is True
        assert d["questions"][1]["cb_ok"] is False
        n = len(d["passages"])
        for k, rows in d["runs"].items():
            assert len(rows) == 4
            for r in rows:
                assert all(0 <= p < n for s in r["s"] for p in s[1])
                assert "long passage" not in r["t"]
        # "Shared" passage appears once in the table although every search retrieves it
        assert sum(1 for t, _ in d["passages"] if t == "Shared") == 1
        plain0 = d["runs"]["plain"][0]
        assert [s[0] for s in plain0["s"]] == ["Q", "q", "Q"] and plain0["n"] == 3


def test_showcase_rules_pick_expected_questions(project, tmp_path):
    rep = ex.build(project, str(tmp_path / "site"))
    kinds = {}
    for i, k in rep["showcase"]:
        kinds.setdefault(k, []).append(i)
    assert kinds == {"repeats": ["hotpotqa_0"],               # plain repeats "Q"; SAAS one search, correct
                     "known": ["2wiki_0", "musique_0", "nq_0"],  # p = 0.75 and closed book right; spread over sets
                     "fixed": ["hotpotqa_2"],                 # hotpotqa_0 is taken already
                     "comparison": ["hotpotqa_3"],
                     "rag": ["hotpotqa_1"]}
    ids = [i for i, _ in rep["showcase"]]
    assert len(ids) == len(set(ids))


def test_limit_keeps_valid_passage_indices_and_showcase(project, tmp_path):
    out = str(tmp_path / "dry")
    rep = ex.build(project, out, limit=2)
    meta = json.load(open(f"{out}/meta.json"))
    kept = set()
    for ds in SETS:
        d = json.load(open(f"{out}/q_{ds}.json"))
        assert len(d["questions"]) == 2
        kept |= {q["id"] for q in d["questions"]}
        n = len(d["passages"])
        used = {p for rows in d["runs"].values() for r in rows for s in r["s"] for p in s[1]}
        assert used == set(range(n))  # every kept passage is used, every index valid
    assert all(s["id"] in kept for s in meta["showcase"])
    assert meta["metrics"]["plain"]["hotpotqa"]["n"] == 4  # metrics still use every row
    assert rep["questions"]["hotpotqa"] == 4
