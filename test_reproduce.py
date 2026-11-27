"""Tests for reproduce.py: python -m pytest -q test_reproduce.py (CPU only, no model or retriever needed)."""
import json
import os

import pytest

import reproduce as rp
from sot.scoring import em, f1

IN_DOMAIN = {"hotpotqa", "nq"}


def test_plan_follows_the_original_evaluation():
    jobs = rp.plan(list(rp.MODELS))
    assert jobs[:3] == [("base_0.5B", ["agent"], False), ("base_0.5B", ["no_search", "rag"], False),
                        ("base_0.5B_ood", ["agent", "no_search", "rag"], True)]
    assert ("B_saas_s0_ood", ["agent"], True) in jobs and len(jobs) == 14
    cmd = rp.command("/w", "/w/checkpoints/B_saas_s0/final", "B_saas_s0_ood", ["agent"], True)
    assert cmd[1:] == ["run_inference.py", "--project", "/w", "--n", "1000", "--n_nq", "500", "--model",
                       "/w/checkpoints/B_saas_s0/final", "--name", "B_saas_s0_ood", "--tasks", "agent",
                       "--dtype", "bfloat16", "--datasets", "2wiki,musique,popqa"]
    assert rp.command("/w", "m", "stageA2", ["agent"], False, "float16")[-2:] == ["--dtype", "float16"]
    assert "--datasets" not in rp.command("/w", "m", "stageA2", ["no_search", "rag"], False)


def test_every_report_row_has_a_model_and_five_sets():
    assert {r[2] for r in rp.ROWS} == set(rp.MODELS)
    assert all(len(rp.REPORT[r[0]]) == len(rp.SETS) for r in rp.ROWS)


def test_passage_key_matches_the_demo_data():
    assert rp.passage_key("T", "a" * 500) == ["T", "a" * 500]
    assert rp.passage_key("T", "a" * 501) == ["T", "a" * 500 + "…"]


def test_recorded_searches_cover_every_recorded_query():
    rec = rp.recorded_searches()
    assert len(rec) > 10000 and all(len(p) <= 3 for p in rec.values())


def test_find_model_dir_skips_macos_copies(tmp_path):
    (tmp_path / "__MACOSX" / "final").mkdir(parents=True)
    (tmp_path / "__MACOSX" / "final" / "config.json").write_text("{}")
    (tmp_path / "__MACOSX" / "final" / "model.safetensors").write_text("")
    assert rp.find_model_dir(str(tmp_path)) is None
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "config.json").write_text("{}")
    (tmp_path / "final" / "model.safetensors").write_text("")
    assert rp.find_model_dir(str(tmp_path)) == str(tmp_path / "final")


def _write(work, folder, task, rows):
    for ood in (False, True):
        part = [r for r in rows if (r["dataset"] in IN_DOMAIN) != ood]
        d = f"{work}/outputs/{folder}{'_ood' if ood else ''}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/{task}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in part)


def _recorded_work(work):
    """An outputs folder holding exactly the recorded runs: answers from docs/data, EM and F1 rescored from them."""
    golds = {}
    for ds in rp.SETS:
        golds.update({q["id"]: (ds, q["golds"]) for q in rp.demo_data(ds)["questions"]})
    for key, _, folder, task, rec_key in rp.ROWS:
        if rec_key:
            rows = [{"id": i, "dataset": golds[i][0], "answer": a, "n_search": n, "format_ok": True,
                     "em": em(a, golds[i][1]), "f1": f1(a, golds[i][1])}
                    for i, (a, n) in rp.recorded_rows(rec_key).items()]
        else:  # rows with exactly the report's EM / F1 / SC
            rows = []
            for ds, (e_em, e_f1, e_sc) in zip(rp.SETS, rp.REPORT[key]):
                n = rp.EXPECTED_N[ds]
                k = round(e_em * n / 100)
                rows += [{"id": f"{ds}_{j}", "dataset": ds, "answer": "x", "n_search": round(e_sc),
                          "format_ok": True, "em": float(j < k), "f1": e_f1 / 100} for j in range(n)]
        _write(work, folder, task, rows)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    w = str(tmp_path_factory.mktemp("work"))
    _recorded_work(w)
    return w


def test_recorded_runs_are_the_same_and_match_the_report(work):
    results = rp.compare(work, n_boot=200)
    assert {r["status"] for r in results} == {"same"}
    assert rp.verdict(results) == "PASS"
    for r in results:  # rescored answers land on the report's appendix table, within its rounding (0.06)
        assert all(abs(g - e) <= t for g, e, t in zip(r["got"], r["report"], (0.06, 0.06, 0.006))), r
        assert r.get("f1_diff", [0, 0, 0]) == [0, 0, 0]


def _edit(work, folder, task, ds, n_changed):
    path = f"{work}/outputs/{folder}{'' if ds in IN_DOMAIN else '_ood'}/{task}.jsonl"
    rows = [json.loads(x) for x in open(path)]
    original = [dict(r) for r in rows]
    done = 0
    for r in rows:
        if r["dataset"] == ds and done < n_changed:
            r.update(answer="zzz", em=0.0, f1=0.0)
            done += 1
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path, original


def _status(results, row, ds):
    return next(r["status"] for r in results if r["row"] == row and r["set"] == ds)


def test_noise_sized_changes_pass_and_real_changes_fail(work):
    path, original = _edit(work, "B_saas_s0", "agent", "hotpotqa", 30)
    try:
        results = rp.compare(work, n_boot=200)
        r = next(x for x in results if x["row"] == "saas" and x["set"] == "hotpotqa")
        assert r["same_answers"] == 970 and r["f1_diff"][0] < 0 and r["status"] == "same"
        assert rp.verdict(results) == "PASS"
        _edit(work, "B_saas_s0", "agent", "hotpotqa", 300)
        results = rp.compare(work, n_boot=200)
        assert _status(results, "saas", "hotpotqa") == "differs" and rp.verdict(results) == "FAIL"
    finally:
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in original)


def test_search_count_and_retrieval_gates():
    res = {"got": [20.0, 27.0, 1.2], "report": [20.0, 27.0, 1.0], "recorded": [20.0, 27.0, 1.0],
           "f1_diff": [0.0, -0.5, 0.5]}
    assert rp.judge(res) == "differs"
    res["got"][2] = 1.05
    assert rp.judge(res) == "same"
    table = {"got": [7.7, 14.2, 0.0], "report": [7.3, 13.9, 0.0]}
    assert rp.judge(table) == "same" and rp.judge({**table, "got": [7.7, 15.2, 0.0]}) == "differs"
    ok = [{"status": "same"}]
    assert rp.verdict(ok, {"agreement": 1.0}) == "PASS" and rp.verdict(ok, {"agreement": 0.98}) == "FAIL"


def test_noise_check_compares_two_runs_of_the_start_policy(tmp_path, monkeypatch):
    first = [{"id": f"h{i}", "dataset": "hotpotqa", "answer": "a", "n_search": 2, "f1": 1.0, "format_ok": True}
             for i in range(10)]
    first += [{"id": f"n{i}", "dataset": "nq", "answer": "b", "n_search": 1, "f1": 0.0, "format_ok": True}
              for i in range(4)]
    second = [dict(r, answer="c", f1=0.0) if r["id"] == "h0" else r for r in first]
    for name, rows in (("stageA2", first), ("stageA2_rerun", second)):
        os.makedirs(tmp_path / "outputs" / name)
        with open(tmp_path / "outputs" / name / "agent.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    calls = []
    monkeypatch.setattr(rp, "run_inference", lambda *a: calls.append(a[1:4] + a[6:]))
    out = rp.noise_check(str(tmp_path), "ckpt", False, "float16", n_boot=200)
    assert calls == [("stageA2_rerun", "ckpt", ["agent"], "float16")]
    assert rp.format_share(str(tmp_path), "stageA2") == 1.0 and rp.format_share(str(tmp_path), "none") is None
    assert out["hotpotqa"]["same_answers"] == 9 and out["hotpotqa"]["f1_diff"][0] == -10.0
    assert out["nq"] == {"same_answers": 4, "n": 4, "f1_diff": [0.0, 0.0, 0.0]}


def test_missing_outputs_fail_and_unselected_policies_are_not_run(work, tmp_path):
    assert rp.verdict(rp.compare(str(tmp_path), n_boot=200)) == "FAIL"
    results = rp.compare(work, models=("B_saas_s0",), n_boot=200)
    assert {r["status"] for r in results if r["row"] != "saas"} == {"not run"}
    assert rp.verdict(results) == "PASS (45 of 50 results not run)"


def test_checkpoint_download_is_verified(tmp_path, monkeypatch):
    import zipfile
    rel = tmp_path / "rel"
    rel.mkdir()
    with zipfile.ZipFile(rel / "B_saas_s0.zip", "w") as z:
        z.writestr("final/config.json", "{}")
        z.writestr("final/model.safetensors", "weights")
    work = tmp_path / "work"
    monkeypatch.setitem(rp.CHECKSUMS, "B_saas_s0", rp.sha256(str(rel / "B_saas_s0.zip")))
    assert rp.fetch_checkpoint("B_saas_s0", str(work), f"file://{rel}").endswith("B_saas_s0/final")
    monkeypatch.setitem(rp.CHECKSUMS, "B_soft_s0", "0" * 64)
    (rel / "B_soft_s0.zip").write_bytes((rel / "B_saas_s0.zip").read_bytes())
    with pytest.raises(RuntimeError, match="SHA-256"):
        rp.fetch_checkpoint("B_soft_s0", str(work), f"file://{rel}")


def test_download_resumes_after_a_dropped_connection(tmp_path):
    import http.server
    import threading
    payload = os.urandom(300_000)
    requests = []

    class Flaky(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
            requests.append(start)
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            body = payload[start:]
            self.wfile.write(body[:100_000] if len(requests) == 1 else body)  # the first response is cut short

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        path = str(tmp_path / "x.zip")
        rp.download(f"http://127.0.0.1:{srv.server_port}/x.zip", path)
        assert open(path, "rb").read() == payload and requests == [0, 100_000]
    finally:
        srv.shutdown()
