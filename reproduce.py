"""Reproduces the report's evaluation results from the trained checkpoints. No training is run.

Every number in the report was produced on an NVIDIA L4 with vLLM 0.31.0 and TRL 1.16.0, greedy decoding, bfloat16.
notebooks/10_reproduce_results.ipynb installs the same versions in Colab and runs this script. It also runs on the
free T4 GPU, which has no bfloat16: the policies then run in float16, and more answers differ from the recorded run.

Steps
  1. environment   GPU and library versions.
  2. unit tests    test_sot.py, test_export_site.py, test_reproduce.py: scoring, rewards, agent loop, analysis,
                   export, and this script's comparison.
  3. checkpoints   the base model from the Hugging Face Hub, the five trained policies from this repo's release.
  4. vLLM check    loads the model and answers one prompt in bfloat16 where the GPU supports it (compute
                   capability 8.0 and up), else float16, else float32 (--dtype forces one). Fails within minutes
                   rather than after the search index is set up.
  5. retriever     setup_retriever.sh starts the BM25 server over Pyserini's prebuilt wikipedia-dpr-100w index.
                   Then queries that the recorded runs issued are searched again and the top 3 passages are compared
                   with the passages recorded at evaluation time.
  6. inference     run_inference.py with the same commands, task grouping and order as the original evaluation
                   (notebooks 05 to 08).
  7. noise         the start policy's in domain evaluation runs a second time, to measure how much two runs of the
                   same command differ (vLLM's greedy decoding is not bit for bit repeatable, see below).
  8. comparison    every policy and test set against the recorded run (docs/data, the demo page's data) with a
                   paired bootstrap over the same questions, as in the report; the two baselines without recorded
                   answers against the report's appendix table. Writes <work>/reproduce_report.json (also when a
                   step stops the run early) and prints PASS or FAIL.

usage: python reproduce.py [--work /content/repro] [--policies base_0.5B,stageA2,...] [--compare_only]
"""
import argparse
import functools
import hashlib
import http.client
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import export_site as ex  # noqa: E402
from sot.analysis import by_dataset, load_jsonl, mean, paired_bootstrap  # noqa: E402
from sot.scoring import f1  # noqa: E402

RELEASE = "https://github.com/Kodezilla0725/search_or_think/releases/download/checkpoints-v1"
# SHA-256 of each release zip, taken on the uploader's machine before upload: a download that differs is refused.
CHECKSUMS = {
    "stageA2": "2e8a66ea30fa4c23b8cd352b960aead8bfb5acd960ce1ca2f9949242e12943df",
    "B_plain_s0": "8de78e27842463db1d6c51f858f1dfdfd0c44dc4c5342e80c39809a4831576b0",
    "B_fixed05_s0": "be8843b9b47f59cd2a7cfa6dbe26b89f81437ca627b97dfc3b6c62170e13e004",
    "B_saas_s0": "b79fab170f0bae19371a5324bf88739eb15ba81c7a0ed0095e04923ed720ebc2",
    "B_soft_s0": "5f11556ecdf0b16b1520d8e88747b3186164e1546f95af76d2be416e72021873",
}
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SETS = ex.SETS
OOD = "2wiki,musique,popqa"
INF = ["--n", "1000", "--n_nq", "500"]  # HotpotQA 1000, NQ the first 500; the out of domain sets whole
EXPECTED_N = {"hotpotqa": 1000, "2wiki": 1000, "musique": 500, "nq": 500, "popqa": 500}
ORIGINAL = {"gpu": "NVIDIA L4", "vllm": "0.31.0", "trl": "1.16.0"}
TESTS = ["test_sot.py", "test_export_site.py", "test_reproduce.py"]

# Output folder: (where the weights come from, in domain task groups, out of domain task groups). Each group is one
# run_inference.py process, as in the original evaluation: the in domain agent run came first in its own process
# (notebooks 05 and 06), no_search and rag followed together (notebook 07), and the out of domain sets ran all of a
# policy's tasks in one process (notebook 08).
MODELS = {
    "base_0.5B": ("hub", [["agent"], ["no_search", "rag"]], [["agent", "no_search", "rag"]]),
    "stageA2": ("release", [["agent"], ["no_search", "rag"]], [["agent", "no_search", "rag"]]),
    "B_plain_s0": ("release", [["agent"]], [["agent"]]),
    "B_fixed05_s0": ("release", [["agent"]], [["agent"]]),
    "B_saas_s0": ("release", [["agent"]], [["agent"]]),
    "B_soft_s0": ("release", [["agent"]], [["agent"]]),
}

# Report appendix rows: key, label, output folder, task, key of the recorded run in docs/data (None: not recorded
# there, so only the report's table is checked).
ROWS = [
    ("base_closed", "base, no search", "base_0.5B", "no_search", None),
    ("base_rag", "base, RAG", "base_0.5B", "rag", None),
    ("base_agent", "base, agent", "base_0.5B", "agent", "base_agent"),
    ("closed", "start, no search", "stageA2", "no_search", "closed"),
    ("start_rag", "start, RAG", "stageA2", "rag", "start_rag"),
    ("start_agent", "start, agent", "stageA2", "agent", "start_agent"),
    ("plain", "plain s0", "B_plain_s0", "agent", "plain"),
    ("fixed", "fixed 0.05", "B_fixed05_s0", "agent", "fixed"),
    ("saas", "SAAS s0", "B_saas_s0", "agent", "saas"),
    ("soft", "Soft-SAAS s0", "B_soft_s0", "agent", "soft"),
]
# EM / F1 in % and SC per set (SETS order), from the report's appendix table. The other rows are in export_site.
REPORT = {
    **ex.REPORT,
    "base_closed": [(7.3, 13.9, 0.00), (17.0, 22.7, 0.00), (0.6, 7.1, 0.00), (3.4, 8.4, 0.00), (9.0, 13.2, 0.00)],
    "base_rag": [(15.9, 24.1, 1.00), (22.9, 28.2, 1.00), (2.0, 7.2, 1.00), (10.8, 17.6, 1.00), (23.2, 27.0, 1.00)],
}

# How a result is judged, per policy and test set. vLLM does not make greedy decoding bit for bit repeatable by
# default (https://docs.vllm.ai/en/v0.31.0/usage/reproducibility/): batches are scheduled differently from run to
# run, a few uncertain answers change, and a changed search query changes the rest of its episode. The original
# evaluation ran the same way, so identical answers are not the test. The test is whether the numbers are the same:
#   same:    the paired bootstrap 95% interval of the F1 difference (reproduced minus recorded, over the same
#            questions, 2,000 resamples, as in the report) lies within MARGIN_F1 points, and SC is within MARGIN_SC.
#            Two runs of the same command on the same L4 gave intervals reaching about 2 points (start policy, NQ),
#            so 2.5 allows that noise; seeds of the same reward differ by up to 4 F1 points.
#            The two base model baselines without recorded answers here (no search, RAG) need EM and F1 within
#            MARGIN_TABLE points of the report's table instead.
#   differs: anything else. A "differs" or missing result, or a retrieval check below MIN_RETRIEVAL, is a FAIL.
MARGIN_F1, MARGIN_SC, MARGIN_TABLE, MIN_RETRIEVAL = 2.5, 0.1, 1.0, 0.99


def log(msg):
    print(f"[reproduce] {msg}", flush=True)


# 1. environment
def environment():
    info = {}
    for lib in ("vllm", "trl", "transformers", "torch", "pyserini"):
        try:
            info[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            info[lib] = None
    try:
        import torch
        cuda = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if cuda else None
        # vLLM runs bfloat16 only from compute capability 8.0 (A100, L4). torch.cuda.is_bf16_supported() is no
        # test for that: it also counts the T4's slow emulation and returns True there.
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0))) if cuda else None
        info["bf16"] = bool(cuda and torch.cuda.get_device_capability(0) >= (8, 0))
    except ImportError:
        info["gpu"], info["bf16"] = None, False
    info["warnings"] = [f"{k} is {info.get(k)}, the original evaluation used {v}"
                        for k, v in ORIGINAL.items() if info.get(k) != v]
    return info


# 2. unit tests
def unit_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS], cwd=REPO, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    return {"passed": r.returncode == 0, "summary": tail[0]}


# 5. retriever (runs after steps 3 and 4)
def start_retriever(work):
    # shortcut: setup_retriever.sh is the Colab script the evaluation used (Java 21, a pyserini venv and the index
    # under /content); it is reused unchanged rather than generalised to other machines.
    subprocess.run(["bash", "setup_retriever.sh", work, "0", REPO], cwd=REPO, check=True)


def passage_key(title, text, cap=500):
    """A passage as the demo data stores it (export_site.build): the title and the first 500 characters."""
    return [title, text[:cap] + ("…" if len(text) > cap else "")]


@functools.lru_cache(maxsize=None)
def demo_data(ds):
    with open(f"{REPO}/docs/data/q_{ds}.json") as f:
        return json.load(f)


def recorded_searches():
    """Every distinct query in the recorded runs, with the top 3 passages it retrieved at evaluation time."""
    out = {}
    for ds in SETS:
        q = demo_data(ds)
        table = q["passages"]
        for rows in q["runs"].values():
            for r in rows:
                for query, pids in (r or {}).get("s", []):
                    out.setdefault(query, [table[p] for p in pids])
    return out


def retrieval_check(n):
    from sot.retrieval import Retriever
    rec = recorded_searches()
    queries = sorted(rec)
    step = max(1, len(queries) // n) if n else 1
    sample = queries[::step][:n] if n else queries
    got = Retriever(k=3)(sample)
    diff = [q for q, ps in zip(sample, got) if [passage_key(p["title"], p["text"]) for p in ps] != rec[q]]
    return {"queries": len(sample), "of": len(queries), "same_passages": len(sample) - len(diff),
            "agreement": round(1 - len(diff) / len(sample), 4), "examples": diff[:5]}


# 3. checkpoints
def find_model_dir(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith("__MACOSX")]  # Finder's zip adds resource fork copies
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            return d
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, attempts=5):
    """Downloads url to path; a dropped connection resumes where it stopped (HTTP range), up to `attempts` times."""
    for i in range(attempts):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resume = bool(have) and getattr(r, "status", None) == 206  # else the whole file comes again
                length = r.headers.get("Content-Length")
                total = (have if resume else 0) + int(length) if length else None
                with open(path, "ab" if resume else "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if total is None or os.path.getsize(path) >= total:
                return
            log(f"download stopped at {os.path.getsize(path)} of {total} bytes; resuming ({i + 2} of {attempts})")
            continue
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # nothing left to send: the file is complete (the checksum decides)
                return
            raise
        except (OSError, http.client.HTTPException) as e:
            log(f"download interrupted ({type(e).__name__}); resuming, attempt {i + 2} of {attempts}")
            time.sleep(3)
    raise RuntimeError(f"could not download {url} in {attempts} attempts")


def fetch_checkpoint(tag, work, release):
    dest = f"{work}/checkpoints/{tag}"
    found = find_model_dir(dest) if os.path.isdir(dest) else None
    if found:
        return found
    os.makedirs(dest, exist_ok=True)
    zpath = f"{dest}.zip"
    log(f"downloading {release}/{tag}.zip")
    download(f"{release}/{tag}.zip", zpath)
    digest = sha256(zpath)
    if tag in CHECKSUMS and digest != CHECKSUMS[tag]:
        os.remove(zpath)
        raise RuntimeError(f"{tag}.zip has SHA-256 {digest}, expected {CHECKSUMS[tag]}: a different or broken file")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    os.remove(zpath)
    found = find_model_dir(dest)
    if not found:
        raise RuntimeError(f"{tag}.zip holds no config.json next to a .safetensors file")
    return found


# 4. vLLM check
ENGINE_CHECK = """
import sys
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], dtype=sys.argv[2], max_model_len=2048, gpu_memory_utilization=0.5, seed=0,
          enforce_eager=True)
out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=8))
print("ENGINE_OK", repr(out[0].outputs[0].text))
"""


def root_cause(lines):
    """The first error message in a log (vLLM prints the real cause well above its final traceback)."""
    errors = [x.strip() for x in lines if re.search(r"\b[A-Za-z]*(Error|Exception): ", x)]
    return errors[0][:300] if errors else (lines[-1].strip()[:300] if lines else "no output")


def engine_check(work, model_path, dtype):
    """Loads the model in vLLM with this precision and answers one prompt. Returns (ok, message)."""
    os.makedirs(f"{work}/logs", exist_ok=True)
    path = f"{work}/logs/engine_check_{dtype}.log"
    with open(path, "w") as f:
        r = subprocess.run([sys.executable, "-c", ENGINE_CHECK, model_path, dtype], cwd=REPO, stdout=f,
                           stderr=subprocess.STDOUT)
    with open(path) as f:
        lines = f.read().splitlines()
    ok = [x for x in lines if x.startswith("ENGINE_OK")]
    return (True, ok[0]) if r.returncode == 0 and ok else (False, f"{root_cause(lines)} (full log: {path})")


def pick_dtype(work, model_path, env, forced=None):
    """bfloat16 where vLLM supports it (compute capability 8.0 and up), else float16, else float32: the first that
    loads and answers. Returns (dtype, notes)."""
    candidates = [forced] if forced else (["bfloat16"] if env["bf16"] else ["float16", "float32"])
    notes = []
    for dtype in candidates:
        ok, msg = engine_check(work, model_path, dtype)
        notes.append(f"{dtype}: {msg}")
        log(f"vLLM check in {dtype}: {'ok, ' if ok else 'FAILED, '}{msg}")
        if ok:
            return dtype, notes
    return None, notes


# 6. inference
def plan(models):
    """(output name, task list, out of domain?) per run_inference.py process, in the order they run."""
    jobs = []
    for name in models:
        _, ind, ood = MODELS[name]
        jobs += [(name, tasks, False) for tasks in ind] + [(name + "_ood", tasks, True) for tasks in ood]
    return jobs


def command(work, model_path, name, tasks, ood, dtype="bfloat16"):
    cmd = [sys.executable, "run_inference.py", "--project", work, *INF, "--model", model_path, "--name", name,
           "--tasks", ",".join(tasks), "--dtype", dtype]
    return cmd + (["--datasets", OOD] if ood else [])


def format_share(work, name):
    """Share of a policy's agent episodes that end in a well formed answer (the trained policies: 99.6% or more)."""
    path = f"{work}/outputs/{name}/agent.jsonl"
    rows = load_jsonl(path) if os.path.exists(path) else []
    return mean(r["format_ok"] for r in rows) if rows else None


def run_inference(work, name, model_path, tasks, ood, fresh, dtype="bfloat16"):
    folder = f"{work}/outputs/{name}"
    if not fresh and all(os.path.exists(f"{folder}/{t}.jsonl") for t in tasks):
        log(f"{name}: {','.join(tasks)} already done, skipped (use --fresh to redo)")
        return 0.0
    t0 = time.time()
    os.makedirs(f"{work}/logs", exist_ok=True)
    path = f"{work}/logs/{name}_{'_'.join(tasks)}.log"
    log(f"{name}: running {','.join(tasks)} (full log: {path})")
    with open(path, "w") as f:  # vLLM's progress bars go to the log; its summary lines are printed below
        r = subprocess.run(command(work, model_path, name, tasks, ood, dtype), cwd=REPO, stdout=f,
                           stderr=subprocess.STDOUT)
    with open(path) as f:
        lines = f.read().splitlines()
    if r.returncode != 0:
        print("\n".join(lines[-15:]))
        raise RuntimeError(f"run_inference.py failed for {name} ({','.join(tasks)}): {root_cause(lines)} (full log: "
                           f"{path})")
    for line in lines:
        if line.startswith(f"[{name}] "):
            log(line)
    return (time.time() - t0) / 60


# 7. noise
def noise_check(work, model_path, fresh, dtype="bfloat16", n_boot=2000):
    """Evaluates the start policy in domain a second time and compares the two runs: the noise between two runs of
    the same command, measured rather than assumed."""
    run_inference(work, "stageA2_rerun", model_path, ["agent"], False, fresh, dtype)
    a = {r["id"]: r for r in load_jsonl(f"{work}/outputs/stageA2/agent.jsonl")}
    b = {r["id"]: r for r in load_jsonl(f"{work}/outputs/stageA2_rerun/agent.jsonl")}
    out = {}
    for ds in ("hotpotqa", "nq"):
        ids = [i for i in a if a[i]["dataset"] == ds and i in b]
        same = sum((a[i]["answer"], a[i]["n_search"]) == (b[i]["answer"], b[i]["n_search"]) for i in ids)
        bs = paired_bootstrap([a[i]["f1"] for i in ids], [b[i]["f1"] for i in ids], n_boot=n_boot)
        out[ds] = {"same_answers": same, "n": len(ids),
                   "f1_diff": [round(100 * bs[k], 2) for k in ("diff", "lo", "hi")]}
    return out


# 8. comparison
def recorded_rows(rec_key):
    """id -> (answer, searches) for one recorded policy, from the demo data."""
    out = {}
    for ds in SETS:
        q = demo_data(ds)
        if rec_key == "closed":
            out.update({x["id"]: (x["cb"], 0) for x in q["questions"] if x["cb"] is not None})
        else:
            out.update({x["id"]: (r["a"], r["n"]) for x, r in zip(q["questions"], q["runs"][rec_key]) if r})
    return out


def golds():
    return {q["id"]: q["golds"] for ds in SETS for q in demo_data(ds)["questions"]}


def judge(res):
    got, sc_ref = res["got"], res["recorded"][2] if "recorded" in res else res["report"][2]
    if abs(got[2] - sc_ref) > MARGIN_SC:
        return "differs"
    if "f1_diff" in res:
        return "same" if max(abs(res["f1_diff"][1]), abs(res["f1_diff"][2])) <= MARGIN_F1 else "differs"
    return "same" if all(abs(g - e) <= MARGIN_TABLE for g, e in zip(got[:2], res["report"][:2])) else "differs"


def compare(work, models=tuple(MODELS), n_boot=2000):
    with open(f"{REPO}/docs/data/meta.json") as f:
        metrics = json.load(f)["metrics"]
    gold = golds()
    results = []
    for key, label, folder, task, rec_key in ROWS:
        if folder not in models:
            results += [{"row": key, "label": label, "set": ds, "status": "not run"} for ds in SETS]
            continue
        rows = ex.rows_of(f"{work}/outputs", folder, task)
        groups = by_dataset(rows)
        rec = recorded_rows(rec_key) if rec_key and rows else {}
        for i, ds in enumerate(SETS):
            res = {"row": key, "label": label, "set": ds, "report": list(REPORT[key][i])}
            v = groups.get(ds, [])
            if len(v) != EXPECTED_N[ds] or (rec_key and any(r["id"] not in rec for r in v)):
                res.update(status="missing" if not v else "differs", n=len(v),
                           note=f"expected the {EXPECTED_N[ds]} recorded questions")
                results.append(res)
                continue
            got = (100 * mean(r["em"] for r in v), 100 * mean(r["f1"] for r in v), mean(r["n_search"] for r in v))
            res.update(n=len(v), got=[round(got[0], 2), round(got[1], 2), round(got[2], 3)])
            if rec_key:
                m = metrics[rec_key][ds]
                changed = [r["id"] for r in v if rec[r["id"]] != (r["answer"], r["n_search"])]
                bs = paired_bootstrap([f1(rec[r["id"]][0], gold[r["id"]]) for r in v], [r["f1"] for r in v],
                                      n_boot=n_boot)
                res.update(recorded=[round(100 * m["em"], 2), round(100 * m["f1"], 2), round(m["sc"], 3)],
                           f1_diff=[round(100 * bs[k], 2) for k in ("diff", "lo", "hi")],
                           same_answers=len(v) - len(changed), changed=changed[:5])
            res["status"] = judge(res)
            results.append(res)
    return results


def verdict(results, retrieval=None):
    run = [r for r in results if r["status"] != "not run"]
    if not run or any(r["status"] != "same" for r in run):
        return "FAIL"
    if retrieval is not None and retrieval["agreement"] < MIN_RETRIEVAL:
        return "FAIL"
    return "PASS" + (f" ({len(results) - len(run)} of {len(results)} results not run)" if len(run) < len(results)
                     else "")


def print_table(results):
    log("policy            set        EM     F1     SC | report EM   F1    SC | F1 vs recorded [95% CI] | same answers"
        " | status")
    for r in results:
        if "got" not in r:
            log(f"{r['label']:<17} {r['set']:<9} {r['status']}")
            continue
        g, e = r["got"], r["report"]
        d = "{:+.2f} [{:+.2f}, {:+.2f}]".format(*r["f1_diff"]) if "f1_diff" in r else "no recorded answers"
        same = f"{r['same_answers']}/{r['n']}" if "same_answers" in r else ""
        log(f"{r['label']:<17} {r['set']:<9} {g[0]:6.2f} {g[1]:6.2f} {g[2]:6.2f} | {e[0]:9.1f} {e[1]:5.1f} "
            f"{e[2]:5.2f} | {d:>23} | {same:>12} | {r['status']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/content/repro", help="folder for data, checkpoints and outputs")
    ap.add_argument("--policies", default=",".join(MODELS), help="output folders to run, comma separated")
    ap.add_argument("--release", default=RELEASE, help="where the trained checkpoints are downloaded from")
    ap.add_argument("--retrieval_check", type=int, default=2000, help="recorded queries to search again (0: skip)")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"],
                    help="model precision (default: bfloat16 if the GPU has it, else float16, else float32)")
    ap.add_argument("--skip_tests", action="store_true")
    ap.add_argument("--skip_noise", action="store_true", help="skip the second run of the start policy")
    ap.add_argument("--fresh", action="store_true", help="rerun inference even when outputs exist")
    ap.add_argument("--compare_only", action="store_true", help="only compare outputs already in --work")
    a = ap.parse_args()
    models = [m for m in a.policies.split(",") if m]
    unknown = set(models) - set(MODELS)
    if unknown:
        sys.exit(f"unknown policies {sorted(unknown)}; choose from {list(MODELS)}")
    os.makedirs(a.work, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": {}}
    t_all = time.time()
    try:
        if not a.compare_only:
            run_pipeline(a, models, report)
        results = report["rows"] = compare(a.work, models)
        report["verdict"] = verdict(results, report.get("retrieval"))
    except (Exception, SystemExit) as e:  # an early stop still leaves a report that says why
        msg = str(e.code if isinstance(e, SystemExit) else e)
        report["verdict"] = f"FAIL: stopped early. {msg}"
        report["minutes"]["total"] = round((time.time() - t_all) / 60, 1)
        with open(f"{a.work}/reproduce_report.json", "w") as f:
            json.dump(report, f, indent=1)
        log(f"VERDICT: {report['verdict']}")
        sys.exit(1)
    report["minutes"]["total"] = round((time.time() - t_all) / 60, 1)
    with open(f"{a.work}/reproduce_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print_table(results)
    if "retrieval" in report:
        r = report["retrieval"]
        log(f"retrieval: {r['same_passages']} of {r['queries']} recorded queries returned the recorded passages")
    for ds, n in report.get("noise", {}).items():
        log(f"noise: two runs of the same command here, start policy on {ds}: {n['same_answers']}/{n['n']} answers "
            "identical, F1 difference {:+.2f} [{:+.2f}, {:+.2f}]".format(*n["f1_diff"]))
    log(f"rule: same = the 95% interval of the F1 difference lies within +-{MARGIN_F1} points and SC within "
        f"{MARGIN_SC} (baselines without recorded answers: within {MARGIN_TABLE} point of the report)")
    log(f"VERDICT: {report['verdict']}   (details in {a.work}/reproduce_report.json)")
    sys.exit(0 if report["verdict"].startswith("PASS") else 1)


def run_pipeline(a, models, report):
    env = report["environment"] = environment()
    log(f"environment: {json.dumps({k: v for k, v in env.items() if k != 'warnings'})}")
    for w in env["warnings"]:
        log(f"WARNING: {w}.")
    if not env["gpu"]:
        raise SystemExit("No GPU found. In Colab: Runtime, Change runtime type, a GPU (T4 is enough, L4 is closest).")

    if not a.skip_tests:
        report["unit_tests"] = unit_tests()
        log(f"unit tests: {report['unit_tests']['summary']}")
        if not report["unit_tests"]["passed"]:
            raise SystemExit(f"unit tests failed; run: python -m pytest -q {' '.join(TESTS)}")

    t0 = time.time()
    paths = {m: BASE_MODEL if MODELS[m][0] == "hub" else fetch_checkpoint(m, a.work, a.release) for m in models}
    report["minutes"]["checkpoints"] = round((time.time() - t0) / 60, 1)

    t0 = time.time()
    dtype, report["engine_check"] = pick_dtype(a.work, paths[models[0]], env, a.dtype)
    report["dtype"] = dtype
    report["minutes"]["engine_check"] = round((time.time() - t0) / 60, 1)
    if dtype is None:
        raise SystemExit("vLLM could not run the model on this GPU: " + " | ".join(report["engine_check"]))
    if dtype != "bfloat16":
        log(f"WARNING: running in {dtype}; every reported run used bfloat16 (an L4 has it, a T4 does not). More "
            "answers will differ from the recorded runs than between two bfloat16 runs; the same rule judges them.")

    os.makedirs(f"{a.work}/data", exist_ok=True)
    for ds in SETS:
        shutil.copy(f"{REPO}/data/{ds}_test.jsonl", f"{a.work}/data/")
    t0 = time.time()
    start_retriever(a.work)
    if a.retrieval_check:
        report["retrieval"] = retrieval_check(a.retrieval_check)
        log(f"retrieval check: {json.dumps(report['retrieval'])}")
    report["minutes"]["retriever"] = round((time.time() - t0) / 60, 1)

    for name, tasks, ood in plan(models):
        base = name[:-4] if ood else name
        mins = run_inference(a.work, name, paths[base], tasks, ood, a.fresh, dtype)
        report["minutes"][f"{name}:{','.join(tasks)}"] = round(mins, 1)
        share = format_share(a.work, name) if MODELS[base][0] == "release" and "agent" in tasks else None
        if share is not None and share < 0.9:  # the trained policies answer well formed 99.6% of the time
            raise SystemExit(f"{name}: only {share:.0%} of episodes end in a well formed answer, so the {dtype} "
                             "outputs look broken. Rerun with --dtype float32 (slower) or on an L4.")

    if "stageA2" in models and not a.skip_noise:
        t0 = time.time()
        report["noise"] = noise_check(a.work, paths["stageA2"], a.fresh, dtype)
        report["minutes"]["noise"] = round((time.time() - t0) / 60, 1)


if __name__ == "__main__":
    main()
