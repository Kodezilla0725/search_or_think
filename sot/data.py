"""HotpotQA (distractor) and NQ-open loaders from the Hugging Face Hub, as flat question records, and the out of
domain test sets (2WikiMultihopQA, MuSiQue, PopQA) from the FlashRAG copies."""
import json
import random

FLASHRAG = "RUC-NLPIR/FlashRAG_datasets"
# short name -> (file in the FlashRAG repo, questions kept). Dev sets as test, as for HotpotQA; PopQA has only test.
OOD = {"2wiki": ("2wikimultihopqa/dev.jsonl", 1000), "musique": ("musique/dev.jsonl", 500),
       "popqa": ("popqa/test.jsonl", 500)}


def _first_ok(*loaders):
    err = None
    for load in loaders:
        try:
            return load()
        except Exception as e:  # dataset ids move around on the Hub; try the fallback id
            err = e
            print("falling back after:", repr(e)[:150])
    raise err


def load_split(name, split):
    from datasets import load_dataset
    if name == "hotpotqa":
        return _first_ok(lambda: load_dataset("hotpotqa/hotpot_qa", "distractor", split=split),
                         lambda: load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True))
    return _first_ok(lambda: load_dataset("google-research-datasets/nq_open", split=split),
                     lambda: load_dataset("nq_open", split=split))


def sup_titles(meta):
    """Supporting page titles from a FlashRAG metadata dict, whichever layout it uses: supporting_facts as
    {"title": [...]} or as [[title, sent], ...] (2Wiki), or paragraphs with is_supporting (MuSiQue). [] if none."""
    if not isinstance(meta, dict):
        return []
    sf = meta.get("supporting_facts")
    if isinstance(sf, dict) and isinstance(sf.get("title"), list):
        return sorted({str(t) for t in sf["title"]})
    if isinstance(sf, list):
        titles = {f[0] if isinstance(f, (list, tuple)) and f else f.get("title") if isinstance(f, dict) else None
                  for f in sf}
        return sorted(str(t) for t in titles if t)
    paras = meta.get("paragraphs")
    if isinstance(paras, list):
        return sorted({str(p["title"]) for p in paras if isinstance(p, dict) and p.get("is_supporting") and p.get("title")})
    return []


def flashrag_record(name, i, x):
    golds = x.get("golden_answers") or x.get("answers") or ([x["answer"]] if x.get("answer") else [])
    golds = [golds] if isinstance(golds, str) else golds
    return {"id": f"{name}_{x.get('id', i)}", "dataset": name, "split": "test", "question": x["question"],
            "golds": [str(g) for g in golds if str(g).strip()], "sup_titles": sup_titles(x.get("metadata"))}


def prepare_ood(data_dir, seed=42):
    """Writes data/<name>_test.jsonl for each out of domain set: a fixed random sample of the FlashRAG file, read
    line by line (its metadata varies between rows, which Arrow schema inference can choke on). Returns a report:
    pool size, questions kept, share with supporting titles, and the metadata keys seen, to check the parsing."""
    from huggingface_hub import hf_hub_download
    report = {}
    for name, (path, n) in OOD.items():
        with open(hf_hub_download(FLASHRAG, path, repo_type="dataset")) as f:
            raw = [json.loads(line) for line in f if line.strip()]
        recs = [flashrag_record(name, i, x) for i, x in enumerate(raw) if x.get("question")]
        recs = [r for r in recs if r["golds"]]
        pick = sorted(random.Random(seed).sample(range(len(recs)), n))
        keep = [recs[i] for i in pick]
        assert len({r["id"] for r in keep}) == n, f"{name}: ids are not unique"
        with open(f"{data_dir}/{name}_test.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in keep)
        meta = next((x.get("metadata") for x in raw if isinstance(x.get("metadata"), dict)), {}) or {}
        report[name] = {"pool": len(recs), "n": n, "with_sup_titles": sum(bool(r["sup_titles"]) for r in keep) / n,
                        "metadata_keys": sorted(meta)[:20], "example": {k: keep[0][k] for k in ("id", "question", "golds",
                                                                                                "sup_titles")}}
    return report


def to_record(name, hf_split, out_split, i, x):
    """Same ids and fields as the original data prep cell, so records from different runs match."""
    if name == "hotpotqa":
        return {"id": x["id"], "dataset": name, "split": out_split, "question": x["question"],
                "golds": [x["answer"]], "sup_titles": sorted(set(x["supporting_facts"]["title"])),
                "type": x["type"], "level": x["level"]}
    return {"id": f"nq_{hf_split}_{i}", "dataset": name, "split": out_split, "question": x["question"],
            "golds": list(x["answer"]), "sup_titles": []}