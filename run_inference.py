"""Inference only experiments for one model (no RL). Writes one jsonl per task to outputs/<model>/.

Tasks
  closed_sampled  K search disabled samples (T=1.0) on probe + test  -> p(q)
  no_search       greedy, no retrieval, test                         -> baseline
  rag             one forced search with the question, greedy, test  -> baseline
  agent           prompted search agent, greedy, test                -> baseline + QOR / SOR
  agent_sampled   K prompted agent samples (T=1.0) on probe          -> N_min, SAAS labels, reward signal check
  closed_pool     K search disabled samples (T=1.0) on the training pool -> p(q) for the enriched training set
  closed_test     K search disabled samples (T=1.0) on test only     -> p(q) on test sets (QOR, adaptivity)
  agent_val       prompted search agent, greedy, probe split         -> validation (checkpoint choice)
  agent_train     K agent samples (T=1.0) on the training set        -> N_min at a checkpoint (relabelling)
                  with --max_searches 1 --prompt_searches 4: the one-search budget probe (same prompt, and the
                  environment answers a second search with the search limit message)

--model may be a checkpoint folder; give it a --name, which becomes the folder under outputs/.
--datasets picks the test sets (data/<name>_test.jsonl). HotpotQA and NQ are cut to --n and --n_nq; the out of
domain sets (2wiki, musique, popqa, written by data.prepare_ood) are used whole unless --n_ood is given.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sot.agent import agent_instruction, closed_book_instruction, format_info, rag_instruction, run_agent
from sot.retrieval import Retriever
from sot.scoring import em, f1, n_distinct

ALL_TASKS = "closed_sampled,no_search,rag,agent,agent_sampled"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_row(rec, s, sample, slim=False):
    row = {"id": rec["id"], "dataset": rec["dataset"], "split": rec["split"], "sample": sample,
           "answer": s["answer"], "format_ok": s["format_ok"], "em": em(s["answer"], rec["golds"]),
           "f1": f1(s["answer"], rec["golds"]), "n_search": s["n_search"], "stop_reason": s["stop_reason"],
           "n_distinct": n_distinct([x["query"] for x in s["searches"]])}
    if not slim:  # slim rows (relabelling) skip the retrieved passages and the full text: 24,000 of them per run
        row["searches"] = [{"query": x["query"], "titles": [p["title"] for p in x["passages"]],
                            "texts": [p["text"] for p in x["passages"]]} for x in s["searches"]]
        row["text"] = s["text"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--n", type=int, default=500, help="questions per dataset per split")
    ap.add_argument("--n_nq", type=int, default=None, help="NQ questions per split (default: --n)")
    ap.add_argument("--name", default=None, help="output folder name (default: model name plus --tag)")
    ap.add_argument("--train_file", default=None, help="for agent_train (default: data/train_enriched.jsonl)")
    ap.add_argument("--train_n", type=int, default=None, help="first N training questions only (dry runs)")
    ap.add_argument("--k", type=int, default=4, help="samples per question for p(q) and N_min")
    ap.add_argument("--max_searches", type=int, default=4)
    ap.add_argument("--prompt_searches", type=int, default=None,
                    help="search limit named in the prompt (default: --max_searches); lets a probe cap the searches "
                         "below what the training prompt says")
    ap.add_argument("--tasks", default=ALL_TASKS)
    ap.add_argument("--datasets", default="hotpotqa,nq", help="comma separated test sets")
    ap.add_argument("--n_ood", type=int, default=None, help="cap on the out of domain test sets (dry runs)")
    ap.add_argument("--pool_n", type=int, default=10000, help="pool questions per dataset for closed_pool")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16",
                    help="model precision; every reported run used bfloat16 (float16 / float32 for GPUs without it)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    data = os.path.join(args.project, "data")
    n_of = {"hotpotqa": args.n, "nq": args.n_nq or args.n}
    sets = args.datasets.split(",")
    test = sum((load_jsonl(f"{data}/{d}_test.jsonl")[:n_of.get(d, args.n_ood)] for d in sets), [])
    probe = sum((load_jsonl(f"{data}/{d}_probe.jsonl")[:n_of.get(d, args.n_ood)] for d in sets
                 if os.path.exists(f"{data}/{d}_probe.jsonl")), [])
    tasks = args.tasks.split(",")
    train = load_jsonl(args.train_file or f"{data}/train_enriched.jsonl")[:args.train_n] \
        if "agent_train" in tasks else []
    pool = sum((load_jsonl(f"{data}/{d}_pool.jsonl")[:args.pool_n] for d in ("hotpotqa", "nq")), []) \
        if "closed_pool" in tasks else []
    name = args.name or args.model.split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.project, "outputs", name)
    os.makedirs(out_dir, exist_ok=True)

    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, seed=0, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    retrieve = Retriever(k=3)

    def chat(instr):
        return tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)

    def generate(prompts, stop, max_tokens, temperature):
        lens = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]
        keep = [i for i, n in enumerate(lens) if n + max_tokens <= args.max_model_len]  # too long -> "" -> agent stops it
        outs = [""] * len(prompts)
        if keep:
            sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, stop=stop,
                                include_stop_str_in_output=True)
            for i, r in zip(keep, llm.generate([prompts[i] for i in keep], sp, use_tqdm=True)):
                outs[i] = r.outputs[0].text
        return outs

    def run(task, recs, instr_fn, k=1, temperature=0.0, allow_search=True, max_turn_tokens=512, rag=False,
            prefix="", slim=False):
        """prefix starts the assistant turn ("<answer>" or "<think>") so small models stay in the tag format."""
        t0 = time.time()
        recs_k = [r for r in recs for _ in range(k)]
        prefill = [{"text": prefix, "searches": []} for _ in recs_k]
        if rag:
            passages = retrieve([r["question"] for r in recs_k])
            prefill = [{"text": f"<search> {r['question']} </search>" + format_info(p) + prefix,
                        "searches": [{"query": r["question"], "passages": p}]} for r, p in zip(recs_k, passages)]
        states = run_agent(generate, retrieve, [chat(instr_fn(r["question"])) for r in recs_k],
                           allow_search=allow_search, max_searches=args.max_searches,
                           max_turn_tokens=max_turn_tokens, temperature=temperature, prefill=prefill)
        rows = [to_row(r, s, i % k, slim) for i, (r, s) in enumerate(zip(recs_k, states))]
        with open(f"{out_dir}/{task}.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        mins = (time.time() - t0) / 60
        print(f"[{name}] {task}: {len(rows)} trajectories in {mins:.1f} min, "
              f"EM={sum(x['em'] for x in rows) / len(rows):.3f}, "
              f"searches={sum(x['n_search'] for x in rows) / len(rows):.2f}", flush=True)
        return mins

    agent_instr = lambda q: agent_instruction(q, args.prompt_searches or args.max_searches)
    jobs = {
        "closed_sampled": lambda: run("closed_sampled", probe + test, closed_book_instruction, k=args.k,
                                      temperature=1.0, allow_search=False, max_turn_tokens=32, prefix="<answer>"),
        "closed_test": lambda: run("closed_test", test, closed_book_instruction, k=args.k, temperature=1.0,
                                   allow_search=False, max_turn_tokens=32, prefix="<answer>", slim=True),
        "no_search": lambda: run("no_search", test, closed_book_instruction, allow_search=False, max_turn_tokens=32,
                                 prefix="<answer>"),
        "rag": lambda: run("rag", test, rag_instruction, allow_search=False, max_turn_tokens=32, rag=True,
                           prefix="<answer>"),
        "agent": lambda: run("agent", test, agent_instr, prefix="<think>"),
        "agent_sampled": lambda: run("agent_sampled", probe, agent_instr, k=args.k, temperature=1.0, prefix="<think>"),
        "closed_pool": lambda: run("closed_pool", pool, closed_book_instruction, k=args.k, temperature=1.0,
                                   allow_search=False, max_turn_tokens=32, prefix="<answer>"),
        "agent_val": lambda: run("agent_val", probe, agent_instr, prefix="<think>"),
        "agent_train": lambda: run("agent_train", train, agent_instr, k=args.k, temperature=1.0, prefix="<think>",
                                   slim=True),
    }
    timings = {t: jobs[t]() for t in tasks}
    tpath = f"{out_dir}/timings.json"  # merged, so a later partial run keeps the earlier tasks' entries
    log = json.load(open(tpath)) if os.path.exists(tpath) else {"model": args.model, "n": args.n, "minutes": {}}
    log["minutes"].update(timings)
    log.setdefault("k_by_task", {}).update({t: args.k for t in timings})
    with open(tpath, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()