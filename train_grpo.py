"""GRPO through the multi-turn search loop for all four rewards (TRL 1.16, colocated vLLM, one GPU).

  python train_grpo.py --project P --reward {plain,fixed,saas,soft} [--lam L] [--alpha A]
      [--warmup {off,manual_step:N,self}] [--init_checkpoint DIR] [--boundary FILE] [--seed S] [--lora] ...

Each run lives in <project>/checkpoints/grpo_<model>_<tag>/:
  log.jsonl      one line per optimizer step: reward, f1, searches_mean, n0_share / n1_share / n2plus_share
                 (rollouts with 0, 1, 2+ searches), repeat_share (searches that repeat an earlier query of their
                 episode), w, trigger_step, penalised_share, zero_var_share,
                 fake_info_rate, format_ok, entropy, completions/mean_length, wall_s, ...
  samples.jsonl  two rollouts per step, for reading
  warmup.json    reward history behind w(t), so a resumed run keeps its warm-up state
  checkpoint-N/  resume state, written every --save_steps steps; rerunning the same command resumes from the last one
  final/         bf16 weights for evaluation and for --init_checkpoint of a later run
  done.json      written last; rerunning a finished command does nothing
"""
import argparse
import itertools
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sot.agent import STOP, agent_instruction, extract_answer
from sot.rewards import ALPHA, Boundary, Warmup, compute_rewards, group_stats, saas_label, tighten_n_min
from sot.rollout import multi_turn_rollout, resample_flat_groups
from sot.scoring import f1

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MICRO_BATCH = 4  # sequences per forward pass, fixed so that G = 16 needs no more memory than G = 8
BOUNDARY_COLS = ("id", "dataset", "golds", "p", "n_min", "label")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="base model (names the run)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--reward", choices=["plain", "fixed", "saas", "soft"], default="plain")
    ap.add_argument("--lam", type=float, default=0.1, help="fixed penalty per search")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="SAAS / Soft-SAAS penalty per unneeded search")
    ap.add_argument("--warmup", default="off", help="off (w = 1) | manual_step:N | self")
    ap.add_argument("--ramp", type=int, default=None, help="warm-up ramp in steps (default: 50 for self, 0 for manual)")
    ap.add_argument("--init_checkpoint", default=None, help="start from these weights instead of the base model")
    ap.add_argument("--boundary", default=None, help="jsonl with id, p, n_min per training question (from relabelling)")
    ap.add_argument("--saas_rule", choices=["code", "paper"], default="code")
    ap.add_argument("--online_nmin", action="store_true", help="also lower N_min to the group's own best, as SAAS does")
    ap.add_argument("--format_bonus", type=float, default=0.0, help="reward for a well formed but wrong answer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters (for 1.5B)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--questions_per_step", type=int, default=8)
    ap.add_argument("--G", type=int, default=8, help="rollouts per question")
    ap.add_argument("--resample", type=int, default=0, help="rounds of swapping zero-variance groups for new questions")
    ap.add_argument("--force_search", type=int, default=0, help="rollouts per question that must search first")
    ap.add_argument("--force_answer", type=int, default=0, help="rollouts per question that must answer without search")
    ap.add_argument("--force_strict", action="store_true",
                    help="forcing cannot be dodged: a forced rollout that opens <search> or <answer> without closing "
                         "<think> is cut back and given the forced tag (default: such a rollout stays free)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--train_file", default=None, help="default: <project>/data/train_enriched.jsonl")
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--keep_checkpoints", action="store_true", help="keep every checkpoint, not only the last")
    ap.add_argument("--save_optimizer", action="store_true", help="also checkpoint Adam state (3x the disk)")
    ap.add_argument("--cleanup", action="store_true", help="delete the resume checkpoints once final/ is written")
    ap.add_argument("--no_save", action="store_true", help="write no weights at all (timing runs and sweeps)")
    ap.add_argument("--fresh", action="store_true", help="delete an existing run of the same name instead of resuming")
    ap.add_argument("--tag", default=None, help="default: <reward>_s<seed>")
    args = ap.parse_args(argv)
    if args.G % MICRO_BATCH:
        ap.error(f"--G must be a multiple of {MICRO_BATCH}")
    if args.force_search + args.force_answer >= args.G:
        ap.error("--force_search + --force_answer must leave some free rollouts in each group")
    args.tag = args.tag or f"{args.reward}_s{args.seed}"
    return args


def load_questions(args):
    """Training questions with their knowledge boundary. n_min = -1 means no correct search enabled rollout."""
    path = args.train_file or f"{args.project}/data/train_enriched.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()]
    bmap = {r["id"]: r for r in map(json.loads, open(args.boundary))} if args.boundary else {}
    out = []
    for r in rows:
        b = bmap.get(r["id"], {})
        p, n_min = float(b.get("p", r.get("p_pool", 0.0))), b.get("n_min")
        out.append({"prompt": r["question"], "id": r["id"], "dataset": r["dataset"], "golds": r["golds"], "p": p,
                    "n_min": -1 if n_min is None else int(n_min), "label": saas_label(p, n_min, args.saas_rule)})
    if args.reward in ("saas", "soft") and len(bmap) < len(rows):
        print(f"WARNING: {len(rows) - sum(r['id'] in bmap for r in rows)} of {len(rows)} questions have no relabelled "
              "boundary; they use the base model's p(q) and no N_min. Fine for a dry run, not for a real run.")
    return path, out


def score(args, w, completions, golds, n_search, p, n_min, label, forced=None):
    """Rewards for one batch of rollouts. Strict parsing: an answer outside <answer> tags earns nothing, which
    also teaches the format. A rollout that was forced to search is not charged for that first search: it did not
    choose it. Returns (rewards, penalised, f1s, format_ok)."""
    answers = [extract_answer(c) for c in completions]
    f1s = [f1(a or "", g) for a, g in zip(answers, golds)]
    bs = [Boundary(pi, None if ni < 0 else ni, li) for pi, ni, li in zip(p, n_min, label)]
    if args.online_nmin:
        bs = tighten_n_min(bs, f1s, n_search, args.G, args.saas_rule)
    ok = [a is not None for a in answers]
    chosen = [n - (f == 1) for n, f in zip(n_search, forced or [0] * len(n_search))]  # searches it chose itself
    rewards, penalised = compute_rewards(args.reward, f1s, chosen, bs, lam=args.lam, alpha=args.alpha, w=w,
                                         format_bonus=args.format_bonus, format_ok=ok)
    return rewards, penalised, f1s, ok


def make_reward_fn(args, warm, out):
    """The TRL reward function. Dataset columns and the rollout's extra fields arrive as keyword lists."""
    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    def reward_fn(completions, golds, n_search, fake_info, p, n_min, label, trainer_state, log_metric, **kw):
        step = trainer_state.global_step  # optimizer steps finished so far: constant within one optimizer step
        w = warm.weight(step)
        forced = kw.get("forced") or [0] * len(completions)
        rewards, penalised, f1s, ok = score(args, w, completions, golds, n_search, p, n_min, label, forced)
        free = [i for i, f in enumerate(forced) if f == 0]  # rollouts that made the search decision themselves
        f1_free, n_free = [f1s[i] for i in free], [n_search[i] for i in free]
        n_repeat = kw.get("n_repeat") or [0] * len(completions)
        warm.observe(step, mean(f1_free))  # the trigger watches answer F1, never the penalised reward
        with open(f"{out}/warmup.json", "w") as f:
            json.dump(warm.state(), f)
        stats = group_stats(rewards, args.G)
        metrics = [("f1", mean(f1_free)), ("searches_mean", mean(n_free)), ("w", w),  # f1, searches: free rollouts
                   ("trigger_step", -1 if warm.trigger_step is None else warm.trigger_step + 1),
                   ("penalised_share", mean(penalised)), ("zero_var_share", stats["zero_var_share"]),
                   ("zero_var_all_zero", stats["zero_var_all_zero"]), ("fake_info_rate", mean(fake_info)),
                   ("format_ok", mean(ok)), ("resampled_groups", kw.get("resampled", [0])[0]),
                   ("n0_share", mean([n == 0 for n in n_free])), ("n1_share", mean([n == 1 for n in n_free])),
                   ("n2plus_share", mean([n >= 2 for n in n_free])), ("forced_share", 1 - len(free) / len(forced)),
                   ("repeat_share", sum(n_repeat[i] for i in free) / max(sum(n_free), 1))]
        for code, name in ((1, "f1_forced_search"), (2, "f1_forced_answer")):
            arm = [f1s[i] for i, f in enumerate(forced) if f == code]
            if arm:
                metrics.append((name, mean(arm)))
        for name, value in metrics:
            log_metric(name, float(value))
        with open(f"{out}/samples.jsonl", "a") as f:
            for i in (0, len(completions) // 2):
                f.write(json.dumps({"step": step + 1, "id": kw.get("id", [None] * len(completions))[i],
                                    "n_search": n_search[i], "forced": forced[i], "f1": f1s[i], "reward": rewards[i],
                                    "golds": golds[i], "completion": completions[i][-1500:]}) + "\n")
        return rewards
    return reward_fn


def bf16_dir(path, tokenizer_from):
    """vLLM takes its dtype from config.json and samples in fp16 when that says float32 (as Trainer checkpoints do),
    while the base model samples in bf16. Start every run from bf16 weights so sampling is the same everywhere."""
    cfg = json.load(open(f"{path}/config.json"))
    if (cfg.get("dtype") or cfg.get("torch_dtype")) == "bfloat16":
        return path
    dst = path.rstrip("/") + "_bf16"
    if not os.path.exists(f"{dst}/config.json"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).save_pretrained(dst)
        has_tok = os.path.exists(f"{path}/tokenizer_config.json")
        AutoTokenizer.from_pretrained(path if has_tok else tokenizer_from).save_pretrained(dst)
    return dst


def main(argv=None, llm=None, retrieve=None):
    """llm / retrieve: fakes for the CPU test of this script; a real run leaves them None."""
    args = parse_args(argv)
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    on_gpu = llm is None
    out = f"{args.project}/checkpoints/grpo_{args.model.split('/')[-1]}_{args.tag}"
    final = f"{out}/final"
    if args.fresh and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    done_path = f"{out}/done.json"
    if os.path.exists(done_path) and json.load(open(done_path))["steps"] >= args.steps:
        print(f"{out}: already finished {args.steps} steps, nothing to do (use --fresh to redo)")
        return out

    # data
    path, rows = load_questions(args)
    ds = Dataset.from_list(rows).shuffle(seed=0)
    by_q = {r["prompt"]: r for r in rows}  # shortcut: two records with the same question text share one boundary
    reserve = itertools.cycle(random.Random(args.seed + 1).sample(rows, len(rows)))  # questions for --resample
    labels = {k: sum(r["label"] == k for r in rows) for k in ("NoSearch", "NeedSearch", "Undetermined")}

    # resume
    last = get_last_checkpoint(out)
    start = int(last.rsplit("-", 1)[1]) if last else 0
    warm = Warmup(args.warmup, args.ramp)
    log_path = f"{out}/log.jsonl"
    if last and os.path.exists(f"{out}/warmup.json"):
        warm.restore(json.load(open(f"{out}/warmup.json")), start)
    for name in ("log.jsonl", "samples.jsonl"):  # keep only the steps the checkpoint already covers
        old = [json.loads(line) for line in open(f"{out}/{name}")] if last and os.path.exists(f"{out}/{name}") else []
        with open(f"{out}/{name}", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in old if x["step"] <= start)

    if retrieve is None:
        from sot.retrieval import Retriever
        retrieve = Retriever(k=3)
        print("search server:", retrieve.health())
    print(f"run: {out}\nreward={args.reward} warmup={args.warmup} lr={args.lr} G={args.G} "
          f"(forced search {args.force_search}, forced answer {args.force_answer}, strict {args.force_strict}) "
          f"questions/step={args.questions_per_step} seed={args.seed} | {len(rows)} questions from {path} | "
          f"labels {labels} | " + (f"resuming from step {start}" if last else "starting at step 0"), flush=True)

    stop_force = STOP + ["</think>"] + (["<search>", "<answer>"] if args.force_strict else [])
    if on_gpu:
        from vllm import SamplingParams
        kw = dict(temperature=1.0, max_tokens=512, logprobs=0, include_stop_str_in_output=True)
        sp, sp_force = SamplingParams(stop=STOP, **kw), SamplingParams(stop=stop_force, **kw)
    else:
        sp, sp_force = argparse.Namespace(max_tokens=64, stop=STOP), argparse.Namespace(max_tokens=64, stop=stop_force)
    # Exploration floor on the search decision: in every group the first --force_search rollouts must search and the
    # next --force_answer must answer without search; the rest decide for themselves.
    arms = ["search"] * args.force_search + ["answer"] * args.force_answer + [None] * args.G

    def rollout(prompts, trainer):
        tok = trainer.processing_class
        engine = trainer.vllm_generation.llm if on_gpu else llm

        def roll(recs):
            chat = [tok.apply_chat_template([{"role": "user", "content": agent_instruction(r["prompt"])}],
                                            tokenize=False, add_generation_prompt=True) + "<think>" for r in recs]
            return multi_turn_rollout(engine, tok, retrieve, chat, sp, force=[arms[i % args.G] for i in range(len(recs))],
                                      force_params=sp_force, strict=args.force_strict)

        recs = [by_q[q] for q in prompts]
        res = roll(recs)
        swapped = 0
        if args.resample:
            w = warm.weight(trainer.state.global_step)

            def rewards_of(o, rs):
                texts = tok.batch_decode(o["completion_ids"], skip_special_tokens=True)
                golds, p, n_min, label = ([r[k] for r in rs] for k in ("golds", "p", "n_min", "label"))
                return score(args, w, texts, golds, o["n_search"], p, n_min, label, o["forced"])[0]

            swapped = resample_flat_groups(res, recs, roll, rewards_of, reserve, args.G, args.resample)
            for k in BOUNDARY_COLS:  # the rollout's fields replace the dataset columns, so swapped groups score right
                res[k] = [r[k] for r in recs]
        res["resampled"] = [swapped] * len(prompts)
        return res

    class StepLog(TrainerCallback):
        t0 = time.time()

        def on_log(self, a, state, control, logs=None, **kw):
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, "wall_s": time.time() - self.t0, **(logs or {})}) + "\n")

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                 task_type="CAUSAL_LM")

    # One generation batch of questions x G rollouts per optimizer step:
    # MICRO_BATCH sequences per forward pass x (questions x G / MICRO_BATCH) accumulation steps.
    # Constant learning rate: stage B branches continue at the rate stage A ended on, and a resumed or extended run
    # needs no scheduler state. Checkpoints hold weights only unless --save_optimizer (a resumed run then restarts
    # Adam's moment estimates). shortcut: so a resumed run is not bit-identical to an uninterrupted one.
    cfg = GRPOConfig(output_dir=out, max_steps=args.steps, learning_rate=args.lr, lr_scheduler_type="constant",
                     seed=args.seed, num_generations=args.G, per_device_train_batch_size=MICRO_BATCH,
                     gradient_accumulation_steps=args.questions_per_step * args.G // MICRO_BATCH,
                     beta=0.0, temperature=1.0, bf16=on_gpu, gradient_checkpointing=on_gpu,
                     use_vllm=on_gpu, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
                     vllm_max_model_length=4096, max_completion_length=3072, logging_steps=1,
                     save_strategy="no" if args.no_save else "steps", save_steps=args.save_steps,
                     save_total_limit=None if args.keep_checkpoints else 1,
                     save_only_model=not args.save_optimizer, report_to="none", use_cpu=not on_gpu)
    model = bf16_dir(args.init_checkpoint, args.model) if args.init_checkpoint else args.model
    trainer = GRPOTrainer(model=model, reward_funcs=make_reward_fn(args, warm, out), args=cfg, train_dataset=ds,
                          rollout_func=rollout, callbacks=[StepLog()], peft_config=peft_config)
    trainer.train(resume_from_checkpoint=last)

    # final weights in bf16: what vLLM sampled from during training, and half the disk of float32
    if not args.no_save:
        net = trainer.model.merge_and_unload() if args.lora else trainer.model
        net.to(torch.bfloat16).save_pretrained(final)
        trainer.processing_class.save_pretrained(final)
    with open(done_path, "w") as f:
        json.dump({"steps": args.steps, "resumed_from": start, "args": vars(args),
                   "trigger_step": None if warm.trigger_step is None else warm.trigger_step + 1}, f, indent=1)
    if args.cleanup:
        for d in os.listdir(out):
            if d.startswith("checkpoint-"):
                shutil.rmtree(f"{out}/{d}")
    print(f"finished: {out} | " + (f"resumed from step {start}" if last else "started at step 0"), flush=True)
    return out


if __name__ == "__main__":
    main()