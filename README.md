# Search or Think? Soft Knowledge Boundaries for Cost-Aware Retrieval in Small Language Models

MSAI 635 Reinforcement Learning, final project.

**Team:** Dhairya Parikh ([Dhairya2412](https://github.com/Dhairya2412)) and Kshitij Sahu ([Kodezilla0725](https://github.com/Kodezilla0725)).

**Demo page:** https://kodezilla0725.github.io/search_or_think/

We trained Qwen2.5-0.5B-Instruct with multi-turn GRPO to answer questions with a BM25 search over Wikipedia, and compared four rewards from one shared starting policy: answer F1 (plain GRPO), a fixed per-search penalty, SAAS, and our Soft-SAAS, which weights SAAS's knowledge-boundary penalty by the model's closed-book accuracy p(q). The demo page shows the recorded test trajectories of every policy side by side, on all five test sets.

## What is in this repository

| Path | Contents |
|---|---|
| `notebooks/01` to `08` | The Colab notebooks that ran the project, in order, with their outputs: interim baselines, the p(q) fix, the GRPO smoke test, stage 0 fixes, stage A, stage A2 (forced exploration), stage B (the four rewards), and the evaluation on five test sets. |
| `notebooks/09_search_or_think_demo_export.ipynb` | Reads the saved evaluation outputs and writes the demo page's data (`docs/data/`). CPU only; nothing is rerun. |
| `notebooks/10_reproduce_results.ipynb` | Reproduces the evaluation results from the trained checkpoints (see below). |
| `sot/`, `train_grpo.py`, `run_inference.py`, `retriever_server.py`, `setup_retriever.sh`, `test_sot.py` | The project code, exactly as notebook 08 writes it, except a `--dtype` option in `run_inference.py` (default bfloat16, as evaluated) for GPUs without bfloat16. |
| `reproduce.py`, `test_reproduce.py` | The reproduction pipeline behind notebook 10, and its tests. |
| `export_site.py`, `test_export_site.py` | The export behind notebook 09, and its tests. |
| `data/` | The five test sets, exactly as evaluated: HotpotQA (1,000 dev questions), NQ-open (the first 500 are used), and the FlashRAG copies of 2WikiMultihopQA (1,000), MuSiQue (500) and PopQA (500). |
| `docs/` | The demo page (GitHub Pages), a single static HTML file plus its data. The data also holds every recorded answer, which `reproduce.py` checks against. |

## Reproducing the results

Open `notebooks/10_reproduce_results.ipynb` in Colab on a GPU and run all cells. The free T4 works; an **NVIDIA L4** (the GPU of the original evaluation) gives the closest match. No Google Drive and no training are needed. It runs `reproduce.py`, which:

1. checks the GPU and library versions (the evaluation used an L4, vLLM 0.31.0 and TRL 1.16.0, bfloat16);
2. runs the unit tests;
3. downloads the untrained Qwen2.5-0.5B-Instruct from the Hugging Face Hub and the five trained policies from this repository's [release](https://github.com/Kodezilla0725/search_or_think/releases) (each checked against its SHA-256): the start policy (stage A2), plain GRPO, fixed penalty (lambda 0.05), SAAS and Soft-SAAS, seed 0 each;
4. checks that vLLM can run the model, in bfloat16 where the GPU supports it (compute capability 8.0 and up, such as the L4 or A100), else in float16 (the T4), else float32;
5. starts the BM25 search server over Pyserini's prebuilt `wikipedia-dpr-100w` index, and checks that 2,000 queries from the recorded runs retrieve the same top 3 passages as at evaluation time;
6. evaluates the policies with greedy decoding on all five test sets, with the same commands as the original evaluation;
7. evaluates the start policy a second time, to measure how much two runs of the same command differ;
8. compares every policy on every test set with the recorded run, and prints PASS or FAIL.

vLLM does not make greedy decoding bit for bit repeatable by default ([vLLM docs](https://docs.vllm.ai/en/v0.31.0/usage/reproducibility/)): requests are batched differently from run to run, so a few uncertain answers change. In our own run on an L4, two runs of the same command agreed on 79% of HotpotQA answers while their F1 differed by 0.4 points. So the test is not identical answers but the same numbers: for each policy and test set, the paired bootstrap 95% interval of the F1 difference between the reproduced and the recorded run (the report's own test) must lie within ±2.5 points, and the search count within 0.1. On a T4, float16 arithmetic changes more answers than that; the same rule judges the numbers. The full run takes about an hour on an L4, longer on a T4.

Training is not part of the reproduction: the 22 GRPO runs took many hours of GPU time. Notebooks 01 to 08 contain the exact training commands and their logs.

## Running the tests

```bash
pip install pytest requests
python -m pytest -q test_sot.py test_export_site.py test_reproduce.py
```

## Rebuilding the demo data

Open `notebooks/09_search_or_think_demo_export.ipynb` in Colab on a CPU runtime, with the project's Drive folder (`search_or_think`) mounted, and run all cells. It checks every exported number against the report's tables before writing `site_data.zip`; unzip it into `docs/data/`.
