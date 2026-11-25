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
| `sot/`, `train_grpo.py`, `run_inference.py`, `retriever_server.py`, `setup_retriever.sh`, `test_sot.py` | The project code, exactly as notebook 08 writes it. |
| `reproduce.py`, `test_reproduce.py` | The reproduction pipeline behind notebook 10, and its tests. |
| `export_site.py`, `test_export_site.py` | The export behind notebook 09, and its tests. |
| `data/` | The five test sets, exactly as evaluated: HotpotQA (1,000 dev questions), NQ-open (the first 500 are used), and the FlashRAG copies of 2WikiMultihopQA (1,000), MuSiQue (500) and PopQA (500). |
| `docs/` | The demo page (GitHub Pages), a single static HTML file plus its data. The data also holds every recorded answer, which `reproduce.py` checks against. |

## Reproducing the results

Open `notebooks/10_reproduce_results.ipynb` in Colab on an **NVIDIA L4** GPU and run all cells. No Google Drive and no training are needed. It runs `reproduce.py`, which:

1. checks the GPU and library versions (the evaluation used an L4, vLLM 0.31.0 and TRL 1.16.0; the policies need bfloat16, so the free T4 GPU cannot run them);
2. runs the unit tests;
3. starts the BM25 search server over Pyserini's prebuilt `wikipedia-dpr-100w` index, and checks that 2,000 queries from the recorded runs retrieve the same top 3 passages as at evaluation time;
4. downloads the untrained Qwen2.5-0.5B-Instruct from the Hugging Face Hub and the five trained policies from this repository's [release](https://github.com/Kodezilla0725/search_or_think/releases): the start policy (stage A2), plain GRPO, fixed penalty (lambda 0.05), SAAS and Soft-SAAS, seed 0 each;
5. evaluates them with greedy decoding on all five test sets, with the same commands as the original evaluation;
6. compares every answer with the recorded run and every EM, F1 and search count with the report's appendix table, and prints PASS or FAIL. A result is *exact* when every answer matches, and *close* when at least 97% of answers match and EM and F1 are within 1 point. Anything else fails.

Training is not part of the reproduction: the 22 GRPO runs took many hours of GPU time. Notebooks 01 to 08 contain the exact training commands and their logs.

## Running the tests

```bash
pip install pytest requests
python -m pytest -q test_sot.py test_export_site.py test_reproduce.py
```

## Rebuilding the demo data

Open `notebooks/09_search_or_think_demo_export.ipynb` in Colab on a CPU runtime, with the project's Drive folder (`search_or_think`) mounted, and run all cells. It checks every exported number against the report's tables before writing `site_data.zip`; unzip it into `docs/data/`.
