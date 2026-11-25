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
| `sot/`, `train_grpo.py`, `run_inference.py`, `retriever_server.py`, `setup_retriever.sh`, `test_sot.py` | The project code, exactly as notebook 08 writes it. |
| `export_site.py`, `test_export_site.py` | The export behind notebook 09, and its tests. |
| `docs/` | The demo page (GitHub Pages), a single static HTML file plus its data. |

## Running the tests

```bash
pip install pytest requests
python -m pytest -q test_sot.py test_export_site.py
```

## Rebuilding the demo data

Open `notebooks/09_search_or_think_demo_export.ipynb` in Colab on a CPU runtime, with the project's Drive folder (`search_or_think`) mounted, and run all cells. It checks every exported number against the report's tables before writing `site_data.zip`; unzip it into `docs/data/`.
