# Customer Success AI Copilot (DSS)

Containerized multi-agent decision support for proactive B2B account outreach, built for a B2B textile exporter.

> **All customer data in this repository is synthetic.** No real company, customer or employee data is included.
Monitoring → Analyst (CLV + churn + TreeSHAP) → RAG → NBA playbook → Writer (guardrailed) → human review.

## Laptop demo (start here)

There are two ways to run the copilot, both using the same agent code in `src/copilot`:

- **Laptop demo (this section):** one process, no Docker, no server. Needs Python 3.12 on Windows. SQLite and
  in-memory document search stand in for Postgres and ChromaDB. Best for showing the system.
- **Docker stack (pilot and production):** the five containers in `docker-compose.yml` (Postgres, ChromaDB,
  pipeline, API, review screen). See [Services](#services) and [Run it](#run-it-windows--powershell--docker-desktop-with-wsl2) below.

Double-click **`run_demo.bat`** in File Explorer, or from a terminal in the project folder:

```powershell
.\run_demo.bat
```

If Windows blocks a script with "is not digitally signed" (for example `run_demo.ps1` or
`.venv312\Scripts\Activate.ps1`), allow scripts for the current PowerShell window only, then run the command again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This lasts until you close the window and doesn't change any system settings.

(`run_demo.bat` starts `run_demo.ps1` for you, so Windows' "script is not digitally signed" block doesn't apply.)

The first run creates a `.venv-demo` environment and installs packages (a few minutes), then opens
http://localhost:8501. The first page load runs the full pipeline once (~10 seconds); after that it's instant.
Data lives in `demo/_data/`. **Reset demo** in the sidebar clears all decisions and reruns the pipeline.

Set `ANTHROPIC_API_KEY` in `.env` to have Claude write the drafts; without it, the writer uses grounded templates.

### 6-minute demo script

1. **Overview (1 min).** "Every night it scans 2,000 accounts. Tonight 326 changed behaviour and 175 are high risk.
   If we do nothing, we estimate about €4.6M of next year's revenue from them is at risk: each customer's predicted
   value weighted by how likely they are to leave." Point at the table: ordered by revenue at risk, not just churn.
   (Synthetic data: the real figure comes from the pilot.)
2. **Account story → ★ Quality problems on a valuable account (2 min).** Walk 1 → 4: why flagged, what drives
   the risk (red bars push risk up, green pull it down), the playbook action, and a draft that opens on the
   customer's actual problem. Open *Customer history used to write this draft* to show it's grounded.
3. **★ Guardrail: not enough information (30 s).** "When there's no customer history, it refuses to write.
   It says what's missing instead of inventing something."
4. **Review queue (1 min).** Edit a sentence, approve. Reject another: choose a reason (e.g. *Tone*) and explain
   ("too formal for this buyer"). "Nothing is ever sent without a person deciding."
5. **Feedback log (1 min).** Show the rejection reason and the approved example, then *What the writer will see on the
   next cycle*. Click **Run next cycle**: the rejected account gets a new draft written with that feedback, and the
   approved account isn't contacted again. "Every decision teaches the writer, and becomes training data."
6. **Results (30 s).** Approval rate and the audit log.
7. **Trust metrics (1 min).** "This is how the system proves itself over time": approval rate and top rejection
   reasons; flagged vs. actually churned on accounts the model never saw; contacted vs. control group (20% of
   flagged accounts are held back at random, and retention is measured in the pilot); model version and last
   retraining date. Close with the honest line: "Today these are synthetic labels. In the pilot the same page
   fills with real ERP outcomes."

### Local LLM with Ollama (no data leaves the machine)

The writer can use an open model running on your own computer instead of a cloud API.

1. Install Ollama from ollama.com and pull a model (once):
   ```powershell
   ollama pull mistral:7b
   ```
2. In `.env` set:
   ```dotenv
   WRITER_BACKEND=ollama
   OLLAMA_MODEL=mistral:7b
   LLM_MAX_DRAFTS=15
   ```
   Only the top 15 accounts by revenue at risk get an LLM draft; the rest use templates, so a
   run stays fast even on CPU. Delete `demo\_data\` after changing the backend so the demo
   regenerates its drafts.
3. Start the demo (`.\run_demo.ps1`) or the Docker stack as usual. The sidebar shows
   *Writer: local LLM via Ollama*. Containers reach Ollama on the host automatically.

The guardrails are identical: drafts must cite the retrieved documents. If the local model's
draft is ungrounded or malformed, or Ollama isn't running, the writer uses the grounded template
instead, so the demo never breaks. A 7B model needs about 8 GB of free RAM and takes from a few
seconds (with a GPU) to a minute or more (CPU only) per draft.

### Trust metrics: how the retention uplift is calculated

Panel 3 of the *Trust metrics* page compares flagged accounts that received copilot outreach with the
control group (flagged accounts held back at random, `CONTROL_GROUP_PCT`). It is calculated in
`retention_comparison()` in `src/copilot/trust.py`:

```
retention rate = customers who kept buying ÷ customers with a known outcome
uplift         = retention rate (contacted) − retention rate (control)
```

Outcomes come from the ERP after the fact (`python -m copilot.trust import-outcomes data/outcomes.csv`).
Until both groups have outcomes, the panel shows "—".

**Worked example (pilot, 6 months later)**

| Group | With outcome | Retained | Retention rate |
|---|---|---|---|
| Contacted | 150 | 123 | 123 ÷ 150 = 82% |
| Control | 60 | 41 | 41 ÷ 60 = 68% |
| **Uplift** | | | 82% − 68% = **+14 points** |

Meaning: among similar at-risk customers, the copilot group kept 14 more out of every 100.

The figures above are illustrative, not results. Two points to keep in mind when reading the real numbers:

- **Contacted** means outreach a reviewer approved, which is not a random group. The fairest comparison is
  all non-control flagged accounts vs. the control group (intention-to-treat).
- With small groups, part of the difference can be chance; report a margin of error alongside the uplift.

### Using your own data

The demo reads data through `load_gold()` in `src/copilot/data.py` and context documents through
`synthetic_context()` in `src/copilot/rag.py`. Replace those two functions to read from `dss.db`
(same column names), delete `demo/_data/`, and restart. Nothing else changes.

## Services

| Service | Image | Role | Port |
|---|---|---|---|
| `db` | postgres:16 | Hand-off store between agents, audit log | — |
| `vectorstore` | chromadb 0.5 | Per-account context for RAG | — |
| `pipeline` | `docker/Dockerfile` → `pipeline` | Runs all agents on a schedule, trains models | — |
| `api` | `docker/Dockerfile` → `api` | FastAPI: scores, explanations, NBA, review | 8000 |
| `frontend` | `docker/Dockerfile` → `frontend` | Streamlit review console | 8501 |

Agents never call each other over the network: each stage writes to Postgres and the next stage reads it.
That keeps one container per concern without a message broker, and any run can be replayed by `run_id`.

Images are split by target, so the API image has no ML libraries and the frontend has no backend code.

## Run it (Windows · PowerShell · Docker Desktop with WSL2)

```powershell
cd cs_ai_copilot_dss
Copy-Item .env.example .env          # set POSTGRES_PASSWORD; ANTHROPIC_API_KEY optional
docker compose up -d --build
docker compose logs -f pipeline      # first run trains models and creates drafts (~30 s)
```

Then open the review console at http://localhost:8501 and the API docs at http://localhost:8000/docs.

### Useful commands

```powershell
# Force a retrain + fresh run without waiting for the schedule
docker compose run --rm pipeline python -m copilot.pipeline.orchestrator --once --retrain

# Rebuild just one service after a code change
docker compose up -d --build api

# Inspect the database
docker compose exec db psql -U copilot -d copilot -c "select status, count(*) from drafts group by 1;"

# Full reset (deletes data, models and vectors)
docker compose down -v
```

## Configuration (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Empty → deterministic template drafts (no LLM). Set → Claude drafts with citation guardrail. |
| `WRITER_MODEL` | `claude-sonnet-5-5` | Model used by the writer agent |
| `PIPELINE_INTERVAL_SECONDS` | 86400 | Time between monitoring runs |
| `RISK_HIGH` / `RISK_MEDIUM` | 0.65 / 0.35 | Churn-probability cut-offs for risk levels |

The NBA playbook (`playbook/nba_playbook.yaml`) is bind-mounted, so edits take effect on the next run without a rebuild.

## Guardrails

1. No retrieved context for an account → draft status `insufficient_data`; nothing is invented.
2. LLM drafts must cite retrieved document ids; uncited or wrongly cited drafts are discarded.
3. Retrieval is always filtered by `account_id`.
4. Nothing is sent from the system: every draft waits in `pending_review`; decisions are logged in `review_events`.
5. Every review is logged; rejections and edit requests require a reason and an explanation. Approved drafts become examples and rejections become corrections for the writer (see docs/ARCHITECTURE.md, "Feedback log").

## Notebooks (VS Code, no Docker needed)

One notebook per agent, using the same `copilot` package as the containers. They hand off in order:

| Notebook | What it does | Hand-off |
|---|---|---|
| `notebooks/01_monitoring_agent.ipynb` | Rule hit rates, `min_signals` tuning, who was flagged and why | `_dev/flagged.json` |
| `notebooks/02_analyst_agent.ipynb` | Train + evaluate, threshold trade-off, global and per-account SHAP, NBA | `risk_scores` table |
| `notebooks/03_human_review_agent.ipynb` | RAG + writer drafts, interactive approve / edit / reject, audit log, metrics | `review_events` table |

Setup (PowerShell, Python 3.12):

```powershell
cd cs_ai_copilot_dss
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements/base.txt -r requirements/pipeline.txt -r requirements/api.txt -r requirements/notebooks.txt
code .
```

In VS Code, open a notebook, choose **Select Kernel → Python Environments → .venv312**, then **Run All**.
Notebooks write to a local SQLite database in `notebooks/_dev/`. Delete that folder to start fresh.

## Tests (no Docker needed)

```powershell
pip install -r requirements/base.txt -r requirements/pipeline.txt -r requirements/api.txt httpx pytest
$env:PYTHONPATH="src"; pytest -q
```

## Plugging in real data

Replace `copilot.data.load_gold()` with a reader for your Gold table (Databricks SQL connector or a Postgres view),
keeping the column contract in `FEATURES`. Replace `copilot.rag.synthetic_context()` with your CRM / email / ticket export.
Nothing else needs to change.
