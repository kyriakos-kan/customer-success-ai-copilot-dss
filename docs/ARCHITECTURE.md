# Architecture: Customer Success AI Copilot

The copilot helps a B2B textile exporter spot customers at risk of leaving, explain why, and draft outreach for an account manager to approve. It splits into two halves: **the pipeline does the work at night**, and **the API and review console serve the results by day**. Every human decision is recorded in a **feedback log** that the writer learns from on the next run.

```
                    ┌───────────────── NIGHT: pipeline container ─────────────────┐
  ERP / Gold data ─▶│ Monitoring → Analyst → RAG lookup → Playbook → Writer       │◀─────┐
                    └────┬──────────────────────┬───────────────────────┬─────────┘      │
                         │ asks for context      │ writes results         │ saves models   │
                         ▼                       ▼                        ▼                │
                  ┌─────────────┐        ┌───────────────┐        ┌──────────────┐        │ feedback log:
                  │  ChromaDB   │        │   Postgres    │        │ models volume│        │ approved examples
                  │ (documents) │        │ (all results) │        │ (.joblib)    │        │ + corrections
                  └─────────────┘        └───────┬───────┘        └──────┬───────┘        │ (next night)
                                                 │ reads / writes        │ reads metrics  │
                    ┌──────────────── DAY ───────▼───────────────────────▼────────┐       │
                    │                 FastAPI  (api container, :8000)              │───────┘
                    └──────────────────────────────▲──────────────────────────────┘
                                                   │ HTTP requests (JSON)
                    ┌──────────────────────────────┴──────────────────────────────┐
  Reviewer ◀──────▶ │              Streamlit (frontend container, :8501)           │
   (browser)        └──────────────────────────────────────────────────────────────┘
```

All components run as containers defined in `docker-compose.yml`. A laptop demo runs the same code without Docker (see the last section).

---

## Project structure

```
cs_ai_copilot_dss/
├── src/                          ← source code: the program itself
│   └── copilot/                  ← the Python package (import copilot)
│       ├── agents/
│       │   ├── monitoring.py     ← flags accounts whose behaviour changed
│       │   ├── analyst.py        ← CLV + churn models, SHAP explanations
│       │   ├── nba.py            ← next best action from the playbook
│       │   └── writer.py         ← drafts outreach (template / Ollama / Claude) + guardrails
│       ├── pipeline/
│       │   └── orchestrator.py   ← runs the agents in order, on a schedule
│       ├── api/
│       │   └── main.py           ← FastAPI service (endpoints below)
│       ├── config.py             ← reads the .env settings
│       ├── data.py               ← load_gold(): the customer data
│       ├── db.py                 ← database tables (+ automatic upgrade of older databases)
│       ├── feedback.py           ← feedback log: review rules, examples, corrections, training export
│       ├── trust.py              ← trust metrics: review quality, flagged vs churned, control group, model card
│       └── rag.py                ← document search (ChromaDB or in-memory)
├── demo/
│   └── app.py                    ← laptop demo (Streamlit, no Docker)
├── frontend/
│   └── app.py                    ← review console for the Docker stack (Streamlit → API)
├── notebooks/                    ← one notebook per agent, run in VS Code
│   ├── 01_monitoring_agent.ipynb
│   ├── 02_analyst_agent.ipynb
│   └── 03_human_review_agent.ipynb   (includes the feedback log and training export)
├── playbook/
│   └── nba_playbook.yaml         ← editable next-best-action rules
├── docker/
│   └── Dockerfile                ← builds the pipeline, api and frontend images
├── db/
│   └── init.sql                  ← Postgres first-start script
├── requirements/                 ← Python packages per part (base, pipeline, api, frontend, demo, notebooks)
├── tests/
│   └── test_pipeline.py          ← automated checks, including the feedback rules
├── docs/
│   ├── ARCHITECTURE.md           ← this file
│   └── ENV_SETTINGS.md           ← what each .env setting does
├── docker-compose.yml            ← defines the 5 containers
├── run_demo.bat                  ← double-click launcher for the laptop demo (Windows)
├── run_demo.ps1                  ← the demo setup script that run_demo.bat starts
├── .env.example                  ← settings template (copy to .env)
├── .env                          ← your private settings (not shared, not in Git)
├── .gitignore / .dockerignore    ← files Git / Docker should skip
└── README.md                     ← start here
```

Folders created when the project runs (not part of the code): `demo/_data/` (demo database and models), `notebooks/_dev/` (notebook database and exported training set), `.venv-demo/` (Python environment for the demo).

---

## What each component does

| Component | Role | Analogy |
|---|---|---|
| **Pipeline** | Runs the agents on a schedule (`PIPELINE_INTERVAL_SECONDS`, default once a day), trains the models, and applies the feedback log | The night shift |
| **Postgres** | The single source of truth: accounts, risk scores, drafts, and every review decision (the audit and feedback log) | The filing cabinet |
| **ChromaDB** | Stores customer documents (emails, call notes, tickets, contracts) as vectors, so the writer can find the relevant ones by similarity (see the note on embeddings) | The searchable archive of correspondence |
| **FastAPI** | The only door into the data: answers questions like "what's this account's risk?" and records decisions | The front desk |
| **Streamlit** | The screen the reviewer uses; it shows what FastAPI returns and sends decisions back | The office window |

### The agents, in pipeline order

| Agent | File | Input → output |
|---|---|---|
| Monitoring | `agents/monitoring.py` | All accounts → accounts that trip 2+ behaviour rules (order drop, late payments, complaints…) |
| Analyst | `agents/analyst.py` | Flagged accounts → churn probability, 12-month CLV, revenue at risk, top SHAP factors |
| Playbook (NBA) | `agents/nba.py` + `playbook/nba_playbook.yaml` | Risk level, segment, top factors → recommended action and owner |
| Writer | `agents/writer.py` | Action + customer documents + feedback log → draft email, or *insufficient data* |
| Human review | `feedback.py` (used by API, demo, notebook) | Draft → approved / needs edit / rejected, with reason and explanation |

---

## The flows

### 1. Night: the pipeline produces results (writes)

1. **Monitoring** flags accounts whose behaviour changed.
2. **Analyst** scores churn risk and CLV, computes SHAP explanations, and writes them to Postgres (`risk_scores`).
3. The **playbook** picks the action for each flagged account.
4. The **writer asks ChromaDB** for the most relevant documents *for that account only*, and reads the **feedback log** for that type of action.
5. The writer drafts the email, grounded in those documents. The draft goes into Postgres (`drafts`) with status `pending_review`, or `insufficient_data` if there was no customer history to ground it.

Accounts are skipped when they already have a draft waiting for review, or when their outreach was approved within `RECONTACT_COOLDOWN_DAYS` (default 30). An *insufficient data* placeholder is replaced on each run, not stacked.

### 2. Day: the reviewer reads results (reads)

1. The reviewer opens Streamlit at `http://localhost:8501`.
2. Streamlit calls FastAPI, e.g. `GET /drafts` and `GET /accounts/ACC-0955/explanation`.
3. FastAPI reads from Postgres and returns JSON; Streamlit turns it into tables, charts and the draft text, ordered by revenue at risk.

### 3. Day: the reviewer decides (writes back)

1. The reviewer clicks **Approve**, **Needs edit** or **Reject**, optionally editing the text first.
2. Streamlit sends `POST /drafts/{id}/review` to FastAPI.
3. The feedback rules are checked (below). If they pass, the draft's status is updated and a row is added to `review_events`.
4. Nothing is sent to the customer by the system. An approved draft is handed to the account owner's mailbox or CRM.

### 4. Next night: the writer learns (the feedback loop)

The pipeline reads the feedback log and gives the writer approved examples and reviewer corrections. That closes the loop.

---

## Feedback loop: how human review trains the writer

```
   Writer drafts ──▶ Reviewer decides ──┬── Approve ───────────▶ approved example  ──┐
        ▲                               │   (edits optional)     (final text)         │
        │                               │                                             ▼
        │                               └── Needs edit/Reject ─▶ correction        feedback log
        │                                   (reason + explanation  (reason +       (review_events)
        │                                    required)              explanation)       │
        │                                                                              │
        └──────────── next run: examples + corrections added to the writer's prompt ◀──┘
                                                                                       │
                                                         training dataset (JSONL) ◀────┘
```

### The rules

All review decisions go through one function, `record_review()` in `src/copilot/feedback.py`, so the API, the laptop demo and the notebooks apply exactly the same rules.

| Decision | What the reviewer must provide | What it becomes |
|---|---|---|
| **Approve** | Nothing; edits are optional | An **approved example**: the final text (including edits) shows the writer what a good draft looks like |
| **Needs edit** | A **reason** and an **explanation** (at least 10 characters) | A **correction** for the writer, and a negative training record |
| **Reject** | A **reason** and an **explanation** (at least 10 characters) | A **correction** for the writer, and a negative training record |

A decision without the required reason or explanation is refused (HTTP 422 from the API; an error message in the demo and notebook).

**Reason codes**

| Code | Shown to reviewers as |
|---|---|
| `factual_error` | Factually wrong or unsupported claim |
| `wrong_action` | Wrong action for this customer |
| `tone` | Tone not right (too formal / too casual / too pushy) |
| `too_long` | Too long or unclear |
| `missing_context` | Misses something important about the customer |
| `internal_info` | Reveals internal information (risk score, models, notes) |
| `other` | Other (explain below) |

### What is stored

| Table / column | Content |
|---|---|
| `review_events.decision` | approved / needs_edit / rejected |
| `review_events.reason` | Reason code (needs_edit / rejected only) |
| `review_events.comment` | The reviewer's explanation |
| `review_events.original_body` | The draft exactly as the writer produced it |
| `review_events.edited_body` | The reviewer's edited version, if they changed the text |
| `review_events.reviewer`, `seconds_to_decision`, `created_at` | Who decided, how fast, and when (UTC) |
| `drafts.feedback_used` | Which approved examples and corrections a new draft was written with |

### How the writer uses it

For each draft, the pipeline calls `writer_feedback(action, account_id)`, which filters both
approved drafts and reviewer corrections to that exact account before applying action relevance
and recency ordering. Feedback is never shared between accounts. The in-memory cache is keyed by
`(account_id, action)`, and the writer rejects a feedback object whose account ID does not match
the draft being written. The query returns:

- up to **2 approved examples**, preferring the same type of action, newest first
- up to **5 corrections**, preferring the same type of action, newest first

These are added to the writer's prompt below the customer context, for example:

```
Drafts that reviewers APPROVED for similar cases. Match their tone, length and structure,
but never copy their facts; use only the facts in <context>:
<approved_example n="1" action="escalate_quality_review">
Subject: Following up on the recent quality issues
Dear … team, …
</approved_example>
Reviewer CORRECTIONS on past drafts. Do not repeat these mistakes:
- [Tone not right (too formal / too casual / too pushy)] Too formal for this buyer; open with their shrinkage complaint.
```

The writer is told to copy the *style* of approved examples, never their facts. All the usual guardrails still apply: the draft must cite the customer's own documents, or it is discarded.

The same account-scoped block can be previewed before the next run on the demo's **Feedback log**
page, in the API (`GET /feedback?account_id=…&action=…`), and at the end of notebook 03.

### Which accounts are re-drafted

| Previous decision | Next run |
|---|---|
| Still pending review | Skipped (no duplicate drafts) |
| Rejected or needs edit | **New draft**, written with the corrections |
| Approved | Not drafted again for `RECONTACT_COOLDOWN_DAYS` (default 30) |

### Training dataset

Every decision is also available as one record in a training dataset, for evaluating or fine-tuning the writer later:

| Field | Meaning |
|---|---|
| `label` | `positive` (approved) or `negative` (needs edit / rejected) |
| `decision`, `reason`, `reason_label`, `explanation` | The review decision and why |
| `action`, `action_rationale`, `context` | What the writer was asked to do, and the customer documents it had |
| `subject`, `original_body` | What the writer produced |
| `final_body`, `was_edited` | The approved text (the target for positive records), and whether the reviewer edited it |
| `account_id`, `reviewer`, `decided_at` | Traceability |

Ways to get it:

- **Demo:** *Feedback log* page → **Download training dataset (JSON Lines)**
- **API:** `GET /feedback/export`
- **Command line:**
  ```powershell
  $env:PYTHONPATH="src"; python -m copilot.feedback export writer_feedback.jsonl
  ```
- **Notebook 03:** saved to `notebooks/_dev/writer_feedback.jsonl`

### Limits to know

- **Template mode** (no LLM): feedback is logged and exported, but drafts come from fixed templates, so they don't change. With `WRITER_BACKEND=ollama` or `anthropic`, the writer applies it.
- **This is prompt-based learning, not model retraining.** The writer improves by seeing examples and corrections; the language model itself is unchanged. The training dataset is what you'd use later if you want to fine-tune or formally evaluate the writer.
- **Feedback quality matters.** Vague explanations ("bad") teach little, which is why a reason and a real explanation are required.

---

## Trust metrics: how the system proves itself

`src/copilot/trust.py` (API `/trust`, demo page *Trust metrics*) answers four questions:

| # | Question | Measured by | Data needed |
|---|---|---|---|
| 1 | Do our people trust the drafts? | Approval rate, top rejection reasons | `review_events` (available now) |
| 2 | Do flagged accounts really churn? | Churn rate per risk level vs. not flagged; precision and recall at high risk | `outcomes` (imported after the fact) |
| 3 | Does outreach keep customers? | Retention of contacted accounts vs. the control group | `outcomes` + control group |
| 4 | Which model is live, and is it current? | Model version, last retraining date, days since retraining (amber after 90 days) | `models/metrics.json` |

- **Control group:** `CONTROL_GROUP_PCT` (e.g. 20) holds back that share of flagged accounts at random: they are scored but get no draft, and are handled as today. Assignment is fixed per account (a hash of `account_id`), so an account never switches group.
- **Outcomes:** `python -m copilot.trust import-outcomes outcomes.csv` with columns `account_id, period_start, period_end, churned, revenue`. Churn is defined by the business (e.g. no order in the 6 months after `period_start`).
- **Fair checks:** each training run saves the held-out account ids (`models/holdout_accounts.json`). The demo checks predictions only on those, because accounts the model trained on would flatter it.
- **Pipeline vs model:** check 2 measures the whole pipeline (monitoring filter, then the model), so it differs from the model-only test metrics in the model card.

---

## Data model

| Table | One row per | Key columns |
|---|---|---|
| `accounts` | Customer (latest snapshot) | `account_id`, `name`, `segment`, `supplier_origin`, `features` |
| `risk_scores` | Flagged account per pipeline run | `churn_probability`, `clv_score`, `risk_level`, `top_factors`, `monitor_signals` |
| `drafts` | Draft written by the writer | `action`, `subject`, `body`, `context_used`, `status`, `feedback_used` |
| `review_events` | Human decision (the feedback log) | `decision`, `reason`, `comment`, `original_body`, `edited_body`, `reviewer` |
| `outcomes` | Observed outcome per account, imported from the ERP | `account_id`, `period_start`, `period_end`, `churned`, `revenue` |

`risk_scores.cohort` is `copilot` or `control` (pilot control group, see Trust metrics).

Tables are created automatically. When the code adds new columns, older databases (such as an existing `demo/_data/demo.db`) are upgraded automatically on start.

---

## API endpoints

| Method | Endpoint | Returns |
|---|---|---|
| `GET` | `/health` | Service status |
| `GET` | `/accounts` | Flagged accounts, sorted by revenue at risk |
| `GET` | `/accounts/{id}/risk` | Churn probability, CLV, risk level |
| `GET` | `/accounts/{id}/explanation` | Top SHAP factors and monitoring signals |
| `GET` | `/accounts/{id}/next-best-action` | Playbook recommendation |
| `GET` | `/drafts?status=pending_review` | Drafts by status, including `feedback_used` |
| `POST` | `/drafts/{id}/review` | Records a decision. Body: `reviewer`, `decision`, optional `edited_body`; `reason` + `explanation` required for needs_edit / rejected |
| `GET` | `/feedback/reasons` | Reason codes and labels |
| `GET` | `/feedback?account_id=…&action=…` | That account's approved examples, corrections, reason counts, and prompt block |
| `GET` | `/feedback/export` | Training dataset: one record per decision |
| `GET` | `/metrics` | Model metrics, approval rate, queue sizes |
| `GET` | `/trust` | Trust metrics: review quality, flagged vs churned, contacted vs control, model card |

Interactive documentation: `http://localhost:8000/docs`

---

## Why it is split this way

- **Streamlit never talks to the database directly**, only to FastAPI. The screen can be replaced (React, a CRM plugin, a script) without touching the data layer, and access control lives in one place.
- **Heavy work happens at night, not when someone clicks.** Training and scoring take seconds to minutes; the API only reads finished results, so the screen responds instantly.
- **Each part can fail or restart alone.** If the pipeline fails overnight, reviewers still see yesterday's queue.
- **One place for review rules.** Because every decision goes through `record_review()`, the feedback log is consistent whichever screen the reviewer uses.
- **ChromaDB is separate from Postgres because they are good at different things.** Postgres stores exact records; ChromaDB finds the stored documents most similar to a query, using vectors. It is only used while drafting.

### Note on embeddings

ChromaDB only compares vectors; what "similar" means depends on the **embedding function** that turns text into vectors (`_embed()` in `src/copilot/rag.py`).

- **Current (demo):** scikit-learn `HashingVectorizer`, a bag-of-words method (term counts, no IDF). It matches **shared words, not meaning**: "quality problems" finds documents containing "quality", but not "the fabric shrank and the colour was off". Chosen because it needs no model download and is fully deterministic.
- **For real customer text:** switch to semantic embeddings, which match meaning. The local option is Ollama's `nomic-embed-text`, so no data leaves the machine; sentence-transformers (`all-MiniLM-L6-v2`) is another. Only `_embed()` changes.

---

## Laptop demo vs Docker

| | Docker stack | Laptop demo (`run_demo.bat`) |
|---|---|---|
| Database | Postgres container | SQLite file in `demo/_data/` |
| Document search | ChromaDB container | In-memory store |
| API | FastAPI container | None: Streamlit calls the Python code directly |
| Agents and feedback rules | Same code | Same code |
| Writer | Template, Ollama or Claude (`WRITER_BACKEND`) | Same |
| Next run | On schedule (`PIPELINE_INTERVAL_SECONDS`) | **Run next cycle** button in the sidebar |
| Needs Docker | Yes | No |

The demo runs the same agents in one process, which makes it easy to show on a laptop. The Docker stack is the version for a real pilot with several reviewers.
