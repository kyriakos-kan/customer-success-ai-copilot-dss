# Settings file (`.env`)

The `.env` file holds the configuration the containers read when they start, so behaviour can be changed without touching the code.

`.env.example` is the template. Copy it to `.env` once and fill in your own values:

```powershell
Copy-Item .env.example .env
code .env
```

## Template

```dotenv
POSTGRES_USER=copilot
POSTGRES_PASSWORD=change-me
POSTGRES_DB=copilot
CHROMA_VERSION=0.5.23
# Writer backend: auto | template | ollama | anthropic
#   auto     = Claude if ANTHROPIC_API_KEY is set, otherwise template
#   ollama   = local LLM on this machine; no data leaves it
WRITER_BACKEND=auto
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
# LLM drafts per run for the top accounts by revenue at risk (rest use templates). 0 = all
LLM_MAX_DRAFTS=15
# Seconds to wait for one local draft before using the template instead
OLLAMA_TIMEOUT_SECONDS=600
# Optional: only used when the backend is anthropic (or auto with a key)
ANTHROPIC_API_KEY=
WRITER_MODEL=claude-sonnet-5-5
# Pipeline schedule: seconds between monitoring runs (86400 = nightly)
PIPELINE_INTERVAL_SECONDS=86400
# High-risk threshold for churn probability
RISK_HIGH=0.65
RISK_MEDIUM=0.35
# Days before an account whose outreach was approved can be drafted again
RECONTACT_COOLDOWN_DAYS=30
# Pilot control group: % of flagged accounts that get no draft (handled as today). 0 = off
CONTROL_GROUP_PCT=0
```

Lines starting with `#` are comments and are ignored.

## What each setting does

| Setting | What it does | Change it? |
|---|---|---|
| `POSTGRES_USER` | Username for the database container | Optional |
| `POSTGRES_PASSWORD` | Password for that database | **Yes**, set your own before the first start |
| `POSTGRES_DB` | Name of the database created inside Postgres | No |
| `CHROMA_VERSION` | Version of the Chroma vector store image | No; pinned to match the Python client |
| `WRITER_BACKEND` | Who writes the drafts: `template` (no LLM), `ollama` (local LLM), `anthropic` (Claude), or `auto` (Claude if a key is set, else template) | Set `ollama` for a local LLM with no data leaving the machine |
| `OLLAMA_URL` | Where Ollama is running. In Docker this is set automatically to the host machine | No |
| `OLLAMA_MODEL` | Which local model to use, e.g. `mistral:7b`. Pull it first with `ollama pull` | Optional |
| `LLM_MAX_DRAFTS` | How many of the top accounts (by revenue at risk) get an LLM-written draft per run; the rest use templates. `0` = all | Lower it (e.g. `5`) if Ollama is slow on your laptop |
| `OLLAMA_TIMEOUT_SECONDS` | How long to wait for one local draft before using the template instead | Raise it if a slow laptop still falls back to templates |
| `ANTHROPIC_API_KEY` | Key for Claude. Only used by the `anthropic` backend (or `auto` when set); empty means no cloud LLM is ever called | Leave empty until the company approves sending data to an LLM provider |
| `WRITER_MODEL` | Which Claude model writes drafts (only used if a key is set) | No |
| `PIPELINE_INTERVAL_SECONDS` | How often the pipeline runs. `86400` = once a day | Optional, e.g. `3600` (hourly) while testing |
| `RISK_HIGH` | Churn probability at or above which an account is **high** risk | Tune using the threshold chart in `notebooks/02_analyst_agent.ipynb` |
| `RISK_MEDIUM` | Churn probability at or above which an account is **medium** risk; below it is **low** | Same as above |
| `RECONTACT_COOLDOWN_DAYS` | After a draft is approved, that account isn't drafted again for this many days | Optional |
| `CONTROL_GROUP_PCT` | Share of flagged accounts held back at random (scored, but no draft), so retention can be compared fairly on the Trust metrics page. The laptop demo uses 20 | Set `20` for the pilot |

## Things to know

- **The password only applies on the very first start.** Postgres creates the database with whatever password `.env` has at that moment. If you change it later, reset once with `docker compose down -v` (this **deletes** stored data), then start again.
- **Keep `.env` private.** It is excluded from Git by `.gitignore`, so passwords and API keys never reach GitHub. Share only `.env.example`, which contains no real secrets.
- **Restart after editing.** Run `docker compose up -d` so the containers pick up the new values.

## Data and privacy

With `WRITER_BACKEND=template` or `ollama` (or `auto` with no key), nothing leaves the machine: models, database, document search and, with Ollama, the LLM itself all run locally. If a key is added later for real customer data:

1. Get company approval and check the provider's API data terms (retention, training use), ideally with a data processing agreement (GDPR).
2. Pseudonymise customer names and remove personal details from the context documents.
3. Send only the documents relevant to the recommended action.
4. Keep the human approval step. Nothing is sent to customers automatically.
