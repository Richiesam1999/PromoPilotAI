# Agentic Retail Promotion Planner

An end-to-end agentic system that extracts promotional constraints from unstructured business documents, builds an optimized multi-week promotion plan across stores and SKUs, validates it against hard business rules, and presents results in an interactive Streamlit UI.

## What It Does

1. **Input Extraction** — An LLM agent reads messy business documents (brand emails, budget memos, competitor notes, ops notes) and extracts structured promotion constraints with confidence scores.
2. **Relationship Analysis** — Identifies cross-elasticity groups (complementary/substitute SKUs) to inform bundling and cannibalization logic.
3. **Strategy Agent** — Recommends a shortlist of SKUs and discount depths given the extracted constraints and product relationships.
4. **Optimizer** — Solves a MIP (PuLP/CBC) to select the best store × SKU × week × discount combinations, subject to hard rules (budget, margin floors, discount caps, blackouts, consecutive-week limits, etc.).
5. **Validator** — Deterministically checks every hard constraint; rejects non-compliant plans before they surface.
6. **Simulation** — Monte Carlo (200 scenarios) over uncertain elasticities and Poisson demand noise to report outcome bands and clearance hit rates.
7. **Live Re-plan** — Patch any constraint (budget, margin, discount cap, consecutive weeks) and watch the full pipeline re-solve in real time.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Docker + Docker Compose | any recent version (optional) |
| Ollama | optional local fallback |

---

## Installation

```bash
git clone <repo-url>
cd agentic_retail
pip install -r requirements.txt
```

---

## Environment Setup

Copy the template below into a `.env` file in the project root and fill in your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# Optional: local Ollama fallback
OLLAMA_FALLBACK_ENABLED=true
OLLAMA_MODEL=hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CONTEXT_TOKENS=8192
```

> **Note:** Azure OpenAI is required for the LLM agents. If Azure is unavailable or if the data is highly sensitive and can't be exposed, the system falls back to a local Ollama model (when the data has to be private, need to change config settings to use ollama powered local models as default) when `OLLAMA_FALLBACK_ENABLED=true` and Ollama is running with the model installed. Fallback quality is not equivalent to Azure.

---

## Running the App

### Option 1 — Python directly

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2 — Docker Compose

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). The `data/` directory is mounted as a volume so generated files persist.

> **Docker + Ollama:** If using Ollama fallback inside Docker, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in your `.env`.

---

## UI Walkthrough

Navigate using the sidebar. Steps are sequential — each page unlocks the next.

| Page | What to do |
|---|---|
| **1 · Input Extraction** | Click **Run Input Agent** to extract constraints from the bundled sample documents, or upload your own `.txt`/`.xlsx` files. |
| **2 · Plan & Comparison** | Click **Run Strategy Agent** to generate and validate an optimizer plan. Compare KPIs against flat-20% and no-promo baselines. |
| **3 · Simulation** | Click **▶ Run Simulation** to run Monte Carlo scenarios and see outcome bands. |
| **4 · Live Re-plan** | Adjust budget, margin, discount cap, or consecutive-week sliders and click **Re-plan** to see the plan update live. |
| **5 · Validator** | Select the optimizer plan, flat-20% baseline, or build a custom bad plan to see the deterministic constraint checker accept or reject it. |
| **6 · Promotion Showcase** | Final executive view with KPI summary, product cards, and CSV download. |

---

## CLI Usage

Run the full pipeline headlessly:

```bash
python -m src.agents.orchestrator
```

This chains Input → Relationship → Strategy → Optimizer → Validator and prints a summary including a re-plan demo with a 20% budget cut.

---

## Running Tests

```bash
pytest
```

Key test files:

| File | Coverage |
|---|---|
| `tests/test_phase1.py` | Data schema and generation |
| `tests/test_phase2_elasticity.py` | Elasticity fitting |
| `tests/test_phase2_optimizer.py` | Constraint-driven optimization, tight-budget recovery |
| `tests/test_phase3_agents.py` | Agent extraction and scoring |
| `tests/test_phase4_simulation.py` | Monte Carlo simulation |
| `tests/test_phase5_ui.py` | End-to-end UI pipeline, batching, showcase |

---

## API Reference

### `src.agents.orchestrator.Orchestrator`

```python
orch = Orchestrator()
result = orch.run()                              # full pipeline
result = orch.replan({"marketing_budget": 45000})  # patch constraint and re-solve
result = orch.rerun_relationships()              # re-run after elasticity update
```

**`PipelineResult` fields:**

| Field | Type | Description |
|---|---|---|
| `input_result` | `InputResult` | Extracted constraints, confidence flags, warnings |
| `groups` | `list[RelationshipGroup]` | Cross-elasticity product groups |
| `strategy` | `StrategyResult` | SKU shortlist with rationale and suggested discounts |
| `planning` | `dict` | Solver status, KPIs, plan DataFrame, violations, validation log |
| `active_constraints` | `dict` | Constraints used for this solve |
| `replan_log` | `list[str]` | What changed and which stages were re-run |

### `src.optimizer.run_planning`

```python
from src.optimizer import run_planning
result = run_planning(constraints, strategy, est=est, model=model, inv=inv, time_limit=120)
```

Returns a dict with keys: `status`, `plan` (DataFrame), `optimizer`/`flat_20pct`/`no_promo` KPI dicts, `violations`, `validation_log`, `clearance_slack_units`.

### `src.simulation.simulate_scenarios`

```python
from src.simulation import simulate_scenarios
results = simulate_scenarios(plans_dict, est=est, model=model, constraints=constraints, inv=inv, n=200, seed=0)
```

Returns `dict[str, SimulationResult]`. Each `SimulationResult` exposes `.kpi_band(key)`, `.sell_through_band(key)`, `.clearance_hit_rate(key, target)`, `.feasibility_rate`, `.violation_counts`.

---

## Project Structure

```
agentic_retail/
├── app.py                  # Streamlit UI (6 pages)
├── src/
│   ├── agents/
│   │   ├── input_agent.py      # LLM extraction of constraints
│   │   ├── relationship_agent.py  # Cross-elasticity grouping
│   │   ├── strategy_agent.py   # SKU shortlisting
│   │   └── orchestrator.py     # Pipeline orchestration + CLI
│   ├── optimizer.py        # PuLP MIP solver
│   ├── evaluator.py        # KPI evaluation + constraint validation
│   ├── simulation.py       # Monte Carlo simulation
│   ├── demand.py           # Demand model
│   ├── elasticity.py       # Elasticity fitting
│   ├── baseline.py         # Flat-20% baseline plan
│   └── config.py           # Paths, dates, constants
├── data/
│   ├── inputs/             # Sample business documents for agents
│   └── gold/               # Ground-truth constraints for scoring
├── tests/
├── docs/                   # Architecture and phase design docs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pulp` | MIP optimization (CBC solver bundled) |
| `openai` | Azure OpenAI API client |
| `pandas` / `numpy` | Data manipulation |
| `scipy` | Statistical utilities |
| `plotly` | Interactive charts |
| `tiktoken` | Token counting for context guard |
| `python-dotenv` | `.env` loading |
| `openpyxl` | Excel file parsing |

Full pinned versions: [`requirements.txt`](requirements.txt)

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system design including model routing, optimization formulation, re-plan lifecycle, simulation design, and deployment trust boundaries.
