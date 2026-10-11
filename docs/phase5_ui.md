# Phase 5: UI and Demo (written 2026-10-11)

Status: **code complete**. Run with `streamlit run app.py`.

## Run
```
streamlit run app.py
```

Prerequisites: install dependencies, generate the retail dataset, and configure Azure OpenAI credentials. Saved optimizer/baseline result files are no longer used by the live UI. Gold constraints are used only to score extraction accuracy.

## Pages

Previous and Next buttons connect the five steps; Next is enabled after the
current step finishes successfully. The sidebar remains available for direct
navigation. Input extraction, strategy generation, simulation, re-planning,
and validation run only on their respective button clicks, not on navigation
or widget changes. KPIs appear only after a successful session solve and validation.
Completed extraction, strategy, simulation, and re-plan outputs remain visible
for the current session. Re-running extraction clears the dependent strategy
and re-plan outputs, the current plan, and simulation output. Changed sliders take effect on the next Run click.

### 1 · Input Extraction
- Large uploads are split into token-bounded batches (at most 6,000 input tokens each). Each batch extracts explicit policy evidence; source-linked, deduplicated facts are consolidated into final constraints in a separate bounded request. Source text is not silently truncated. Review warnings and low-confidence conflicts before planning. This requires multiple model calls and can take longer than small-document extraction.
- Oversized evidence is automatically minimized in groups capped at 3,000 input tokens, up to four rounds. Sources and numeric values are checked for preservation; conflicting values remain for review. Invalid evidence, changed values, non-shrinking compaction or truncated responses stop extraction without publishing partial constraints. Batching/compaction do not guarantee factual accuracy. Uploads are capped at 256 batches.
- Select sample documents or upload multiple UTF-8 `.txt` / `.md` documents or `.xlsx` workbooks (1 MB per file, 1 MB combined). Non-empty Excel sheets become labeled text tables for extraction; spreadsheet uploads do not replace retail dataset tables. Workbook expanded contents are limited to 100 MB and extracted text to 500,000 characters. Uploaded documents replace the sample document set; dataset files are not overwritten.
- Uploaded text feeds the same extraction and validated planning pipeline. Changing source/files clears downstream results; custom uploads do not display a demo gold accuracy score.
- Shows the 4 raw input documents in expandable tabs
- "Run Input Agent" button calls `src/agents/input_agent.py` (cached)
- Displays: extraction accuracy vs gold, extracted constraints JSON, confidence flags per field, warnings raised
- Shows a diff table of any misses vs gold

### 2 · Plan & Comparison
- Runs Strategy Agent, optimizer, evaluator and validator using extracted constraints; stores the validated plan and KPIs in the session
- KPI cards for Optimizer / Flat 20% / No Promo, all delta'd against No Promo
- 4 side-by-side bar charts: Net Contribution, Gross Margin %, Marketing Spend, Revenue
- Clearance sell-through table with ✅/❌ per scenario vs target
- Solver status, hard-rule violations, buffered planning slack (units), and evaluated clearance shortfalls (percentage points)
- "Run Strategy Agent" generates the shortlist and a validated solver plan using the completed extraction;
  the button is disabled until extraction has run in the current session

### 3 · Simulation
- Slider for number of scenarios (50–500, default 200)
- "Run Simulation" button calls `src/simulation.py` with the current session plan and its active constraints (cached)
- Scenario selector: Optimizer / Flat 20% / No Promo
- Structural feasibility rate metric
- KPI band table (p10 / p50 / p90 + spread %)
- Clearance sell-through bands + hit rate per target
- Net contribution histogram with p10/p50/p90 vertical lines
- Violation counts table across all scenarios

### 4 · Live Re-plan
- Sliders for 4 constraints: budget, min blended margin, max discount %, max consecutive weeks
- Shows a diff of what changed from the extracted constraints
- "Re-plan" patches active constraints and re-runs strategy, optimizer, evaluator and validator
- Displays solver status, before/after KPIs, the new plan, clearance misses, and any validator retry log
- CBC gets 15 seconds per attempt, at most two attempts; failed solves leave the previous valid plan unchanged
- Demonstrates the demo story step: "change a constraint, watch the plan re-adapt"

### 5 · Validator
- Radio: Optimizer plan / Flat 20% (bad plan) / Custom bad plan
- Custom bad plan builder: pick SKU, discount, store — designed to trigger specific violations
  (D001=no-promo, B001=FizzUp cap, D008=Roast Masters cap, P004=West blackout)
- "Validate Plan" evaluates the selected plan and runs `check_constraints()` against current extracted/patched constraints (no LLM)
- Shows REJECTED (red) or ACCEPTED (green) with full violation list
- KPI summary + sell-through with target delta colours

## Caching strategy

| Resource | Cache type | Notes |
|---|---|---|
| Elasticity model (`fit()`) | `@st.cache_resource` | Fitted once per process, ~3s |
| Demand model | `@st.cache_resource` | Loaded once per process |
| Input agent result | `@st.cache_data` | Cleared by "Run Input Agent" button |
| Relationship agent result | `@st.cache_data` | Depends on elasticity model |
| Strategy agent result | `@st.cache_data` | Keyed on constraints dict JSON; cleared by "Re-plan" button |
| Simulation results | `@st.cache_data` | Keyed on plan, constraints and n_scenarios; session output invalidated on each successful solve |
| Optimizer / baseline results | Session state | Computed from the same active constraints; never loaded from saved result files |

## Demo script (follows CHECKPOINTS.md demo story)

1. **Page 1** — click "Run Input Agent". Point out: $59k budget (not $71k superseded), $5k contingent excluded, FreshCo promo flagged as expired, MegaMart rumour excluded, audience confidence = low.
2. **Page 2** — click "Run Strategy Agent" to generate and validate a plan. Report its actual solver status and current KPIs, not the standalone saved benchmark numbers. Explain clearance misses and planning slack if present. The solver chooses among shortlisted SKUs plus clearance/pairing dependencies; advisory BOGO/bundle suggestions are not executed by this optimizer.
3. **Page 3** — run simulation and report the measured feasibility rate rather than promising a threshold. Show C008 sell-through band and hit rate; deterministic target satisfaction does not guarantee clearance under uncertainty.
4. **Page 4** — reduce the budget below the current plan's spend and click Re-plan. Compare before/after spend and contribution, inspect the new plan, and show zero hard-rule violations. Simulation must be rerun for this new plan.
5. **Page 5** — select "Flat 20% (bad plan)", click Validate, and report the measured violations under the patched constraints. Then validate "Optimizer plan": it is the current re-plan, not the saved CSV. Clearance misses are warnings, not hard-rule rejection.

## Next step
Phase 6: submission artefacts — architecture diagram, 9-block self-assessment, demo script, final evaluation check.

## UI tests
Run `python -m pytest tests/test_phase5_ui.py -q`. The tests mock expensive
operations and verify explicit Run clicks, connected navigation, retained
results, and that navigation or widget changes do not trigger computation.
