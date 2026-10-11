# Phase 3: Agent Layer (written 2026-10-11)

Status: **code complete, not yet run**. Run the commands below locally and tick the Phase 3 boxes in CHECKPOINTS.md once they pass.

## Run
```
python -m src.agents.input_agent          # standalone: extracts constraints, prints accuracy vs gold
python -m src.agents.relationship_agent   # standalone: prints relationship groups
python -m src.agents.strategy_agent       # standalone: prints shortlist + rationale
python -m src.agents.orchestrator         # standalone: full pipeline + replan demo
python -m pytest tests/test_phase3_agents.py -q   # 17 tests (all marked 'llm')
```

Skip LLM tests when offline:
```
python -m pytest -m "not llm" -q
```

## What was added
| File | Purpose |
|---|---|
| `src/agents/_llm.py` | Thin Azure OpenAI wrapper: `call()` and `call_json()`, reads creds from `.env` |
| `src/agents/input_agent.py` | Reads 4 messy text inputs → structured constraints JSON + confidence flags + warnings |
| `src/agents/relationship_agent.py` | Combines `est.pairs()` + basket affinity → named substitute/complement groups |
| `src/agents/strategy_agent.py` | Constraints + groups + inventory pressure → SKU shortlist, mechanism, rationale |
| `src/agents/orchestrator.py` | Chains all three agents; re-runs only affected stages on constraint change |
| `tests/test_phase3_agents.py` | 17 LLM-backed tests covering extraction accuracy, trap handling, constraint compliance, re-planning |

## Architecture

```
text inputs (4 files)
        │
        ▼
  input_agent  ──── confidence flags, warnings
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
relationship_agent                             strategy_agent
(est.pairs + basket affinity)          (constraints + groups + inventory)
        │                                              │
        └──────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                              StrategyResult
                                     (shortlist, bundles, rationale)
```

The **Orchestrator** is stateful and caches each stage. Re-planning dependency rules:

| Trigger | Stages re-run |
|---|---|
| Constraint override (`replan()`) | strategy_agent, optimizer, evaluator and validator |
| New text inputs | all agents, optimizer, evaluator and validator |
| Elasticity model update (`rerun_relationships()`) | relationship_agent, strategy_agent, optimizer, evaluator and validator |

Constraint overrides also re-run the optimizer, evaluator and validator. The shortlist reaches `optimize(candidate_skus=...)`; clearance SKUs and promote-together dependencies remain eligible even if omitted by the LLM. Suggested mechanisms and discount depths remain advisory: the solver produces percentage-off promotions on its supported grid.

`PipelineResult.planning` contains the validated plan, solver status, KPIs, soft clearance misses and retry log. `active_constraints` records the extracted/patched constraints used for that result. Gold constraints are used for extraction scoring, not live planning. Patches accumulate; a failed solve or exhausted validator retry does not replace the last validated re-plan.

Live planning uses 15 seconds per CBC solve, with at most one additional solve after budget or blended-margin validation failure. The retry tightens the approximation but always validates against the original active constraints. A feasible time-limited incumbent is labeled `Feasible`, never `Optimal`; no solution or remaining hard-rule violations raise an error. There is no precomputed fallback.

## Input Agent — trap handling

The system prompt explicitly instructs the LLM on all 7 known traps from `gold/extraction_traps.json`:

| Trap | Handling |
|---|---|
| Superseded budget ($71k → $59k) | Use only the memo marked "FINAL / supersedes Sept version" |
| Contingent $5k | Goes to `contingent_budget_not_to_plan_on`, never added to `marketing_budget` |
| FreshCo SoftRoll promo (expires before window) | Excluded from constraints; flagged in warnings |
| MegaMart butter rumour (unverified hearsay) | Excluded; flagged in warnings |
| Informal product names ("the big cola", "4 cheese") | Mapped to B001, P005 via SKU reference in prompt |
| Mutual exclusion vs promote-together | "never run X at same time as Y" → `mutually_exclusive_same_week`; "together or not at all" → `promote_together` |
| Missing audience guidance | Auto-flagged: `confidence["audience"] = "low"`, warning added |

### Output schema
```python
InputResult.constraints   # dict matching gold/constraints.json schema
InputResult.confidence    # {"marketing_budget": "high", "audience": "low", ...}
InputResult.warnings      # ["Superseded budget $71k ignored", "FreshCo promo expired ...", ...]
```

### Scoring
`score(result, gold)` compares 13 fields against gold and returns `{hits, misses, accuracy}`.
Test threshold: **≥ 80% field accuracy**.

## Relationship Agent

### Method
1. Compute basket co-purchase lift from `basket_items.csv` (top 30 pairs by lift score).
2. Pull top 40 cross-elasticity pairs from `est.pairs()` (already screened for statistical significance in Phase 2a).
3. LLM merges both signals into named groups with `relation`, `planning_note`, and `cannibalization_risk`.

### Output schema
```python
RelationshipGroup.group_id            # short identifier
RelationshipGroup.name                # e.g. "FizzUp Cola Substitutes"
RelationshipGroup.relation            # "substitute" | "complement" | "mixed"
RelationshipGroup.skus                # ["B001", "B002"]
RelationshipGroup.planning_note       # "Discounting B001 will cannibalise B002 — promote only one."
RelationshipGroup.cannibalization_risk # "high" | "medium" | "low" | "none"
```

### Expected groups (from known world)
- FizzUp Cola substitutes: B001 + B002 (cannibalization_risk=high, |C|≈0.45)
- Chips + Salsa complements: S001/S002/S003 + S005 (halo opportunity)
- Pasta + Sauce complements: P001/P003 + P002 (bundle candidate)
- Shampoo + Conditioner complements: C001 + C002 (promote_together constraint)
- Cereal + Milk complements: D005/D006 + D001 (note: D001 is no_promo)

## Strategy Agent

### Inputs
- Extracted constraints (from input_agent)
- Relationship groups (from relationship_agent)
- Inventory pressure: top 12 SKUs by weeks-of-cover (stock / avg weekly sales)
- Own-price elasticities per SKU (from `est.E`)

### Output schema
```python
StrategyResult.shortlist          # list of {sku, name, mechanism, suggested_discount_pct, priority, rationale, cannibalization_warning}
StrategyResult.bundle_suggestions # list of {skus, rationale}
StrategyResult.excluded_skus      # list of {sku, reason}
StrategyResult.overall_rationale  # 2-3 sentence summary
```

### Priority levels
| Priority | Meaning |
|---|---|
| `must_promote` | Has a clearance target in constraints (C008, P006, S008, D002) |
| `high` | High inventory pressure or strong elasticity response |
| `medium` | Moderate opportunity, fits within budget |
| `low` | Nice-to-have; first to drop if budget is tight |

### Mechanism selection logic (LLM-guided)
- `pct_off` — clearance SKUs (simple, measurable sell-through)
- `bogo` — high-volume staples with strong own-elasticity
- `bundle` — complement pairs with strong basket affinity

## Orchestrator

### Re-planning demo (shown in `python -m src.agents.orchestrator`)
1. Full pipeline run → prints extracted budget, groups, shortlist.
2. `replan({"marketing_budget": new_budget})` → patches active constraints, re-runs strategy and deterministic planning, prints status and spend.

### Replan log example
```
constraint 'marketing_budget' changed: 59000 → 47200
strategy_agent re-run; optimizer, evaluator and validator re-run; input_agent and relationship_agent results reused from cache
```

## Test coverage (tests/test_phase3_agents.py)

| Test | What it checks |
|---|---|
| `test_input_extraction_accuracy` | ≥ 80% field accuracy vs gold |
| `test_input_correct_budget` | Final $59k extracted, not superseded $71k |
| `test_input_contingent_budget_excluded` | $5k in contingent field, not in budget |
| `test_input_no_promo_sku` | D001 in no_promo_skus |
| `test_input_mutually_exclusive` | B001/B002 pair present |
| `test_input_promote_together` | C001/C002 pair present |
| `test_input_blackouts_present` | P004 + P005 blacked out in West |
| `test_input_clearance_targets` | All 4 clearance SKUs extracted |
| `test_input_expired_competitor_promo_excluded` | FreshCo trap flagged in warnings |
| `test_input_unverified_rumour_excluded` | MegaMart trap flagged in warnings |
| `test_input_confidence_flags_present` | audience confidence = low |
| `test_input_brand_caps` | FizzUp 25%, Roast Masters 15%, SoftRoll 20% |
| `test_relationship_groups_nonempty` | ≥ 3 groups returned |
| `test_relationship_has_substitute_and_complement` | Both relation types present |
| `test_relationship_fizz_up_substitutes` | B001 + B002 in same substitute group |
| `test_relationship_shampoo_conditioner_complement` | C001 + C002 in complement group |
| `test_relationship_cannibalization_risk_set` | Valid enum value on every group |
| `test_strategy_must_promote_clearance_skus` | C008, P006, S008, D002 all must_promote |
| `test_strategy_no_promo_sku_excluded` | D001 not in shortlist |
| `test_strategy_brand_caps_respected` | Suggested discounts within brand caps |
| `test_strategy_shortlist_size` | 5–20 SKUs |
| `test_strategy_has_rationale` | Non-empty rationale on every item |
| `test_orchestrator_full_run` | Pipeline completes end-to-end |
| `test_orchestrator_replan_only_reruns_strategy` | relationship_agent cache not invalidated |
| `test_orchestrator_replan_constraint_reflected` | C008 still must_promote after budget cut |

## LLM usage notes
- Optional local fallback: `hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M` served by Ollama. `OLLAMA_FALLBACK_ENABLED=true` enables it after Azure availability/authentication/deployment errors; Azure bad requests are not retried locally. Azure remains primary and fallback use is logged.
- Setup: `ollama pull hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M`, then ensure Ollama is running (desktop app or `ollama serve`). The local model was already installed and a real JSON health request succeeded on 2026-10-11.
- Ollama uses `http://127.0.0.1:11434` locally and `http://host.docker.internal:11434` in Docker Desktop. Override with `OLLAMA_BASE_URL` if needed; a container requires access to the host Ollama service. No Ollama container or deployment restart is performed automatically.
- Local context defaults to 8,192 tokens (`OLLAMA_CONTEXT_TOKENS`), reserving 2,048 for output. Oversized prompts are rejected without dropping rows. The 2B fallback is not guaranteed to match Azure extraction/planning quality; normal solver/validator gates still apply. JSON calls request Ollama's JSON mode.
- Model: `gpt-4o-mini` (Azure deployment name from `AZURE_OPENAI_DEPLOYMENT` in `.env`; code fallback also `gpt-4o-mini`). The Azure resource must have a deployment with that name.
- Temperature: 0.0 for all agents (deterministic extraction and planning)
- All agents return structured JSON; `call_json()` strips markdown fences before parsing
- LLM agents do not read gold demand parameters. The deterministic evaluation step uses the synthetic demand world to score and validate plans; this is a prototype simulation, not a production demand guarantee.

## Next step
Phase 4: Monte Carlo simulation — `src/simulation.py`, using `est.draw(rng)` to perturb elasticities and `evaluate_plan(..., noise=True)` to get a distribution of outcomes (revenue, net contribution, sell-through) for the optimizer plan.
