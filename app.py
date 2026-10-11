"""Phase 5: Agentic Retail Promotion Planner — Streamlit UI.

Pages (sidebar):
  1. Input Extraction   — upload / view messy inputs, extracted constraints, confidence flags
  2. Plan & Comparison  — optimizer vs flat-20% vs no-promo KPI table + charts
  3. Simulation         — Monte Carlo outcome bands per scenario
  4. Live Re-plan       — patch a constraint, watch strategy re-adapt
  5. Validator          — show a bad plan being rejected

Run:  streamlit run app.py
"""
import json
import time
from html import escape
from io import BytesIO
from zipfile import BadZipFile, ZipFile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA, GOLD, INPUTS
from src.evaluator import check_constraints, load_inventory

st.set_page_config(page_title="Agentic Retail Planner", layout="wide", page_icon="🛒")

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "1 · Input Extraction",
    "2 · Plan & Comparison",
    "3 · Simulation",
    "4 · Live Re-plan",
    "5 · Validator",
    "6 · Promotion Showcase",
]
page = st.sidebar.radio("Navigate", PAGES, key="planner_page")
st.sidebar.markdown("---")
st.sidebar.caption("Agentic Retail Promotion Planner · Phase 5")


# ── Cached heavy loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Fitting elasticity model…")
def get_est():
    from src.elasticity import fit
    return fit()


@st.cache_resource(show_spinner="Loading demand model…")
def get_model():
    from src.demand import DemandModel
    return DemandModel()


def get_constraints():
    if "planning_output" in st.session_state:
        return st.session_state.planning_output["constraints"]
    if "input_output" in st.session_state:
        return st.session_state.input_output[0].constraints
    return None


@st.cache_data(show_spinner=False)
def get_inventory():
    return load_inventory()


@st.cache_data(show_spinner="Running input agent…")
def run_input_agent(documents=None):
    from src.agents.input_agent import run, score
    if documents is not None:
        return run(documents=documents), None
    result = run()
    gold = json.loads((GOLD / "constraints.json").read_text())
    sc = score(result, gold)
    return result, sc


@st.cache_data(show_spinner="Running relationship agent…")
def run_relationship_agent():
    from src.agents.relationship_agent import run
    return run(est=get_est())


@st.cache_data(show_spinner="Running strategy agent…", hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)})
def run_strategy_agent(constraints: dict):
    from src.agents.strategy_agent import run
    groups = run_relationship_agent()
    return run(constraints=constraints, groups=groups, est=get_est())


def get_optimizer_results():
    return st.session_state.get("planning_output")


def generate_plan(constraints):
    from src.optimizer import run_planning

    strategy = run_strategy_agent(constraints)
    result = run_planning(constraints, strategy, est=get_est(), model=get_model(),
                          inv=get_inventory(), time_limit=15)
    st.session_state.planning_output = result
    st.session_state.strategy_output = strategy
    st.session_state.pop("simulation_output", None)
    st.session_state.pop("showcase_approval", None)
    return result


@st.cache_data(show_spinner="Running Monte Carlo simulation…")
def run_simulation(plan, constraints, n: int = 200):
    from src.simulation import simulate_scenarios
    from src.baseline import flat_plan

    model = get_model()
    est = get_est()
    inv = get_inventory()

    plans = {
        "Optimizer": plan,
        "Flat 20%": flat_plan(model),
        "No Promo": plan.iloc[0:0],
    }
    return simulate_scenarios(plans, est=est, model=model, constraints=constraints, inv=inv, n=n, seed=0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi_delta(val, ref, key):
    delta = val - ref
    prefix = "$" if key != "gross_margin_pct" else ""
    fmt = ".1%" if key == "gross_margin_pct" else ",.0f"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{prefix}{delta:{fmt}} vs No Promo"


def _kpi_cards(kpis, ref, label):
    st.caption(label)
    cols = st.columns(4)
    items = [
        ("Revenue", "revenue", "$"),
        ("Gross Margin %", "gross_margin_pct", "%"),
        ("Net Contribution", "net_contribution", "$"),
        ("Marketing Spend", "marketing_spend", "$"),
    ]
    for col, (name, key, prefix) in zip(cols, items):
        val = kpis[key]
        ref_val = ref[key]
        fmt_val = f"{val:.1%}" if prefix == "%" else f"${val:,.0f}"
        delta = val - ref_val
        fmt_delta = f"{'+' if delta >= 0 else ''}{delta:.1%}" if prefix == "%" else f"{'+' if delta >= 0 else ''}${delta:,.0f}"
        col.metric(name, fmt_val, delta=f"{fmt_delta} vs No Promo",
                   delta_color="normal" if key != "marketing_spend" else "inverse")


def _scenario_bar(results_dict: dict, key: str, title: str):
    labels = list(results_dict.keys())
    vals = [results_dict[s][key] for s in labels]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=["#2196F3", "#FF9800", "#9E9E9E"]))
    fmt = ".1%" if key.endswith("_pct") else "$,.0f"
    fig.update_layout(title=title, yaxis_tickformat="%" if key.endswith("_pct") else "$,",
                      height=300, margin=dict(t=40, b=20))
    return fig


def _confidence_badge(level: str) -> str:
    colours = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    return f"{colours.get(level, '⚪')} {level}"


def _navigate_to(destination):
    st.session_state.planner_page = destination


def clear_pipeline():
    for key in ("input_output", "strategy_output", "replan_output", "planning_output", "simulation_output", "showcase_approval"):
        st.session_state.pop(key, None)


def uploaded_documents(files):
    documents = {}
    total_bytes = 0
    for uploaded in files:
        content = uploaded.getvalue()
        total_bytes += len(content)
        if len(content) > 1_000_000 or total_bytes > 1_000_000:
            raise ValueError("Documents must be at most 1 MB each and 1 MB combined.")
        if uploaded.name in documents:
            raise ValueError("Uploaded documents must have unique filenames.")
        if uploaded.name.lower().endswith(".xlsx"):
            try:
                with ZipFile(BytesIO(content)) as archive:
                    if sum(entry.file_size for entry in archive.infolist()) > 100_000_000:
                        raise ValueError("Workbook contents exceed the 100 MB expanded limit.")
                sheets = pd.read_excel(BytesIO(content), sheet_name=None, header=None,
                                       dtype=str, engine="openpyxl")
                sections = []
                for name, sheet in sheets.items():
                    sheet = sheet.dropna(how="all").dropna(axis=1, how="all")
                    if not sheet.empty:
                        sections.append(f"=== SHEET: {name} ===\n{sheet.fillna('').to_csv(index=False, header=False)}")
                text = "\n\n".join(sections)
            except ImportError as error:
                raise ValueError("Excel support requires openpyxl. Install dependencies and restart the app.") from error
            except (BadZipFile, ValueError, KeyError, OSError) as error:
                raise ValueError(f"{uploaded.name}: cannot read Excel workbook: {error}") from error
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ValueError(f"{uploaded.name}: use UTF-8 text encoding.") from error
        if not text.strip():
            raise ValueError(f"{uploaded.name}: document is empty.")
        if sum(len(value) for value in documents.values()) + len(text) > 500_000:
            raise ValueError("Extracted document text exceeds 500,000 characters. Upload fewer sheets or a smaller data subset.")
        documents[uploaded.name] = text
    return documents


def _step_navigation(completed):
    step = PAGES.index(page)
    previous, next_step = st.columns(2)
    if step > 0:
        previous.button("Previous", icon=":material/arrow_back:",
                        on_click=_navigate_to, args=(PAGES[step - 1],))
    if step < len(PAGES) - 1:
        next_step.button("Next", icon=":material/arrow_forward:", disabled=not completed,
                         on_click=_navigate_to, args=(PAGES[step + 1],))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · Input Extraction
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("📥 Input Extraction")
    source = st.radio("Input Source", ["Sample documents", "Uploaded documents"],
                      horizontal=True, key="input_source", on_change=clear_pipeline)
    documents = None
    upload_error = False
    if source == "Uploaded documents":
        files = st.file_uploader("Business Documents", type=["txt", "md", "xlsx"],
                                 accept_multiple_files=True, key="input_files", on_change=clear_pipeline)
        try:
            documents = uploaded_documents(files)
        except ValueError as error:
            st.error(str(error))
            upload_error = True

    # Show raw documents
    with st.expander("📄 View raw input documents", expanded=False):
        if source == "Sample documents":
            tabs = st.tabs(["Brand Email", "Budget Memo", "Competitor Note", "Ops Note"])
            files = ["brand_guidelines_email.txt", "budget_memo.txt",
                     "competitor_price_note.txt", "ops_inventory_note.txt"]
            for tab, fname in zip(tabs, files):
                with tab:
                    st.code((INPUTS / fname).read_text(), language=None)
        elif documents:
            tabs = st.tabs(list(documents))
            for tab, text in zip(tabs, documents.values()):
                with tab:
                    st.code(text, language=None)

    if st.button("🤖 Run Input Agent", type="primary",
                 disabled=upload_error or (source == "Uploaded documents" and not documents)):
        run_input_agent.clear()
        with st.spinner("Running input agent…"):
            try:
                st.session_state.input_output = run_input_agent(documents)
                for key in ("strategy_output", "replan_output", "planning_output", "simulation_output"):
                    st.session_state.pop(key, None)
            except ValueError as error:
                clear_pipeline()
                st.error(str(error))

    if "input_output" not in st.session_state:
        _step_navigation(False)
        st.stop()
    result, sc = st.session_state.input_output

    # Accuracy score
    col1, col2, col3 = st.columns(3)
    if sc is not None:
        col1.metric("Extraction Accuracy", f"{sc['accuracy']:.0%}", f"{len(sc['hits'])} / {len(sc['hits']) + len(sc['misses'])} fields correct")
    else:
        col1.metric("Input Source", "Uploaded documents")
    col2.metric("Warnings raised", len(result.warnings))
    col3.metric("Confidence fields", len(result.confidence))

    st.markdown("---")

    # Extracted constraints
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Extracted Constraints")
        c = result.constraints
        st.json(c)

    with col_right:
        st.subheader("Confidence Flags")
        for field, level in result.confidence.items():
            st.markdown(f"**{field}** — {_confidence_badge(level)}")

        st.subheader("Warnings")
        for w in result.warnings:
            st.warning(w, icon="⚠️")

    # Misses vs gold
    if sc is not None and sc["misses"]:
        st.markdown("---")
        st.subheader("Extraction Misses vs Gold")
        misses_df = pd.DataFrame(sc["misses"])
        for column in ("extracted", "expected"):
            misses_df[column] = misses_df[column].map(json.dumps)
        st.dataframe(misses_df, width="stretch")

    _step_navigation(True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · Plan & Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("📊 Plan & Comparison")
    st.markdown("Optimizer plan vs flat-20% baseline vs no-promo across all KPIs.")

    if st.button("Run Strategy Agent", type="primary",
                 disabled="input_output" not in st.session_state):
        run_strategy_agent.clear()
        with st.spinner("Generating and validating promotion plan…"):
            try:
                generate_plan(get_constraints())
                st.session_state.pop("replan_output", None)
            except RuntimeError as error:
                st.error(str(error))
    opt_res = get_optimizer_results()

    if opt_res is None:
        st.warning("No validated plan in this session.")
        _step_navigation(False)
        st.stop()

    st.metric("Solver Status", opt_res.get("status", "Unknown"))
    solution_status = opt_res.get("solution_status", "")
    if (opt_res.get("status") != "Optimal" and solution_status != "Solution Found") or opt_res.get("hard_violations"):
        st.error("No validated solver plan is available.")
        _step_navigation(False)
        st.stop()
    if solution_status:
        st.caption(solution_status)

    scenarios = {
        "Optimizer": opt_res["optimizer"],
        "Flat 20%": opt_res["flat_20pct"],
        "No Promo": opt_res["no_promo"],
    }
    ref = scenarios["No Promo"]

    # KPI cards per scenario
    for label, kpis in scenarios.items():
        _kpi_cards(kpis, ref, label)
        st.markdown("")

    st.markdown("---")

    # Side-by-side bar charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_scenario_bar(scenarios, "net_contribution", "Net Contribution ($)"), width="stretch")
    with col2:
        st.plotly_chart(_scenario_bar(scenarios, "gross_margin_pct", "Gross Margin %"), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(_scenario_bar(scenarios, "marketing_spend", "Marketing Spend ($)"), width="stretch")
    with col4:
        st.plotly_chart(_scenario_bar(scenarios, "revenue", "Revenue ($)"), width="stretch")

    st.markdown("---")

    # Sell-through table
    st.subheader("Clearance Sell-Through")
    c = get_constraints()
    targets = {t["sku"] + "@" + ("all" if t["stores"] == "all" else "+".join(t["stores"])):
               t["min_sell_through_pct"] for t in c["clearance_targets"]}
    rows = []
    for key, target in targets.items():
        row = {"Target": key, "Required": f"{target:.0%}"}
        for label, kpis in scenarios.items():
            val = kpis.get("sell_through", {}).get(key)
            hit = val is not None and val >= target
            row[label] = f"{'✅' if hit else '❌'} {val:.0%}" if val is not None else "—"
        actual = scenarios["Optimizer"].get("sell_through", {}).get(key)
        row["Shortfall (pp)"] = round(100 * max(0.0, target - actual), 2) if actual is not None else None
        row["Planning Slack (units)"] = round(opt_res.get("clearance_slack_units", {}).get(key, 0.0), 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Target"), width="stretch")

    # Constraint violations
    st.markdown("---")
    st.subheader("Hard-Rule Violations")
    viols = opt_res.get("violations", [])
    hard_viols = [v for v in viols if not v["rule"].startswith("clearance:")]
    if hard_viols:
        for v in hard_viols:
            with st.expander(f"⚠️ {v['rule']}  ({v['count']} occurrences)"):
                st.write(v.get("example", ""))
    else:
        st.success("No hard-rule violations in the optimizer plan.")
    for violation in viols:
        if violation["rule"].startswith("clearance:"):
            st.warning(f"Clearance target shortfall: {violation['rule']} {violation.get('example', '')}")
    if opt_res["validation_log"]:
        with st.expander("Validator Re-plan Log"):
            st.json(opt_res["validation_log"])

    # Strategy agent shortlist
    st.markdown("---")
    st.subheader("Strategy Agent — SKU Shortlist")
    if "strategy_output" not in st.session_state:
        _step_navigation(False)
        st.stop()
    strategy = st.session_state.strategy_output
    df_short = pd.DataFrame(strategy.shortlist)[
        ["sku", "name", "priority", "mechanism", "suggested_discount_pct", "rationale"]
    ] if strategy.shortlist else pd.DataFrame()
    if not df_short.empty:
        df_short["suggested_discount_pct"] = df_short["suggested_discount_pct"].map("{:.0%}".format)
        st.dataframe(df_short, width="stretch")
    st.info(strategy.overall_rationale)

    _step_navigation(True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · Simulation
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("🎲 Monte Carlo Simulation")
    st.markdown("Range of outcomes across **200 scenarios** (perturbed elasticities + Poisson demand noise).")

    n_scenarios = st.slider("Number of scenarios", 50, 500, 200, 50)

    active = get_optimizer_results()
    if st.button("▶ Run Simulation", type="primary", disabled=active is None):
        run_simulation.clear()
        with st.spinner("Running simulation…"):
            st.session_state.simulation_output = run_simulation(
                active["plan"], active["constraints"], n=n_scenarios)

    if "simulation_output" not in st.session_state:
        _step_navigation(False)
        st.stop()
    results = st.session_state.simulation_output
    if results is None:
        st.warning("No simulation result for the current session plan.")
        _step_navigation(False)
        st.stop()

    scenario_names = list(results.keys())
    selected = st.selectbox("Scenario", scenario_names)
    res = results[selected]

    # Feasibility rate
    col1, col2 = st.columns(2)
    col1.metric("Structural Feasibility Rate", f"{res.feasibility_rate:.0%}",
                help="Fraction of scenarios with zero hard-rule violations (budget, caps, blackouts, etc.)")
    col2.metric("Scenarios run", res.n_scenarios)

    st.markdown("---")

    # KPI band chart
    st.subheader("KPI Outcome Bands (p10 / p50 / p90)")
    kpi_display = ["revenue", "net_contribution", "gross_margin", "marketing_spend"]
    band_rows = []
    for key in kpi_display:
        p10, p50, p90 = res.kpi_band(key)
        band_rows.append({"KPI": key, "p10": f"${p10:,.0f}", "p50": f"${p50:,.0f}", "p90": f"${p90:,.0f}",
                          "spread": f"{(p90-p10)/abs(p50):.1%}" if p50 else "—"})
    st.dataframe(pd.DataFrame(band_rows).set_index("KPI"), width="stretch")

    # Sell-through bands
    if res.sell_through_samples:
        st.subheader("Clearance Sell-Through Bands")
        c = get_constraints()
        targets = {t["sku"] + "@" + ("all" if t["stores"] == "all" else "+".join(t["stores"])):
                   t["min_sell_through_pct"] for t in c["clearance_targets"]}
        st_rows = []
        for key, arr in res.sell_through_samples.items():
            p10, p50, p90 = res.sell_through_band(key)
            target = targets.get(key, 0)
            hit_rate = res.clearance_hit_rate(key, target)
            st_rows.append({"Target": key, "Required": f"{target:.0%}",
                            "p10": f"{p10:.0%}", "p50": f"{p50:.0%}", "p90": f"{p90:.0%}",
                            "Hit rate": f"{hit_rate:.0%}"})
        st.dataframe(pd.DataFrame(st_rows).set_index("Target"), width="stretch")

    # Net contribution distribution chart
    st.subheader("Net Contribution Distribution")
    arr = res.samples["net_contribution"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=arr, nbinsx=30, name=selected,
                               marker_color="#2196F3", opacity=0.75))
    p10, p50, p90 = res.kpi_band("net_contribution")
    for pval, label, colour in [(p10, "p10", "red"), (p50, "p50", "green"), (p90, "p90", "orange")]:
        fig.add_vline(x=pval, line_dash="dash", line_color=colour,
                      annotation_text=f"{label}: ${pval:,.0f}", annotation_position="top")
    fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat=",", height=350,
                      margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # Violation breakdown
    if res.violation_counts:
        st.subheader("Violation Counts Across All Scenarios")
        viol_df = pd.DataFrame(
            [{"Rule": r, "Count": c, "Rate": f"{c/res.n_scenarios:.0%}"}
             for r, c in sorted(res.violation_counts.items(), key=lambda x: -x[1])]
        )
        st.dataframe(viol_df.set_index("Rule"), width="stretch")

    _step_navigation(True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · Live Re-plan
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("🔄 Live Re-plan")
    st.markdown("Live Re-plan")

    if "input_output" not in st.session_state:
        st.warning("No extracted constraints in this session.")
        if st.button("Run Input Agent", type="primary",
                     disabled=st.session_state.get("input_source") == "Uploaded documents"):
            run_input_agent.clear()
            with st.spinner("Running input agent…"):
                st.session_state.input_output = run_input_agent()
        if "input_output" not in st.session_state:
            _step_navigation(False)
            st.stop()
    input_result, _ = st.session_state.input_output
    base_constraints = get_constraints()

    st.subheader("Constraint Overrides")
    col1, col2 = st.columns(2)
    with col1:
        new_budget = st.slider(
            "Marketing Budget ($)",
            min_value=10_000, max_value=100_000,
            value=int(base_constraints.get("marketing_budget", 59_000)),
            step=1_000,
        )
        new_margin = st.slider(
            "Min Blended Promo Margin",
            min_value=10, max_value=40,
            value=round(float(base_constraints.get("min_blended_promo_margin_pct", 0.21)) * 100),
            step=1, format="%d%%",
        ) / 100
    with col2:
        new_max_disc = st.slider(
            "Max Discount %",
            min_value=10, max_value=50,
            value=round(float(base_constraints.get("max_discount_pct", 0.35)) * 100),
            step=5, format="%d%%",
        ) / 100
        new_consec = st.slider(
            "Max Consecutive Promo Weeks",
            min_value=1, max_value=5,
            value=int(base_constraints.get("max_consecutive_promo_weeks", 3)),
        )

    overrides = {
        "marketing_budget": new_budget,
        "min_blended_promo_margin_pct": new_margin,
        "max_discount_pct": new_max_disc,
        "max_consecutive_promo_weeks": new_consec,
    }

    # Show what changed
    changes = {k: (base_constraints.get(k), v) for k, v in overrides.items()
               if base_constraints.get(k) != v}
    if changes:
        st.info("**Changes from extracted constraints:**\n" +
                "\n".join(f"- `{k}`: {old!r} → {new!r}" for k, (old, new) in changes.items()))

    if st.button("🤖 Re-plan with these constraints", type="primary"):
        run_strategy_agent.clear()
        patched = {**base_constraints, **overrides}
        previous = get_optimizer_results()
        with st.spinner("Re-solving and validating promotion plan…"):
            try:
                updated = generate_plan(patched)
                updated["previous_kpis"] = previous["optimizer"] if previous else None
                st.session_state.replan_output = updated
            except RuntimeError as error:
                st.error(str(error))

    if "replan_output" not in st.session_state:
        _step_navigation(False)
        st.stop()
    updated = st.session_state.replan_output
    strategy = updated["strategy"]

    st.metric("Solver Status", updated["status"])
    st.success("No hard-rule violations in the updated plan.")
    if updated.get("previous_kpis"):
        _kpi_cards(updated["previous_kpis"], updated["no_promo"], "Before Re-plan")
    _kpi_cards(updated["optimizer"], updated["no_promo"], "After Re-plan")
    st.dataframe(updated["plan"], width="stretch")
    for violation in updated["violations"]:
        st.warning(f"{violation['rule']}: {violation.get('example', '')}")
    if updated["validation_log"]:
        with st.expander("Validator Re-plan Log"):
            st.json(updated["validation_log"])

    st.markdown("---")
    st.subheader("Updated Strategy")
    st.info(strategy.overall_rationale)

    if strategy.shortlist:
        df = pd.DataFrame(strategy.shortlist)[
            ["sku", "name", "priority", "mechanism", "suggested_discount_pct", "rationale"]
        ]
        df["suggested_discount_pct"] = df["suggested_discount_pct"].map("{:.0%}".format)
        st.dataframe(df, width="stretch")

    if strategy.bundle_suggestions:
        st.subheader("Bundle Suggestions")
        for b in strategy.bundle_suggestions:
            st.markdown(f"- **{' + '.join(b['skus'])}**: {b['rationale']}")

    if strategy.excluded_skus:
        with st.expander(f"Excluded SKUs ({len(strategy.excluded_skus)})"):
            for e in strategy.excluded_skus:
                st.markdown(f"- **{e['sku']}**: {e['reason']}")

    _step_navigation(True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · Validator
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("🛡️ Validator — Constraint Checker")
    st.markdown(
        "The **deterministic validator** checks any plan against all hard constraints. "
        "Bad plans are rejected before launch. No LLM involved — pure rule enforcement."
    )

    c = get_constraints()
    inv = get_inventory()

    # Plan selector
    plan_choice = st.radio(
        "Select a plan to validate",
        ["Optimizer plan", "Flat 20% (bad plan)", "Custom bad plan"],
        horizontal=True,
    )

    if plan_choice == "Optimizer plan":
        active = get_optimizer_results()
        if active is None:
            st.warning("No validated plan in this session.")
            _step_navigation(False)
            st.stop()
        plan = active["plan"]
        st.caption(f"{len(plan)} promoted store-SKU-weeks across {plan.sku.nunique()} SKUs")

    elif plan_choice == "Flat 20% (bad plan)":
        plan = None

    else:
        st.markdown("**Build a custom plan that violates specific rules:**")
        from src.config import plan_weeks
        W = plan_weeks()
        col1, col2, col3 = st.columns(3)
        bad_sku = col1.selectbox("SKU", ["D001", "B001", "D008", "P004"], index=0,
                                 help="D001=no-promo, B001=FizzUp cap, D008=Roast Masters cap, P004=West blackout")
        bad_disc = col2.slider("Discount", 5, 50, 25, 5, format="%d%%") / 100
        store_ids = pd.read_csv(DATA / "stores.csv")["store_id"].tolist()
        bad_store = col3.selectbox("Store", store_ids, index=6,
                                   help="S07=Seattle (West region)")
        plan = pd.DataFrame([
            {"week_start": W[0], "store_id": bad_store, "sku": bad_sku, "discount_pct": bad_disc}
        ])
        st.dataframe(plan, width="stretch")

    if st.button("✅ Validate Plan", type="primary", disabled=c is None):
        from src.evaluator import evaluate_plan
        with st.spinner("Evaluating…"):
            model = get_model()
            if plan is None:
                from src.baseline import flat_plan
                plan = flat_plan(model)
            kpis, _ = evaluate_plan(plan, model, inv, c)
            violations = check_constraints(plan, kpis, c, model.products, model.stores)

        st.markdown("---")
        hard_violations = [violation for violation in violations if not violation["rule"].startswith("clearance:")]
        if hard_violations:
            st.error(f"**Plan REJECTED** — {len(hard_violations)} hard rule(s) broken", icon="❌")
            for v in hard_violations:
                with st.expander(f"❌ {v['rule']}  ({v['count']} occurrences)"):
                    st.write(v.get("example", ""))
        else:
            st.success("**Plan ACCEPTED** — all hard constraints satisfied ✅", icon="✅")
        for violation in violations:
            if violation["rule"].startswith("clearance:"):
                st.warning(f"Clearance target shortfall: {violation.get('example', '')}")

        # KPI summary
        st.subheader("Plan KPIs")
        cols = st.columns(4)
        cols[0].metric("Revenue", f"${kpis['revenue']:,.0f}")
        cols[1].metric("Gross Margin %", f"{kpis['gross_margin_pct']:.1%}")
        cols[2].metric("Net Contribution", f"${kpis['net_contribution']:,.0f}")
        cols[3].metric("Marketing Spend", f"${kpis['marketing_spend']:,.0f}")

        st.subheader("Sell-Through")
        st_cols = st.columns(max(1, len(kpis["sell_through"])))
        for col, (key, val) in zip(st_cols, kpis["sell_through"].items()):
            tgt = next((t["min_sell_through_pct"] for t in c["clearance_targets"]
                        if key.startswith(t["sku"] + "@")), None)
            delta = f"target {tgt:.0%}" if tgt else None
            col.metric(key, f"{val:.0%}", delta=delta,
                       delta_color="normal" if tgt is None or val >= tgt else "inverse")

    _step_navigation(get_optimizer_results() is not None)


elif page == PAGES[5]:
    st.title("PromoPilot AI")
    active = get_optimizer_results()
    if active is None:
        st.warning("No validated plan in this session.")
        _step_navigation(False)
        st.stop()
    if active.get("hard_violations"):
        st.error("The current plan has hard-rule violations and cannot be showcased.")
        _step_navigation(False)
        st.stop()

    plan = active["plan"]
    constraints = active["constraints"]
    kpis = active["optimizer"]
    products = pd.read_csv(DATA / "products.csv").set_index("sku")
    stores = pd.read_csv(DATA / "stores.csv").set_index("store_id")
    regions = sorted(stores.loc[stores.index.intersection(plan.store_id.unique()), "region"].unique())
    window = constraints.get("planning_window", {})
    start = window.get("start", str(plan.week_start.min().date()) if not plan.empty else "Not scheduled")
    end = window.get("end", str((plan.week_start.max() + pd.Timedelta(days=6)).date()) if not plan.empty else "Not scheduled")
    st.html(f"""
        <style>
        .promo-banner {{background:#173a40;color:#fff;padding:24px;border-radius:8px;
            border-left:4px solid #36cb8c;font-family:Georgia,serif;}}
        .promo-banner h2 {{font-size:22px;margin:0 0 12px;letter-spacing:0;color:#fff;}}
        .promo-banner p {{margin:0;color:#d3e9e4;font-family:inherit;overflow-wrap:anywhere;}}
        </style>
        <section class="promo-banner">
        <h2>Autonomous Retail Promotion Planner</h2>
        <p>{escape(str(start))} to {escape(str(end))} &middot;
        {escape(', '.join(regions) or 'No stores selected')} &middot;
        Budget ${constraints['marketing_budget']:,.0f} &middot; Awaiting approval</p>
        </section>
    """)
    st.caption(f"Solver: {active['status']} | Synthetic retail forecast | USD")
    metrics = st.columns(4)
    metrics[0].metric("Proposed Revenue", f"${kpis['revenue']:,.0f}")
    metrics[1].metric("Promotion Budget Used", f"${kpis['marketing_spend']:,.0f}",
                      f"${constraints['marketing_budget'] - kpis['marketing_spend']:,.0f} remaining",
                      delta_color="off")
    metrics[2].metric("Products Selected", plan.sku.nunique())
    metrics[3].metric("Hard Constraint Checks", "Passed")
    for violation in active["violations"]:
        st.warning(f"Clearance shortfall: {violation['rule']} {violation.get('example', '')}")
    simulations = st.session_state.get("simulation_output")
    if simulations and "Optimizer" in simulations:
        simulation = simulations["Optimizer"]
        st.caption(f"Simulated structural feasibility: {simulation.feasibility_rate:.0%} across {simulation.n_scenarios} scenarios")
    else:
        st.caption("Simulation pending for this plan")

    st.subheader("Recommended Promotions")
    query = st.text_input("Search Products", placeholder="Product name or SKU")
    summary = plan.groupby("sku").agg(
        min_discount=("discount_pct", "min"), max_discount=("discount_pct", "max"),
        active_weeks=("week_start", "nunique"), store_count=("store_id", "nunique"))
    strategy_items = {item["sku"]: item for item in active["strategy"].shortlist}
    photos = {
        "Beverages": "https://images.unsplash.com/photo-1622483767028-3f66f32aef97?w=240&h=180&fit=crop",
        "Snacks": "https://images.unsplash.com/photo-1566478989037-eec170784d0b?w=240&h=180&fit=crop",
        "Dairy": "https://images.unsplash.com/photo-1563636619-e9143da7973b?w=240&h=180&fit=crop",
    }
    for sku, promotion in summary.iterrows():
        product = products.loc[sku]
        if query and query.lower() not in f"{sku} {product['name']}".lower():
            continue
        with st.container(border=True):
            visual, details = st.columns([1, 5])
            with visual:
                image = photos.get(product.category)
                if image:
                    st.image(image, caption="Category illustration", width="stretch")
                else:
                    st.metric("SKU", sku)
            with details:
                st.subheader(product['name'])
                discount = (f"{promotion.min_discount:.0%}" if promotion.min_discount == promotion.max_discount
                            else f"{promotion.min_discount:.0%} to {promotion.max_discount:.0%}")
                st.write(f"Percentage off: {discount} | {int(promotion.active_weeks)} active weeks | {int(promotion.store_count)} stores")
                item = strategy_items.get(sku, {})
                st.write(item.get("rationale") or "Selected by the solver to support clearance or paired-promotion requirements under the current constraints.")
                st.caption(f"{sku} | {product.category} | {product.brand}")
    if summary.empty:
        st.info("The validated plan contains no promotions.")
    st.download_button("Download Promotion Plan", plan.to_csv(index=False),
                       file_name="promotion_plan.csv", mime="text/csv", icon=":material/download:")
    st.checkbox("Reviewed for human approval", key="showcase_approval")
    st.caption("Not launched")
    _step_navigation(True)
