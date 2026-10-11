import json
from io import BytesIO
from zipfile import ZIP_STORED, ZipFile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from openai import APIConnectionError
from streamlit.testing.v1 import AppTest

from src.simulation import SimResult


@pytest.fixture
def ui(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    root = Path(__file__).resolve().parents[1]
    constraints = json.loads((root / "data/gold/constraints.json").read_text())
    strategy = SimpleNamespace(
        shortlist=[], overall_rationale="Test strategy",
        bundle_suggestions=[], excluded_skus=[],
    )
    simulation = SimResult(
        plan_label="Optimizer", n_scenarios=200,
        samples={key: np.array([100.0, 200.0, 300.0]) for key in
                 ("revenue", "net_contribution", "gross_margin", "marketing_spend")},
        sell_through_samples={}, feasibility_rate=1.0,
    )
    kpis = json.loads((root / "data/optimizer_results.json").read_text())["optimizer"]
    plan = pd.read_csv(root / "data/optimizer_plan.csv", parse_dates=["week_start"])

    def planning(active, strategy_result, **kwargs):
        return {
            "plan": plan.copy(), "constraints": dict(active), "strategy": strategy_result,
            "optimizer": dict(kpis, marketing_spend=active["marketing_budget"] * 0.8),
            "flat_20pct": kpis, "no_promo": kpis,
            "status": "Optimal", "solution_status": "Optimal Solution Found",
            "hard_violations": [], "violations": [], "validation_log": [],
            "clearance_slack_units": {"D002@S03+S05": 31.51},
        }

    operations = {
        "input": Mock(return_value=SimpleNamespace(
            constraints=constraints, confidence={}, warnings=[])),
        "strategy": Mock(return_value=strategy),
        "relationship": Mock(return_value=[]),
        "fit": Mock(return_value=None),
        "model": Mock(return_value=SimpleNamespace(
            products=None, stores=None, store_ids=[], skus=[])),
        "simulation": Mock(return_value={"Optimizer": simulation}),
        "evaluation": Mock(return_value=(kpis, None)),
        "planning": Mock(side_effect=planning),
    }
    for path, key in (
        ("src.agents.input_agent.run", "input"),
        ("src.agents.strategy_agent.run", "strategy"),
        ("src.agents.relationship_agent.run", "relationship"),
        ("src.elasticity.fit", "fit"),
        ("src.demand.DemandModel", "model"),
        ("src.simulation.simulate_scenarios", "simulation"),
        ("src.evaluator.evaluate_plan", "evaluation"),
        ("src.optimizer.run_planning", "planning"),
    ):
        monkeypatch.setattr(path, operations[key])
    monkeypatch.setattr("src.agents.input_agent.score", lambda result, gold: {
        "hits": ["budget"], "misses": [], "accuracy": 1.0,
    })
    monkeypatch.setattr("src.evaluator.check_constraints", Mock(return_value=[]))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()
    assert not app.exception
    yield app, operations
    st.cache_data.clear()
    st.cache_resource.clear()


def button(app, label):
    return next(control for control in app.button if label in control.label)


def test_navigation_and_controls_do_not_run_operations(ui):
    app, operations = ui
    assert button(app, "Next").disabled
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[4]).run()
    next(control for control in app.radio if control.label == "Select a plan to validate").set_value(
        "Flat 20% (bad plan)").run()
    assert not app.exception
    next(control for control in app.radio if control.label == "Select a plan to validate").set_value(
        "Custom bad plan").run()
    assert not app.exception
    app.slider[0].set_value(30).run()
    assert not app.exception
    assert all(operation.call_count == 0 for operation in operations.values())


def test_explicit_runs_next_navigation_and_retained_results(ui):
    app, operations = ui
    button(app, "Run Input Agent").click().run()
    assert not app.exception
    assert operations["input"].call_count == 1
    assert not button(app, "Next").disabled
    button(app, "Next").click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == app.sidebar.radio[0].options[1]
    assert operations["strategy"].call_count == 0
    assert button(app, "Next").disabled
    button(app, "Run Strategy Agent").click().run()
    assert not app.exception
    assert operations["strategy"].call_count == 1
    assert operations["planning"].call_count == 1
    assert operations["planning"].call_args.args[0] == app.session_state.input_output[0].constraints
    assert operations["planning"].call_args.kwargs["time_limit"] == 15
    button(app, "Next").click().run()
    assert not app.exception
    assert operations["simulation"].call_count == 0
    app.slider[0].set_value(250).run()
    assert operations["simulation"].call_count == 0
    button(app, "Run Simulation").click().run()
    assert not app.exception
    assert operations["simulation"].call_count == 1
    assert operations["simulation"].call_args.kwargs["n"] == 250
    assert operations["simulation"].call_args.kwargs["constraints"] == app.session_state.planning_output["constraints"]
    pd.testing.assert_frame_equal(operations["simulation"].call_args.args[0]["Optimizer"],
                                  app.session_state.planning_output["plan"])
    app.slider[0].set_value(300).run()
    assert not app.exception
    assert operations["simulation"].call_count == 1
    button(app, "Next").click().run()
    assert not app.exception
    app.slider[0].set_value(60000).run()
    app.slider[1].set_value(25).run()
    app.slider[2].set_value(40).run()
    assert app.slider[1].proto.format == "%d%%"
    assert app.slider[2].proto.format == "%d%%"
    assert operations["strategy"].call_count == 1
    button(app, "Re-plan with these constraints").click().run()
    assert not app.exception
    assert operations["strategy"].call_count == 2
    assert operations["strategy"].call_args.kwargs["constraints"]["marketing_budget"] == 60000
    assert operations["strategy"].call_args.kwargs["constraints"]["min_blended_promo_margin_pct"] == 0.25
    assert operations["strategy"].call_args.kwargs["constraints"]["max_discount_pct"] == 0.40
    assert operations["planning"].call_count == 2
    assert app.session_state.planning_output["constraints"]["marketing_budget"] == 60000
    assert app.session_state.planning_output["optimizer"]["marketing_spend"] == 48000
    assert "simulation_output" not in app.session_state
    button(app, "Next").click().run()
    assert not app.exception
    assert operations["evaluation"].call_count == 0
    button(app, "Validate Plan").click().run()
    assert not app.exception
    assert operations["evaluation"].call_count == 1
    assert operations["evaluation"].call_args.args[3]["marketing_budget"] == 60000
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert operations["input"].call_count == 1
    assert operations["strategy"].call_count == 2
    assert operations["simulation"].call_count == 1
    assert operations["evaluation"].call_count == 1
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[0]).run()
    button(app, "Run Input Agent").click().run()
    assert not app.exception
    assert operations["input"].call_count == 2
    assert "strategy_output" not in app.session_state
    assert "replan_output" not in app.session_state
    assert "planning_output" not in app.session_state


def test_live_replan_can_extract_missing_constraints_on_click(ui):
    app, operations = ui
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    assert app.warning[0].value == "No extracted constraints in this session."
    assert button(app, "Next").disabled
    assert operations["input"].call_count == 0
    button(app, "Run Input Agent").click().run()
    assert not app.exception
    assert operations["input"].call_count == 1
    assert len(app.slider) == 4
    assert operations["strategy"].call_count == 0
    assert button(app, "Next").disabled
    button(app, "Re-plan with these constraints").click().run()
    assert not app.exception
    assert operations["strategy"].call_count == 1
    assert not button(app, "Next").disabled


def test_plan_page_displays_solver_status_and_clearance_slack(ui):
    app, _ = ui
    button(app, "Run Input Agent").click().run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[1]).run()
    button(app, "Run Strategy Agent").click().run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == "Solver Status").value == "Optimal"
    table = next(table.value for table in app.dataframe if "Planning Slack (units)" in table.value.columns)
    assert "Shortfall (pp)" in table.columns
    assert table.loc["D002@S03+S05", "Planning Slack (units)"] > 0


def test_plan_page_blocks_failed_solve(ui):
    app, operations = ui
    operations["planning"].side_effect = RuntimeError("No feasible integer solution")
    button(app, "Run Input Agent").click().run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[1]).run()
    button(app, "Run Strategy Agent").click().run()
    assert not app.exception
    assert app.error[0].value == "No feasible integer solution"
    assert "planning_output" not in app.session_state
    assert button(app, "Next").disabled


def test_failed_replan_keeps_previous_valid_plan(ui):
    app, operations = ui
    button(app, "Run Input Agent").click().run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[1]).run()
    button(app, "Run Strategy Agent").click().run()
    previous = app.session_state.planning_output
    operations["planning"].side_effect = RuntimeError("Plan rejected after validation")
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    app.slider[0].set_value(40000).run()
    button(app, "Re-plan with these constraints").click().run()
    assert not app.exception
    assert app.error[0].value == "Plan rejected after validation"
    assert app.session_state.planning_output["constraints"] == previous["constraints"]
    pd.testing.assert_frame_equal(app.session_state.planning_output["plan"], previous["plan"])
    assert "replan_output" not in app.session_state


def test_upload_source_requires_files_and_clears_previous_results(ui):
    app, operations = ui
    button(app, "Run Input Agent").click().run()
    assert "input_output" in app.session_state
    app.radio[0].set_value("Uploaded documents").run()
    assert not app.exception
    assert button(app, "Run Input Agent").disabled
    assert button(app, "Next").disabled
    assert "input_output" not in app.session_state
    assert operations["input"].call_count == 1


def test_uploaded_text_reaches_input_agent(monkeypatch):
    import src.agents.input_agent as agent

    call = Mock(return_value={"constraints": {"marketing_budget": 40000}, "confidence": {}, "warnings": []})
    monkeypatch.setattr(agent, "call_json", call)
    result = agent.run(documents={"custom_budget.txt": "Final marketing budget: $40,000."})
    assert result.constraints["marketing_budget"] == 40000
    assert "custom_budget.txt" in call.call_args.args[1]
    assert "Final marketing budget: $40,000." in call.call_args.args[1]
    with pytest.raises(ValueError, match="non-empty"):
        agent.run(documents={"empty.txt": " "})


def test_uploaded_documents_are_passed_without_demo_accuracy(ui, monkeypatch):
    app, operations = ui
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [SimpleNamespace(
        name="budget.txt", getvalue=lambda: b"Final budget: $40,000.")])
    app.radio[0].set_value("Uploaded documents").run()
    button(app, "Run Input Agent").click().run()
    assert not app.exception
    assert operations["input"].call_args.kwargs["documents"] == {"budget.txt": "Final budget: $40,000."}
    assert all(metric.label != "Extraction Accuracy" for metric in app.metric)
    assert app.session_state.input_output[1] is None
    button(app, "Next").click().run()
    button(app, "Run Strategy Agent").click().run()
    assert not app.exception
    assert operations["planning"].call_args.args[0] == app.session_state.input_output[0].constraints


def test_replan_does_not_extract_samples_in_upload_mode(ui):
    app, operations = ui
    app.radio[0].set_value("Uploaded documents").run()
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[3]).run()
    assert not app.exception
    assert button(app, "Run Input Agent").disabled
    assert operations["input"].call_count == 0


def test_azure_failure_uses_enabled_ollama_fallback(monkeypatch):
    import httpx
    import src.agents._llm as llm

    monkeypatch.setattr(llm, "load_dotenv", Mock())
    monkeypatch.setenv("OLLAMA_FALLBACK_ENABLED", "true")
    client = Mock()
    client.chat.completions.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    fallback = Mock(return_value='{"constraints": {}}')
    monkeypatch.setattr(llm, "_ollama_call", fallback)
    assert llm.call_json("Return JSON", "Budget memo") == {"constraints": {}}
    fallback.assert_called_once_with("Return JSON", "Budget memo", 0.0, True)


def test_disabled_ollama_fallback_preserves_azure_error(monkeypatch):
    import httpx
    import src.agents._llm as llm

    monkeypatch.setattr(llm, "load_dotenv", Mock())
    monkeypatch.setenv("OLLAMA_FALLBACK_ENABLED", "false")
    client = Mock()
    client.chat.completions.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    fallback = Mock()
    monkeypatch.setattr(llm, "_ollama_call", fallback)
    with pytest.raises(APIConnectionError):
        llm.call("System", "Memo")
    fallback.assert_not_called()


def test_ollama_request_uses_named_model_and_json(monkeypatch):
    import src.agents._llm as llm

    monkeypatch.setenv("OLLAMA_MODEL", "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    response = BytesIO(b'{"message": {"content": "{}"}, "done_reason": "stop"}')
    transport = Mock(return_value=response)
    monkeypatch.setattr(llm, "urlopen", transport)
    assert llm._ollama_call("Return JSON", "Memo", 0.0, True) == "{}"
    payload = json.loads(transport.call_args.args[0].data)
    assert payload["model"] == "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M"
    assert payload["format"] == "json"
    assert payload["stream"] is False


def test_ollama_unavailable_fails_clearly(monkeypatch):
    from urllib.error import URLError
    import src.agents._llm as llm

    monkeypatch.setattr(llm, "urlopen", Mock(side_effect=URLError("Connection refused")))
    with pytest.raises(RuntimeError, match="Ollama fallback could not complete"):
        llm._ollama_call("Return JSON", "Memo", 0.0, True)


def test_ollama_context_overflow_is_not_sent(monkeypatch):
    import src.agents._llm as llm

    monkeypatch.setenv("OLLAMA_CONTEXT_TOKENS", "8192")
    monkeypatch.setattr(llm, "validate_context", Mock(side_effect=ValueError("Input too large")))
    transport = Mock()
    monkeypatch.setattr(llm, "urlopen", transport)
    with pytest.raises(ValueError, match="Input too large"):
        llm._ollama_call("System", "Large memo", 0.0, True)
    llm.validate_context.assert_called_once_with("System", "Large memo", max_tokens=6144)
    transport.assert_not_called()


def test_excel_upload_reads_all_sheets(ui, monkeypatch):
    app, operations = ui
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([["Marketing budget", 40000]]).to_excel(writer, sheet_name="Budget", header=False, index=False)
        pd.DataFrame([["Max discount", "25%"]]).to_excel(writer, sheet_name="Policy", header=False, index=False)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [SimpleNamespace(
        name="constraints.xlsx", getvalue=workbook.getvalue)])
    app.radio[0].set_value("Uploaded documents").run()
    assert not app.exception
    assert not button(app, "Run Input Agent").disabled
    button(app, "Run Input Agent").click().run()
    text = operations["input"].call_args.kwargs["documents"]["constraints.xlsx"]
    assert "SHEET: Budget" in text and "SHEET: Policy" in text
    assert "40000" in text and "25%" in text


def test_excel_above_previous_upload_limit_is_accepted(ui, monkeypatch):
    app, operations = ui
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([["Marketing budget", 40000]]).to_excel(writer, index=False, header=False)
    with ZipFile(workbook, "a", compression=ZIP_STORED) as archive:
        archive.writestr("unused_padding.bin", b"0" * 600_000)
    assert len(workbook.getvalue()) > 500_000
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [SimpleNamespace(
        name="larger.xlsx", getvalue=workbook.getvalue)])
    app.radio[0].set_value("Uploaded documents").run()
    assert not app.exception
    assert not app.error
    button(app, "Run Input Agent").click().run()
    assert "40000" in operations["input"].call_args.kwargs["documents"]["larger.xlsx"]


def test_file_above_new_limit_is_rejected(ui, monkeypatch):
    app, operations = ui
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [SimpleNamespace(
        name="too_large.txt", getvalue=lambda: b"0" * 1_000_001)])
    app.radio[0].set_value("Uploaded documents").run()
    assert not app.exception
    assert "1 MB each and 1 MB combined" in app.error[0].value
    assert button(app, "Run Input Agent").disabled
    assert operations["input"].call_count == 0


@pytest.mark.parametrize("empty", [False, True])
def test_invalid_or_empty_excel_blocks_extraction(ui, monkeypatch, empty):
    app, operations = ui
    workbook = BytesIO()
    if empty:
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, index=False)
        content = workbook.getvalue()
    else:
        content = b"not an Excel workbook"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [SimpleNamespace(
        name="invalid.xlsx", getvalue=lambda: content)])
    app.radio[0].set_value("Uploaded documents").run()
    assert not app.exception
    assert app.error
    assert button(app, "Run Input Agent").disabled
    assert operations["input"].call_count == 0


def test_token_dense_upload_is_batched_before_model_call(monkeypatch):
    import src.agents.input_agent as agent

    def extract(system, user):
        agent.validate_context(system, user, max_tokens=6000)
        if system == agent._BATCH_SYSTEM:
            return {"facts": ["Marketing budget is $40,000."]}
        return {"constraints": {"marketing_budget": 40000}, "confidence": {}, "warnings": []}

    call = Mock(side_effect=extract)
    monkeypatch.setattr(agent, "call_json", call)
    text = "1,2,3,4,5,6,7,8,9,0\n" * 12000
    result = agent.run(documents={"dense.txt": text})
    assert result.constraints["marketing_budget"] == 40000
    chunks = [args.args[1].split("\n", 1)[1] for args in call.call_args_list
              if args.args[0] == agent._BATCH_SYSTEM]
    assert "".join(chunks) == text
    assert len(chunks) > 1
    assert "document batches" in result.warnings[0]


def test_batch_failure_does_not_publish_partial_constraints(monkeypatch):
    import src.agents.input_agent as agent

    monkeypatch.setattr(agent, "call_json", Mock(return_value={"facts": "not a list"}))
    with pytest.raises(ValueError, match="invalid evidence"):
        agent.run(documents={"dense.txt": "1,2,3,4,5,6,7,8,9,0\n" * 400})


def test_oversized_batch_evidence_is_minimized_before_consolidation(monkeypatch):
    import src.agents.input_agent as agent

    def extract(system, user):
        agent.validate_context(system, user, max_tokens=6000)
        if system == agent._BATCH_SYSTEM:
            return {"facts": ["Budget is $40,000.\n" * 1500]}
        if system == agent._REDUCE_SYSTEM:
            return {"facts": [{"document": "dense.txt", "fact": "Budget is $40,000."}]}
        assert "Budget is $40,000." in user
        return {"constraints": {"marketing_budget": 40000}, "confidence": {}, "warnings": []}

    call = Mock(side_effect=extract)
    monkeypatch.setattr(agent, "call_json", call)
    result = agent.run(documents={"dense.txt": "1,2,3,4,5,6,7,8,9,0\n" * 400})
    assert result.constraints["marketing_budget"] == 40000
    assert any("compaction rounds" in warning for warning in result.warnings)
    assert any(args.args[0] == agent._REDUCE_SYSTEM for args in call.call_args_list)


def test_compaction_rejects_invented_sources(monkeypatch):
    import src.agents.input_agent as agent

    monkeypatch.setattr(agent, "call_json", Mock(return_value={
        "facts": [{"document": "invented.txt", "fact": "Budget: $1."}]
    }))
    with pytest.raises(ValueError, match="invalid facts or source"):
        agent._compact_evidence([{"document": "budget.txt", "fact": "Budget: $40,000.\n" * 1500}])


def test_compaction_preserves_conflicting_numeric_values(monkeypatch):
    import src.agents.input_agent as agent

    monkeypatch.setattr(agent, "call_json", Mock(return_value={
        "facts": [{"document": "budget.txt", "fact": "Budget: $40,000; alternate budget: $59,000. Cap: 25%."}]
    }))
    evidence, rounds = agent._compact_evidence([{
        "document": "budget.txt", "fact": "Budget: $40,000; alternate budget: $59,000. Cap: 25%.\n" * 1500
    }])
    assert rounds == 1
    assert "$40,000" in evidence[0]["fact"] and "$59,000" in evidence[0]["fact"]


def test_compaction_cannot_silently_drop_numeric_values(monkeypatch):
    import src.agents.input_agent as agent

    monkeypatch.setattr(agent, "call_json", Mock(return_value={
        "facts": [{"document": "budget.txt", "fact": "Budget: $40,000."}]
    }))
    with pytest.raises(ValueError, match="changed numeric values"):
        agent._compact_evidence([{
            "document": "budget.txt", "fact": "Budget: $40,000; alternate budget: $59,000.\n" * 1500
        }])


def test_compaction_accepts_equivalent_number_formatting(monkeypatch):
    import src.agents.input_agent as agent

    monkeypatch.setattr(agent, "call_json", Mock(return_value={
        "facts": [{"document": "budget.txt", "fact": "Budget: $40000.00. Cap: 25.0%. Date: 2026-11-09."}]
    }))
    evidence, rounds = agent._compact_evidence([{
        "document": "budget.txt", "fact": "Budget: $40,000. Cap: 25%. Date: 2026-11-09.\n" * 1500
    }])
    assert rounds == 1
    assert "40000.00" in evidence[0]["fact"]


def test_numeric_normalization_keeps_percentages_and_dates_distinct():
    from src.agents.input_agent import _numeric_values

    assert _numeric_values("40,000 40000.00 040000") == {"40000"}
    assert _numeric_values("25% 25.0% 0.25 2026-11-09") == {"25%", "0.25", "2026-11-09"}
    assert _numeric_values("1,25") == {"1,25"}


def test_compaction_stops_when_model_does_not_reduce_evidence(monkeypatch):
    import src.agents.input_agent as agent

    evidence = [{"document": "budget.txt", "fact": "Budget: $40,000. " + (chr(65 + index) + " unchanged ") * 500}
                for index in range(10)]

    def unchanged(system, user):
        records = json.loads(user)
        return {"facts": [{"document": record["document"], "fact": record["fact"]} for record in records]}

    monkeypatch.setattr(agent, "call_json", unchanged)
    with pytest.raises(ValueError, match="could not be shortened safely"):
        agent._compact_evidence(evidence)


def test_batched_evidence_keeps_conflicting_values_and_sources(monkeypatch):
    import src.agents.input_agent as agent

    def extract(system, user):
        if system == agent._BATCH_SYSTEM:
            return {"facts": ["Budget is $40,000.", "Budget is $59,000; no supersession date."]}
        assert "Budget is $40,000." in user and "Budget is $59,000" in user
        assert '"document": "budget.txt"' in user
        assert "Flag unresolved conflicts" in user
        return {"constraints": {}, "confidence": {"marketing_budget": "low"},
                "warnings": ["Conflicting budget values require review."]}

    monkeypatch.setattr(agent, "call_json", extract)
    result = agent.run(documents={"budget.txt": "1,2,3,4,5,6,7,8,9,0\n" * 400})
    assert result.confidence["marketing_budget"] == "low"
    assert any("Conflicting budget" in warning for warning in result.warnings)


def test_llm_context_guard_runs_before_client_creation(monkeypatch):
    import src.agents._llm as llm

    client = Mock()
    monkeypatch.setattr(llm, "_get_client", client)
    with pytest.raises(ValueError, match="No document text was sent"):
        llm.call("Extract constraints", "1,2,3,4,5,6,7,8,9,0\n" * 12000)
    assert client.call_count == 0


def test_context_error_is_displayed_without_publishing_extraction(ui):
    app, operations = ui
    operations["input"].side_effect = ValueError("Documents exceed the input token budget.")
    button(app, "Run Input Agent").click().run()
    assert not app.exception
    assert app.error[0].value == "Documents exceed the input token budget."
    assert "input_output" not in app.session_state
    assert button(app, "Next").disabled


def test_azure_context_rejection_is_readable_and_never_falls_back(monkeypatch):
    import httpx
    from openai import BadRequestError
    import src.agents._llm as llm

    monkeypatch.setattr(llm, "load_dotenv", Mock())
    monkeypatch.setenv("OLLAMA_FALLBACK_ENABLED", "true")
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
    error = BadRequestError("Context exceeded", response=response, body={
        "error": {"code": "context_length_exceeded", "message": "Maximum context length is 128000 tokens."}
    })
    client = Mock()
    client.chat.completions.create.side_effect = error
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    fallback = Mock()
    monkeypatch.setattr(llm, "_ollama_call", fallback)
    with pytest.raises(ValueError, match="exceeds its context window"):
        llm.call("Extract constraints", "Small prompt")
    fallback.assert_not_called()


def test_showcase_uses_current_validated_plan_without_running_agents(ui):
    app, operations = ui
    button(app, "Run Input Agent").click().run()
    button(app, "Next").click().run()
    button(app, "Run Strategy Agent").click().run()
    calls = {key: operation.call_count for key, operation in operations.items()}
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[5]).run()
    assert not app.exception
    assert any(title.value == "PromoPilot AI" for title in app.title)
    assert next(metric for metric in app.metric if metric.label == "Products Selected").value == str(
        app.session_state.planning_output["plan"].sku.nunique())
    assert next(metric for metric in app.metric if metric.label == "Hard Constraint Checks").value == "Passed"
    assert {key: operation.call_count for key, operation in operations.items()} == calls
    app.text_input[0].set_value("nonexistent product").run()
    assert not app.exception


def test_showcase_requires_validated_session_plan(ui):
    app, operations = ui
    app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[5]).run()
    assert not app.exception
    assert app.warning[0].value == "No validated plan in this session."
    assert all(operation.call_count == 0 for operation in operations.values())