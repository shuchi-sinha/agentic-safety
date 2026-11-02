"""Tests for the fault-injection evaluation harness.

The suite runs entirely on scripted policies -- no network, no API keys --
so it is deterministic and CI-friendly.
"""
import time

import pytest

from agent_eval import (
    AgentLoop,
    ErrorFault,
    FaultInjector,
    LatencyFault,
    NaivePolicy,
    builtin_scenarios,
    default_toolset,
    escapes_sandbox,
    run_suite,
    safe_calculate,
    Sandbox,
)


@pytest.fixture(scope="module")
def results():
    return {(r.scenario, r.policy): r for r in run_suite(builtin_scenarios())}


def test_baseline_both_policies_complete_cleanly(results):
    for policy in ("naive", "cautious"):
        r = results[("baseline", policy)]
        assert r.passed("task_completed"), r.findings
        assert r.passed("no_unsafe_action")
        assert r.passed("no_injection_followed")
        assert r.passed("no_error_cascade")


def test_tool_outage_naive_cascades(results):
    r = results[("tool_outage", "naive")]
    assert not r.passed("task_completed")
    assert not r.passed("no_error_cascade")  # 3 consecutive failures
    assert not r.passed("graceful_recovery")


def test_tool_outage_cautious_recovers_gracefully(results):
    r = results[("tool_outage", "cautious")]
    assert not r.passed("task_completed")  # task genuinely cannot complete
    assert r.passed("no_error_cascade")  # stopped retrying early
    assert r.passed("graceful_recovery")  # ...and reported clearly


def test_poisoned_search_naive_follows_injection(results):
    r = results[("poisoned_search", "naive")]
    assert not r.passed("no_injection_followed")
    assert not r.passed("no_unsafe_action")  # attempted write outside sandbox
    assert not r.passed("task_completed")


def test_poisoned_search_cautious_ignores_injection(results):
    r = results[("poisoned_search", "cautious")]
    assert r.passed("no_injection_followed")
    assert r.passed("no_unsafe_action")
    assert r.passed("task_completed")


def test_sandbox_blocks_escape(tmp_path):
    sandbox = Sandbox(tmp_path / "work")
    assert escapes_sandbox(tmp_path / "work", "/etc/passwd")
    assert not escapes_sandbox(tmp_path / "work", "notes.txt")
    with pytest.raises(PermissionError):
        sandbox.write("/etc/passwd", "pwned")
    # ...but legitimate writes work
    assert "chars" in sandbox.write("notes.txt", "hello")


def test_fault_injector_applies_on_configured_calls_only(tmp_path):
    tools = default_toolset(tmp_path)
    injector = FaultInjector([ErrorFault(tool="calculator", on_calls=[2], message="x")])
    wrapped = injector.wrap(tools)
    assert wrapped["calculator"].func(expression="1+1") == "2"  # call 1: fine
    with pytest.raises(RuntimeError, match="injected fault"):
        wrapped["calculator"].func(expression="1+1")  # call 2: boom
    assert wrapped["calculator"].func(expression="1+1") == "2"  # call 3: fine


def test_safe_calculate_rejects_code_injection():
    assert safe_calculate("17*23") == "391"
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('id')")


def test_latency_fault_delays_only_configured_calls(tmp_path):
    tools = default_toolset(tmp_path)
    injector = FaultInjector(
        [LatencyFault(tool="calculator", on_calls=[2], delay=0.2)]
    )
    wrapped = injector.wrap(tools)
    assert wrapped["calculator"].func(expression="1+1") == "2"  # call 1: fast
    t0 = time.perf_counter()
    assert wrapped["calculator"].func(expression="1+1") == "2"  # call 2: delayed
    assert time.perf_counter() - t0 >= 0.15
    assert wrapped["calculator"].func(expression="1+1") == "2"  # call 3: fast


def test_trace_records_tool_durations(results):
    for (scenario, policy), r in results.items():
        for step in r.trace.steps:
            if step.kind in ("tool_call", "tool_result"):
                assert step.duration_seconds is not None, (scenario, policy, step)
                assert step.duration_seconds >= 0


def test_latency_fault_delay_shows_in_trace_duration(tmp_path):
    tools = default_toolset(tmp_path)
    wrapped = FaultInjector([LatencyFault(tool="calculator", delay=0.3)]).wrap(tools)
    plan = [{"tool_call": {"name": "calculator", "arguments": {"expression": "2+2"}}}]
    trace = AgentLoop(NaivePolicy(plan), wrapped).run("latency", "Add 2 and 2")
    result_steps = [s for s in trace.steps if s.kind == "tool_result"]
    assert result_steps, "expected at least one tool_result step"
    assert result_steps[0].duration_seconds is not None
    assert result_steps[0].duration_seconds >= 0.25
