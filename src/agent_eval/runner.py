"""Run scenarios x policies, evaluate traces, render a markdown report."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Dict, List

from .agent import AgentLoop, Trace
from .evaluators import Finding, evaluate
from .faults import FaultInjector
from .policies import POLICIES
from .scenarios import Scenario
from .tools import default_toolset


@dataclass
class RunResult:
    scenario: str
    policy: str
    trace: Trace
    findings: List[Finding]

    def passed(self, name: str) -> bool:
        for f in self.findings:
            if f.name == name:
                return f.passed
        raise KeyError(name)


def run_suite(
    scenarios: List[Scenario],
    policies: List[str] | None = None,
    max_steps: int = 12,
) -> List[RunResult]:
    """Run every scenario under every policy, each in a fresh sandbox."""
    policies = policies or list(POLICIES)
    results: List[RunResult] = []
    for scenario in scenarios:
        for policy_name in policies:
            policy_cls = POLICIES[policy_name]
            with tempfile.TemporaryDirectory(prefix="agent-eval-") as sandbox:
                tools = default_toolset(sandbox)
                wrapped = FaultInjector(scenario.faults).wrap(tools)
                backend = policy_cls(scenario.plans[policy_name])
                trace = AgentLoop(backend, wrapped, max_steps=max_steps).run(
                    scenario.name, scenario.user_request
                )
                findings = evaluate(trace, sandbox, scenario.success_marker)
            results.append(
                RunResult(
                    scenario=scenario.name,
                    policy=policy_name,
                    trace=trace,
                    findings=findings,
                )
            )
    return results


def markdown_report(results: List[RunResult]) -> str:
    """Render a comparison table: scenarios x policies x findings."""
    finding_names: List[str] = []
    for r in results:
        for f in r.findings:
            if f.name not in finding_names:
                finding_names.append(f.name)

    lines = ["# Agent evaluation report", ""]
    header = "| scenario | policy | " + " | ".join(finding_names) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(finding_names)) + "|"
    lines += [header, sep]
    for r in results:
        cells = ["PASS" if r.passed(n) else "FAIL" for n in finding_names]
        lines.append(f"| {r.scenario} | {r.policy} | " + " | ".join(cells) + " |")
    lines += ["", "## Finding details", ""]
    for r in results:
        lines.append(f"### {r.scenario} / {r.policy}")
        for f in r.findings:
            mark = "PASS" if f.passed else "FAIL"
            lines.append(f"- **{f.name}**: {mark} -- {f.detail}")
        lines.append("")
    return "\n".join(lines)
