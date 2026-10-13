"""agent-eval-harness: fault-injection evaluation for LLM agent workflows."""

from .agent import AgentLoop, Backend, Step, Tool, Trace
from .backends import AnthropicBackend
from .evaluators import Finding, evaluate
from .faults import (
    CorruptFault,
    ErrorFault,
    Fault,
    FaultInjector,
    InjectionFault,
)
from .policies import CautiousPolicy, NaivePolicy
from .runner import RunResult, markdown_report, run_suite
from .scenarios import Scenario, builtin_scenarios
from .tools import Sandbox, default_toolset, escapes_sandbox, safe_calculate

__all__ = [
    "AgentLoop",
    "AnthropicBackend",
    "Backend",
    "CautiousPolicy",
    "CorruptFault",
    "ErrorFault",
    "Fault",
    "FaultInjector",
    "Finding",
    "InjectionFault",
    "NaivePolicy",
    "RunResult",
    "Sandbox",
    "Scenario",
    "Step",
    "Tool",
    "Trace",
    "builtin_scenarios",
    "default_toolset",
    "escapes_sandbox",
    "evaluate",
    "markdown_report",
    "run_suite",
    "safe_calculate",
]

__version__ = "0.1.0"
