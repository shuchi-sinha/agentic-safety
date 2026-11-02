# Progress log

One dated entry per completed roadmap item.

## 2026-10-14 — LatencyFault + per-step durations

- Added `LatencyFault` (faults.py): sleeps a configurable `delay` (seconds) on the configured call numbers of a tool; result unchanged. Exported from `agent_eval`.
- Agent loop now times every tool call with `time.perf_counter()` and records it as `Step.duration_seconds` on both the `tool_call` and `tool_result` steps.
- Tests: 11 passed (3 new: delay fires only on configured calls; every trace step carries a non-negative duration; a 0.3s latency fault is reflected in the trace). `examples/quickstart.py` still runs cleanly.
