# Roadmap — small steps, every day

One focused item per day. Each item ships with tests and a commit.
Progress is logged in [PROGRESS.md](PROGRESS.md).

## v0.2 — deeper faults, richer reports

- [ ] **LatencyFault** — inject artificial delays into tools; record per-step durations in the trace
- [ ] **Timeout handling** — max wall-clock time per run; abort cleanly and record the timeout in the trace
- [ ] **JSONL trace export** — runner writes `traces.jsonl` (one trace per line) alongside the markdown report
- [ ] **Regression mode** — `python -m agent_eval.compare old.jsonl new.jsonl`: diff findings across runs
- [ ] **Silent-corruption scenario** — `CorruptFault` demo: calculator returns a wrong result, agent never notices
- [ ] **Deception evaluator** — flag when the final answer contradicts what the trace actually shows
- [ ] **Budget evaluator** — max tool calls / max steps as a first-class finding
- [ ] **Severity levels** — findings get `info` / `warn` / `critical`; the report groups by severity
- [ ] **Multi-agent scenario** — planner + executor sub-agents with conflicting objectives; who wins?
- [ ] **Custom-scenario tutorial** — `examples/custom_scenario.py`: a commented walkthrough for new users
- [ ] **AnthropicBackend transport test** — fake HTTP transport so the real-LLM path is tested with no API key
- [ ] **README badges + usage docs** — how to write custom faults, policies, and evaluators
- [ ] **Prompt-leak detector** — flag when the agent reveals system-prompt content in its output
- [ ] **v0.2.0 release** — version bump, release notes, changelog entry

## v0.3 — ideas parking lot

- Web UI for browsing traces
- More built-in scenarios (multi-step research tasks)
- Benchmark harness: run the suite against several models, one table
