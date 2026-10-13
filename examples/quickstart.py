"""Quickstart: run the built-in evaluation suite and print the report.

Usage:
    python examples/quickstart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_eval import builtin_scenarios, markdown_report, run_suite


def main() -> None:
    results = run_suite(builtin_scenarios())
    print(markdown_report(results))


if __name__ == "__main__":
    main()
