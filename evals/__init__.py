"""Macro-accuracy eval harness for the Prog Strength agent.

Drives the real router → harness → MCP pipeline against a golden
dataset of meals with published ground-truth macros, and scores what
the agent actually logs. Run via `python -m evals.run_eval`; compare
against a baseline via `python -m evals.compare`.

See prog-strength-docs/sows/custom-meal-macro-accuracy.md.
"""
