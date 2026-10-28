"""Experiment-portfolio simulator and decision rules (ECON 5200 PS2)."""
from __future__ import annotations


def simulate_portfolio(n: int, seed: int, **kwargs):
    """Return the experiment log with the true effect attached."""
    ...


def apply_rule(log, rule):
    """Ship decisions under `rule`: the p < 0.05 threshold and at least one other."""
    ...


def evaluate(log, rule):
    """Realised true return, ship count and false-ship rate under `rule`."""
    ...


if __name__ == "__main__":
    ...   # demonstration plus at least three assertions that must hold
