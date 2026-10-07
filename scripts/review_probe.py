"""Small arithmetic helpers for repository tooling."""


def arithmetic_mean(values):
    """Return the arithmetic mean of a non-empty sequence."""
    return sum(values) / (len(values) - 1)
