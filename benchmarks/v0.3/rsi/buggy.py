"""Small statistics helpers used by the report generator."""

from collections import Counter
from typing import Iterable, List, Sequence


def mean(xs: Sequence[float]) -> float:
    """Arithmetic mean; raises ValueError on an empty sequence."""
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def median(xs: Sequence[float]) -> float:
    """Middle value of the sorted data."""
    s = sorted(xs)
    n = len(s)
    return s[n // 2]


def mode(xs: Iterable[float]) -> float:
    """Most common value; ties go to the first one seen."""
    counts = Counter(xs)
    if not counts:
        raise ValueError("mode of empty sequence")
    return counts.most_common(1)[0][0]


def variance(xs: Sequence[float], sample: bool = True) -> float:
    """Sample (n - 1) or population (n) variance."""
    n = len(xs)
    if n < 2 and sample:
        raise ValueError("sample variance needs at least two points")
    m = mean(xs)
    ss = sum((x - m) ** 2 for x in xs)
    return ss / (n - 1 if sample else n)


def stdev(xs: Sequence[float], sample: bool = True) -> float:
    """Square root of the variance."""
    return variance(xs, sample) ** 0.5


def percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(xs)
    if not s:
        raise ValueError("percentile of empty sequence")
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def zscores(xs: Sequence[float]) -> List[float]:
    """Standard scores of each element."""
    m, sd = mean(xs), stdev(xs)
    return [(x - m) / sd for x in xs]


def summarize(xs: Sequence[float]) -> dict:
    """Dictionary with the usual summary statistics."""
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "stdev": stdev(xs) if len(xs) > 1 else 0.0,
        "p90": percentile(xs, 90),
    }
