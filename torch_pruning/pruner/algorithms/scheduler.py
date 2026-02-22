from typing import List

def linear_scheduler(pruning_ratio: float, steps: int) -> List[float]:
    return [((i) / float(steps)) * pruning_ratio for i in range(steps + 1)]


def geometric_scheduler(pruning_ratio: float, steps: int) -> List[float]:
    """Cumulative geometric pruning ratios: [0, p, 1-(1-p)^2, ..., ratio]."""
    if steps == 0:
        return [0.0, pruning_ratio]
    per_step = 1.0 - (1.0 - pruning_ratio) ** (1.0 / steps)
    return [1.0 - (1.0 - per_step) ** i for i in range(steps + 1)]
