"""Trial-level uncertainty; time points within a trial are not independent samples."""
import numpy as np
from scipy.stats import t


def mean_interval(values, level=0.95):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or not 0 < level < 1:
        raise ValueError("Need at least two finite observations and 0 < level < 1")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    radius = float(t.ppf((1 + level) / 2, len(x) - 1) * se)
    return {"mean": mean, "standard_error": se, "lower": mean - radius,
            "upper": mean + radius, "n": len(x)}

