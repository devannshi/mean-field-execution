"""Supplement: compare policies under a shared exact completion constraint."""
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from meanfield_execution.config import MarketConfig, default_types
from meanfield_execution.simulation import STRATEGIES, simulate
from meanfield_execution.statistics import mean_interval


def main():
    started = time.perf_counter()
    out = Path("results/hard_liquidation")
    out.mkdir(parents=True, exist_ok=True)
    config = MarketConfig()
    types = tuple(replace(t, terminal_penalty=0, hard_liquidation=True) for t in default_types())
    rows = []
    for trial in range(512):
        for strategy in STRATEGIES:
            result = simulate(config, types, strategy, seed=510000 + trial)
            rows.append({"trial": trial, "seed": 510000 + trial,
                         "strategy": strategy, **result.metrics})
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "trials.csv", index=False)
    benchmarks = {name: trials[trials.strategy == name].set_index("trial")
                  for name in ("twap", "ac")}
    summary = []
    for name, group in trials.groupby("strategy", sort=False):
        group = group.set_index("trial")
        for metric in ["objective", "shortfall", "terminal_inventory_rms", "gross_turnover"]:
            summary.append({"strategy": name, "metric": metric, **mean_interval(group[metric])})
        for reference, benchmark in benchmarks.items():
            delta = group.objective - benchmark.loc[group.index].objective
            summary.append({"strategy": name, "metric": f"objective_difference_vs_{reference}",
                            **mean_interval(delta)})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    values = summary[summary.metric == "objective_difference_vs_twap"]
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    for i, (_, row) in enumerate(values.iterrows()):
        ax.errorbar(row["mean"], i, xerr=[[row["mean"] - row.lower], [row.upper - row["mean"]]],
                    fmt="o", color="#008779", capsize=4)
    ax.axvline(0, color="#555", ls="--", lw=1)
    ax.set_yticks(range(len(values)), values.strategy)
    ax.invert_yaxis()
    ax.set_xlabel("Paired penalized-objective difference versus TWAP (lower is better)")
    ax.set_title("Equal completion: zero terminal inventory for every policy\n"
                 "512 common-noise paths; 95% Monte Carlo intervals")
    fig.savefig(out / "equal_completion.png", dpi=180)
    plt.close(fig)
    versus_twap = values[values.strategy == "filtered_mfg"].iloc[0]
    versus_ac = summary[(summary.strategy == "filtered_mfg")
                        & (summary.metric == "objective_difference_vs_ac")].iloc[0]
    lines = ["# Equal-completion supplement", "",
             "All policies must finish at zero inventory, up to floating-point precision.",
             "This is a separately labeled extension after the initial soft-terminal study,",
             "not a replacement of unfavorable results. All 512 paths and seven policies are retained.", "",
             "| Strategy | Paired objective difference vs TWAP | 95% interval |", "|---|---:|---|"]
    for _, row in values.iterrows():
        lines.append(f"| {row.strategy} | {row['mean']:.6f} | [{row.lower:.6f}, {row.upper:.6f}] |")
    lines += ["", "## What equal completion establishes", "",
              (f"Filtered-minus-TWAP is **{versus_twap['mean']:.6f}**, with paired 95% interval "
              f"**[{versus_twap.lower:.6f}, {versus_twap.upper:.6f}]**. "
              "The interval spans zero, so this setting does not establish a filtered-policy improvement."), "",
              ("The simpler AC-style policy has the lower mean penalized objective. "
              f"Filtered-minus-AC is **{versus_ac['mean']:.6f}**, with interval "
              f"**[{versus_ac.lower:.6f}, {versus_ac.upper:.6f}]**; this difference also spans zero. "
              "The AC-style versus TWAP interval is negative in this experiment. "
              "These are pointwise Monte Carlo intervals, not simultaneous guarantees."), "",
              ("A lower penalized objective is not necessarily lower execution shortfall. "
              "Raw shortfall and turnover are retained in the trial and summary files."), "",
              "## Constraint and diagnostic semantics", "",
              ("Every type uses `hard_liquidation=True` and zero terminal penalty. "
              "An exact KKT projection enforces the equality at every planning horizon; "
              "it is not a large-penalty approximation or a forced last-period adjustment. "
              "Rates remain signed, without inventory or participation bounds. "
              "The stochastic policy remains a certainty-equivalent receding-horizon approximation."), "",
              ("`decision_brier` is zero for current-state informed, equals the causal prior "
              "score for the three filter-driven policies, and is unavailable for TWAP, "
              "AC-style, and drift-blind. `filter_prior_brier` and `posterior_brier` "
              "always describe the auxiliary filter, including when a planner ignores it."), "",
              ("See [the KKT derivation](../../docs/methodology.md) and "
              "[the original soft-terminal study](../published/report.md)."), "",
              "![Equal completion](equal_completion.png)"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    sources = sorted(Path("src").rglob("*.py")) + [Path(__file__).resolve().relative_to(Path.cwd())]
    manifest = {
        "seed_base": 510000, "trials": 512, "config": asdict(config),
        "types": [asdict(agent) for agent in types],
        "constraint": "dt*sum(rate)=initial inventory",
        "max_terminal_inventory_rms": float(trials.terminal_inventory_rms.max()),
        "max_accounting_error": float(trials.accounting_error.max()),
        "runtime_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(out.iterdir()) if path.is_file() and path.name != "manifest.json"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed 512 paired hard-completion paths in {manifest['runtime_seconds']:.1f}s; "
          f"maximum terminal residual {manifest['max_terminal_inventory_rms']:.3g}")


if __name__ == "__main__":
    main()
