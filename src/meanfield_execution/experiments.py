"""Prespecified, reproducible studies with retained trial-level observations."""
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .config import MarketConfig, default_types
from .equilibrium import EquilibriumSystem, fixed_point
from .population import population_audit
from .simulation import STRATEGIES, simulate
from .statistics import mean_interval


def _seeds(seed, stream, n):
    seq = np.random.SeedSequence([seed, stream])
    return [int(s.generate_state(1)[0]) for s in seq.spawn(n)]


def compare_strategies(config, trials, seed, strategies=STRATEGIES):
    rows = []
    for trial, path_seed in enumerate(_seeds(seed, 1, trials)):
        for strategy in strategies:
            result = simulate(config, strategy=strategy, seed=path_seed)
            rows.append({"trial": trial, "seed": path_seed, "strategy": strategy, **result.metrics})
    return pd.DataFrame(rows)


def summarize_trials(trials):
    rows = []
    benchmark = trials[trials.strategy == "twap"].set_index("trial")
    metrics = ["objective", "shortfall", "terminal_inventory_rms", "posterior_brier",
               "gross_turnover", "negative_rate_fraction"]
    for strategy, subset in trials.groupby("strategy", sort=False):
        subset = subset.set_index("trial").sort_index()
        for metric in metrics:
            rows.append({"strategy": strategy, "metric": metric,
                         **mean_interval(subset[metric])})
        paired = subset.objective - benchmark.loc[subset.index].objective
        rows.append({"strategy": strategy, "metric": "objective_difference_vs_twap",
                     **mean_interval(paired)})
    return pd.DataFrame(rows)


def convergence_study(config):
    types = default_types()
    q = [a.initial_inventory for a in types]
    drift = np.linspace(0.1, 0, config.steps)
    histories, summaries = [], []
    for coupling in [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]:
        system = EquilibriumSystem(types, config.steps, config.dt, coupling)
        direct, _ = system.solve(q, drift)
        for damping in [0.15, 0.35, 0.6, 1.0]:
            result = fixed_point(system, q, drift, damping=damping, max_iterations=200)
            summaries.append({"permanent_impact": coupling, "damping": damping,
                              "converged": result.converged, "iterations": result.iterations,
                              "residual": result.residuals[-1],
                              "relative_error_to_direct": float(np.linalg.norm(result.mean_rate - direct)
                                                                 / np.linalg.norm(direct)),
                              "iteration_spectral_radius": system.spectral_radius(damping),
                              "equilibrium_condition_number": float(np.linalg.cond(system.equilibrium_matrix))})
            histories.extend({"permanent_impact": coupling, "damping": damping,
                              "iteration": i, "residual": r}
                             for i, r in enumerate(result.residuals))
    return pd.DataFrame(histories), pd.DataFrame(summaries)


def finite_population_study(config, repeats, seed):
    rows = []
    for N in [10, 25, 50, 100, 250, 1000]:
        for rep, path_seed in enumerate(_seeds(seed, 2, repeats)):
            rows.append({"replicate": rep, **population_audit(config, default_types(), N, path_seed)})
    return pd.DataFrame(rows)


def sensitivity_study(config, trials, seed):
    rows = []
    strategies = ["filtered_mfg", "naive_filter_mfg", "blind_mfg", "full_info_mfg"]
    for impact in [0, 0.04, 0.08, 0.16, 0.32]:
        for volatility in [0.06, 0.12, 0.24]:
            c = replace(config, permanent_impact=impact, volatility=volatility)
            for trial, path_seed in enumerate(_seeds(seed, 3, trials)):
                for strategy in strategies:
                    r = simulate(c, strategy=strategy, seed=path_seed)
                    rows.append({"permanent_impact": impact, "volatility": volatility,
                                 "trial": trial, "seed": path_seed, "strategy": strategy,
                                 "objective": r.metrics["objective"],
                                 "posterior_brier": r.metrics["posterior_brier"]})
    return pd.DataFrame(rows)


def _markdown_table(frame, columns, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(formats.get(col, str)(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def make_report(output, summary, population, convergence, trials, manifest):
    objective = summary[summary.metric == "objective"].copy()
    objective["95% interval"] = objective.apply(lambda r: f"[{r.lower:.6f}, {r.upper:.6f}]", axis=1)
    delta = summary[summary.metric == "objective_difference_vs_twap"].copy()
    delta["95% paired interval"] = delta.apply(lambda r: f"[{r.lower:.6f}, {r.upper:.6f}]", axis=1)
    finite = population.groupby("population_size").agg(
        flow_rmse=("mean_flow_rmse", "mean"),
        mean_gain=("mean_deviation_gain", "mean"),
        max_gain=("max_deviation_gain", "mean")).reset_index()
    slope = np.polyfit(np.log(finite.population_size), np.log(finite.flow_rmse), 1)[0]
    selected = delta.loc[delta.strategy == "filtered_mfg"].iloc[0]
    paired_matrix = trials.pivot(index="trial", columns="strategy", values="objective")
    versus_ac = mean_interval(paired_matrix.filtered_mfg - paired_matrix.ac)
    versus_naive = mean_interval(paired_matrix.filtered_mfg - paired_matrix.naive_filter_mfg)
    costs = _markdown_table(objective, ["strategy", "mean", "95% interval"], {"mean": lambda x: f"{x:.6f}"})
    diffs = _markdown_table(delta, ["strategy", "mean", "95% paired interval"], {"mean": lambda x: f"{x:.6f}"})
    finite_table = _markdown_table(finite, ["population_size", "flow_rmse", "mean_gain", "max_gain"],
                                  {"population_size": lambda x: str(int(x)),
                                   "flow_rmse": lambda x: f"{x:.6f}",
                                   "mean_gain": lambda x: f"{x:.3e}", "max_gain": lambda x: f"{x:.3e}"})
    report = f"""# Research results — auditable execution games

This report is generated by the experiment runner from retained trial-level CSV files.
All data are simulated under the documented model. Numbers are normalized objective
units, not dollars, basis points, investment returns, or empirical trading profits.

## Run

- Profile: `{manifest['profile']}`; {manifest['trials']} independent common-noise paths.
- Grid: {manifest['config']['steps']} decisions over one normalized horizon.
- Seed: `{manifest['seed']}`. Every strategy in a trial sees the same fundamental path.
- The full configuration, package versions, runtime, and output hashes are in `manifest.json`.
- Confidence intervals are two-sided 95% Student-t intervals across independent paths.
  They describe Monte Carlo uncertainty conditional on the chosen simulator, not model uncertainty.

## Does strategic execution improve the chosen objective?

Lower objective is better. The objective includes marked implementation shortfall,
running inventory penalties and a terminal residual-inventory penalty. TWAP fully
liquidates; the other policies have a soft terminal objective, so residual inventory
is explicitly retained and charged. This is not an apples-to-apples hard-liquidation claim.

{costs}

Paired differences against TWAP use each path as its own control:

{diffs}

The filtered policy's mean difference is **{selected['mean']:.6f}** with interval
**[{selected['lower']:.6f}, {selected['upper']:.6f}]**. This result is conditional on
the risk and impact parameters. A lower penalized objective need not imply lower
unpenalized shortfall. The full-information policy sees the current regime only,
not future states or noise. It is an information benchmark implemented with the
same receding-horizon approximation, not a certified lower bound on achievable cost.

The filtered policy **does not establish superiority over the simpler AC-style
baseline**: its paired difference is {versus_ac['mean']:.6f}, with interval
[{versus_ac['lower']:.6f}, {versus_ac['upper']:.6f}]. Against the uncorrected filter,
the paired difference is {versus_naive['mean']:.6f}, with interval
[{versus_naive['lower']:.6f}, {versus_naive['upper']:.6f}]. These comparisons are
retained even when additional model complexity fails to improve cost.

![Paired policy comparison](figures/policy_comparison.png)

## Why causal impact correction matters

A seller's own population depresses observed prices. Treating that movement as
exogenous alpha changes regime beliefs. The correct filter subtracts known
permanent impact before the Gaussian observation update. The naive policy omits
this correction; the misspecified policy halves assumed impact, reduces assumed
volatility to 65%, and makes regimes four times more persistent per discrete step.
The latter is a combined stress and cannot identify the contribution of each error.

The posterior Brier score measures the auxiliary filter after each interval.
The three filter-driven policies trade using its causal prior. `filter_prior_brier`
preserves that auxiliary prior diagnostic for all strategies; `decision_brier`
instead scores actual planner information. It is zero for the current-state-informed
policy and unavailable for TWAP, AC-style, and drift-blind policies, which do not
use regime probabilities. The stored `beliefs` and `posteriors` arrays remain
auxiliary-filter diagnostics. The example is the prespecified seed, not a selected
best path.

![A single paired path](figures/execution_path.png)

![Filtering reliability](figures/filter_diagnostics.png)

## Does the fixed point actually converge?

Every best response solves a strictly convex quadratic program in closed form.
The mean-flow map is affine, which supplies an independent dense linear-system
reference. Iteration stops on the undamped consistency residual, not the damped
step size. Across the prespecified coupling/damping grid,
**{int(convergence.converged.sum())}/{len(convergence)}** cases reached tolerance
within 200 iterations. Failures are retained in `convergence_summary.csv`.
The iteration matrix's spectral radius explains local/global stability of this
affine map; it is not a general convergence theorem for nonlinear MFGs.

![Convergence and stability](figures/equilibrium_convergence.png)

## What survives a finite population?

For each population size, {manifest['population_repeats']} independent populations
are sampled from the same heterogeneous type distribution. Initial holdings have
bounded mean-zero variation. Each agent applies the limiting open-loop schedule
using its own inventory. We measure mean-flow sampling error and compute every
agent's exact unilateral optimum, holding other schedules fixed and internalizing
its own 1/N permanent impact. All finite-player Hessians must pass a convexity check.

{finite_table}

The descriptive log-log slope of mean flow error against N is **{slope:.3f}**.
This is a finite-sample diagnostic, not a fitted asymptotic rate theorem. The
reported maximum gain is first maximized over agents within a population, then
averaged across populations. It covers deterministic open-loop deviations only;
it does not certify an epsilon-Nash bound over adaptive strategies.

![Finite-population audit](figures/finite_population.png)

## Sensitivity and failure modes

The coupling/noise grid reuses paired random paths within each scenario and retains
every result. Heatmap entries compare filtered and naive objective values, so a
positive value means omitting the impact correction was better on that scenario's
sample. These exploratory comparisons use {manifest['sensitivity_trials']} trials
per cell; there is no multiple-testing adjustment and no cell selection for a
profitability claim.

![Sensitivity grid](figures/sensitivity.png)

## Numerical integrity

- Largest cash/inventory accounting error across the main study:
  `{trials.accounting_error.max():.3e}`.
- Inventory dynamics, objective gradients, finite-player Hessians, independent
  numerical minimization, causal prefix invariance and analytic zero-impact
  cases are covered by the test suite.
- Signed trading rates are allowed. The raw results include buyback frequency,
  turnover, terminal inventory and unpenalized shortfall.
- All study outcomes are reported. No financial market data were used and no
  calibration, brokerage execution, limit-order fills, or live profitability is claimed.

## Scope of the result

The separate [equal-completion supplement](../hard_liquidation/REPORT.md) repeats
all seven policies under an exact zero-terminal-inventory equality on 512 additional
paired paths. It is a labeled extension of this soft-terminal study. It does not
establish a filtered-policy improvement over TWAP; its paired interval spans zero.
The simpler AC-style policy has a lower mean penalized objective, although the
filtered-minus-AC interval also spans zero. Rate and inventory bounds remain absent.

This project solves a specified discrete deterministic forecast game exactly and
uses it as a causal receding-horizon policy in a stochastic simulator. It does not
solve the full partial-information stochastic mean-field game of Casgrain and
Jaimungal. Common noise, shared information, endogenous prices, and a latent Markov
chain are simulated, but uncertainty in future learning is replaced by conditional
mean forecasts when planning. See [the derivation](../../docs/methodology.md),
[assumptions and limitations](../../docs/limitations.md), and
[research provenance](../../docs/references.md) for the precise boundary.
"""
    (output / "report.md").write_text(report)


def run_suite(output, profile="quick", trials=None, seed=20261117):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if profile not in {"quick", "full"}:
        raise ValueError("Choose quick or full profile")
    full = profile == "full"
    trials = trials or (512 if full else 32)
    if trials < 2:
        raise ValueError("At least two trials are required")
    config = MarketConfig(steps=40 if full else 24)
    repeats, sensitivity_trials = (128, 64) if full else (12, 8)
    print(f"Running {trials} paired paths × {len(STRATEGIES)} policies", flush=True)
    paths = compare_strategies(config, trials, seed)
    summary = summarize_trials(paths)
    print("Auditing equilibrium convergence and finite populations", flush=True)
    histories, convergence = convergence_study(config)
    population = finite_population_study(config, repeats, seed)
    print("Running prespecified impact/noise sensitivity grid", flush=True)
    sensitivity = sensitivity_study(config, sensitivity_trials, seed)
    tables = {"trials": paths, "summary": summary, "convergence_history": histories,
              "convergence_summary": convergence, "finite_population": population,
              "sensitivity": sensitivity}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    # The fixed seed is used regardless of whether the example looks flattering.
    example = {s: simulate(config, strategy=s, seed=seed) for s in STRATEGIES}
    np.savez_compressed(output / "example_paths.npz", **{
        f"{strategy}_{field}": getattr(result, field)
        for strategy, result in example.items()
        for field in ["states", "fundamental", "price", "inventories", "rates", "beliefs", "posteriors"]})
    from .plotting import make_figures
    make_figures(output, config, paths, summary, population, histories, convergence, sensitivity, example)
    source_dir = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob("*.py"))}
    manifest = {
        "project": "mean-field-execution", "package_version": __version__, "profile": profile,
        "seed": seed, "trials": trials, "population_repeats": repeats,
        "sensitivity_trials": sensitivity_trials, "config": asdict(config),
        "agent_types": [asdict(a) for a in default_types()], "strategies": list(STRATEGIES),
        "data_type": "synthetic; no empirical market data or calibration",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - start,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "source_sha256": source_hashes,
        "confidence_interval": "two-sided 95% Student t across independent path seeds",
        "seed_scheme": "SeedSequence([root_seed, study_stream]).spawn(n); uint32 path seed",
    }
    make_report(output, summary, population, convergence, paths, manifest)
    manifest["artifact_sha256"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed in {manifest['runtime_seconds']:.1f}s: {output / 'report.md'}", flush=True)
    return manifest

