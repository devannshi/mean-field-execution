"""Static research figures generated exclusively from retained experiment data."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import default_types

COLORS = {"twap": "#8995a5", "ac": "#8d65b8", "filtered_mfg": "#087f8c",
          "full_info_mfg": "#1d3557", "naive_filter_mfg": "#df7148",
          "misspecified_mfg": "#bb466b", "blind_mfg": "#b6a044"}
LABELS = {"twap": "TWAP", "ac": "AC-style", "filtered_mfg": "Filtered MFG",
          "full_info_mfg": "Current-state informed", "naive_filter_mfg": "Uncorrected filter",
          "misspecified_mfg": "Misspecified MFG", "blind_mfg": "Drift-blind MFG"}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#23344d", "text.color": "#23344d",
                         "axes.edgecolor": "#b6bec8", "grid.alpha": .22,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.titleweight": "bold", "legend.frameon": False})


def _save(fig, output, name, note=None):
    if note:
        fig.text(.02, .015, note, fontsize=8, color="#596579")
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures(output, config, trials, summary, population, histories,
                 convergence, sensitivity, example):
    output = Path(output) / "figures"
    output.mkdir(exist_ok=True)
    _style()
    weights = np.array([a.weight for a in default_types()])
    time = np.arange(config.steps + 1) * config.dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    r = example["filtered_mfg"]
    for k, agent in enumerate(default_types()):
        axes[0, 0].plot(time, r.inventories[:, k], label=agent.name.title(), linewidth=2)
    axes[0, 0].set(title="Heterogeneous inventory paths", ylabel="Inventory", xlabel="Time")
    axes[0, 0].legend()
    for s in ["twap", "ac", "filtered_mfg", "full_info_mfg"]:
        axes[0, 1].plot(time[:-1], example[s].rates @ weights, color=COLORS[s], label=LABELS[s])
    axes[0, 1].set(title="Population mean selling rate", ylabel="Inventory / time", xlabel="Time")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(time, r.fundamental - config.initial_price, label="Fundamental", color="#1d3557")
    axes[1, 0].plot(time, r.price - config.initial_price, label="Impacted price", color="#087f8c")
    axes[1, 0].set(title="Permanent impact decomposition", ylabel="Price change", xlabel="Time")
    axes[1, 0].legend()
    axes[1, 1].step(time[1:], r.states, color="#b6bec8", where="pre", label="True regime")
    for s in ["filtered_mfg", "naive_filter_mfg", "misspecified_mfg"]:
        axes[1, 1].plot(time[1:], example[s].posteriors[:, 1], color=COLORS[s], label=LABELS[s])
    axes[1, 1].set(title="Posterior after each observed return", ylabel="P(positive drift)", xlabel="Time", ylim=(-.03, 1.03))
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(True)
    fig.suptitle("One prespecified synthetic market path", fontsize=17, weight="bold")
    _save(fig, output, "execution_path")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    order = ["twap", "ac", "blind_mfg", "naive_filter_mfg", "misspecified_mfg", "filtered_mfg", "full_info_mfg"]
    for ax, metric, title in zip(axes, ["objective", "objective_difference_vs_twap"],
                                ["Penalized execution objective", "Paired difference versus TWAP"]):
        tab = summary[summary.metric == metric].set_index("strategy").loc[order]
        for i, (s, row) in enumerate(tab.iterrows()):
            ax.errorbar(row["mean"], i, xerr=[[row["mean"] - row.lower], [row.upper - row["mean"]]],
                        fmt="o", color=COLORS[s], capsize=4, markersize=7)
        ax.set_yticks(range(len(order)), [LABELS[s] for s in order])
        ax.invert_yaxis()
        ax.axvline(0, color="#b6bec8", linewidth=1)
        ax.set(title=title, xlabel="Normalized cost; lower is better")
        ax.grid(axis="x")
    fig.suptitle("Policy comparison with trial-level uncertainty", fontsize=17, weight="bold")
    _save(fig, output, "policy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    filter_order = ["filtered_mfg", "naive_filter_mfg", "misspecified_mfg"]
    for i, s in enumerate(filter_order):
        row = summary[(summary.strategy == s) & (summary.metric == "posterior_brier")].iloc[0]
        axes[0].bar(i, row["mean"], color=COLORS[s], width=.55)
        axes[0].errorbar(i, row["mean"], yerr=[[row["mean"] - row.lower], [row.upper - row["mean"]]],
                         fmt="none", color="#23344d", capsize=4)
        sub = trials[trials.strategy == s]
        axes[1].scatter(sub.posterior_brier, sub.objective, s=10, color=COLORS[s], alpha=.25, label=LABELS[s])
    axes[0].axhline(.25, color="#8995a5", linestyle="--", label="Uninformative p = 0.5")
    axes[0].set_xticks(range(3), ["Corrected", "Uncorrected", "Misspecified"])
    axes[0].set(title="Posterior Brier score", ylabel="Mean squared probability error")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Belief error and realized cost", xlabel="Trial-average Brier score", ylabel="Penalized objective")
    axes[1].legend(fontsize=8)
    fig.suptitle("The observation model is part of the execution model", fontsize=16, weight="bold")
    _save(fig, output, "filter_diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for k, damping in [(0.08, 1.0), (0.32, 1.0), (0.64, 1.0), (0.64, .35), (1.28, 1.0), (1.28, .15)]:
        sub = histories[(histories.permanent_impact == k) & (histories.damping == damping)]
        axes[0].semilogy(sub.iteration, np.maximum(sub.residual, 1e-15), label=f"κ={k:g}, d={damping:g}")
    axes[0].axhline(1e-10, color="#8995a5", linestyle="--", linewidth=1)
    axes[0].set(title="Undamped consistency residual", xlabel="Iteration", ylabel="Relative residual", ylim=(1e-12, 20))
    axes[0].legend(fontsize=8, ncols=2)
    tab = convergence.pivot(index="damping", columns="permanent_impact", values="iteration_spectral_radius")
    im = axes[1].imshow(tab.values, origin="lower", aspect="auto", cmap="YlOrRd", vmin=0, vmax=2)
    axes[1].set_xticks(range(len(tab.columns)), [f"{v:g}" for v in tab.columns])
    axes[1].set_yticks(range(len(tab.index)), [f"{v:g}" for v in tab.index])
    for i in range(len(tab)):
        for j in range(len(tab.columns)):
            axes[1].text(j, i, f"{tab.iloc[i,j]:.2f}", ha="center", va="center", color="#23344d", fontsize=9)
    axes[1].set(title="Iteration spectral radius (stable < 1)", xlabel="Permanent impact κ", ylabel="Damping")
    fig.colorbar(im, ax=axes[1], shrink=.85)
    fig.suptitle("A small update is not evidence of equilibrium", fontsize=17, weight="bold")
    _save(fig, output, "equilibrium_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    means = population.groupby("population_size").mean(numeric_only=True)
    for metric, label, color in [("mean_deviation_gain", "Mean agent gain", "#087f8c"),
                                  ("max_deviation_gain", "Largest agent gain", "#df7148")]:
        axes[1].loglog(means.index, means[metric], "o-", label=label, color=color)
    axes[0].loglog(means.index, means.mean_flow_rmse, "o-", color="#087f8c", label="Mean flow RMSE")
    guide = means.mean_flow_rmse.iloc[0] * np.sqrt(means.index[0] / means.index)
    axes[0].loglog(means.index, guide, "--", color="#8995a5", label="N⁻¹ᐟ² reference")
    axes[0].set(title="Transfer to sampled finite populations", xlabel="Number of agents N", ylabel="Flow RMSE")
    axes[1].set(title="Exact open-loop unilateral deviation", xlabel="Number of agents N", ylabel="Cost improvement from deviation")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, which="both")
    fig.suptitle("Finite-population error and strategic deviation audit", fontsize=17, weight="bold")
    _save(fig, output, "finite_population")

    pivot = sensitivity.pivot(index=["permanent_impact", "volatility", "trial"], columns="strategy", values="objective")
    pivot["delta"] = pivot.filtered_mfg - pivot.naive_filter_mfg
    table = pivot.delta.groupby(["permanent_impact", "volatility"]).mean().unstack("permanent_impact")
    se = pivot.delta.groupby(["permanent_impact", "volatility"]).sem().unstack("permanent_impact")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    lim = max(abs(table.values).max(), 1e-6)
    im = ax.imshow(table.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    ax.set_xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
    for i in range(len(table)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:+.4f}\nSE {se.iloc[i,j]:.4f}", ha="center", va="center",
                    color="white" if abs(value) > .65 * lim else "#23344d", fontsize=10)
    ax.set(xlabel="Permanent impact κ", ylabel="Fundamental volatility σ",
           title="Filtered − uncorrected objective: negative favors correction")
    fig.colorbar(im, ax=ax, label="Paired mean cost difference")
    fig.suptitle("Prespecified impact and observation-noise stress grid", fontsize=16, weight="bold")
    _save(fig, output, "sensitivity")

