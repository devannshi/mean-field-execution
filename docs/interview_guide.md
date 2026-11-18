# Research discussion guide

This guide is for understanding and defending the work, not a substitute for
reading the implementation and reproducing the results.

## Explain the research question in one minute

The project asks how much confidence to place in a simulated execution equilibrium.
It checks three different gaps: numerical fixed-point consistency, information
errors caused by confusing price impact with latent drift, and strategic deviations
when an infinite-population policy is deployed in a finite population. The emphasis
is on auditable evidence and failure modes rather than a favorable strategy curve.

## Derivations to know

1. Derive the marked-wealth identity and explain why post-trade inventory multiplies
   the next price change. Changing that index changes the economic timing.
2. Write the LQ Hessian and show why temporary impact makes the nonatomic best
   response strictly convex.
3. Derive the affine mean-flow map and the direct equilibrium matrix. Distinguish
   existence of a direct solution from stability of a chosen fixed-point iteration.
4. Explain why tiny damping can create a tiny step without creating a small
   consistency residual.
5. Derive the finite-player own-impact Hessian correction and explain when it can
   cease to be positive definite.
6. Explain why a deterministic open-loop deviation audit is weaker than an
   adaptive epsilon-Nash guarantee.
7. Derive the hard-completion KKT projection
   $v^*=v_{\mathrm{unc}}+H^{-1}d(q_0-d^\top v_{\mathrm{unc}})/(d^\top H^{-1}d)$.
   Explain why the constrained gradient is normal to the feasible hyperplane,
   and why the same projection applies after finite-player self-impact corrections.

## Questions an interviewer could reasonably ask

**Why does a simpler AC-style policy sometimes compare well to the MFG policy?**
The mean-field solution is individually strategic, not a social planner minimizing
population-wide cost. Everyone responding to adverse price impact can accelerate
the population's sales. Filtering also adds imperfect information and the policy
uses a certainty-equivalent approximation. Additional complexity need not improve
the chosen metric. Inspect paired differences and their uncertainty.

**Does the full-information policy give the true optimum?**
No. It knows the current regime but still uses a receding-horizon forecast policy.
It is a comparator with a richer current information set, not a certified bound
from solving the full stochastic control problem.

**Why are the confidence intervals paired?**
Different policies are exposed to the same fundamental returns. Comparing within
each path removes part of the common market noise. Intervals are then calculated
across independent paths. Treating each time observation or agent as independent
would understate uncertainty because they share common noise.

**What happens when every policy must finish liquidating?**
The separate 512-path hard-completion study weakens the filtered policy's result:
its objective difference versus TWAP is -0.001241, with interval
[-0.003270, 0.000788]. There is no resolved improvement on this setting. AC-style
has a lower mean objective, but filtered-minus-AC is +0.001190 with interval
[-0.000126, 0.002505], so that ordering is also uncertain. AC-style versus TWAP
is negative with an interval excluding zero. These results concern the penalized
objective, not guaranteed improvements in raw execution shortfall.

**Why are there two prior Brier diagnostics?**
Every strategy runs an auxiliary filter for diagnosis, but not every planner uses
it. `filter_prior_brier` scores that filter. `decision_brier` scores actual planner
information: zero for current-state informed, the filter-prior score for the three
filter-driven policies, and unavailable for policies without regime beliefs.

**What is the biggest empirical gap?**
Impact and latent-state parameters are supplied, not calibrated. Real market data
may not identify drift regimes at useful execution horizons, and aggregate flow
is only imperfectly observed. The simulator has no spread, fill or queue model.

**How would you improve the work next?**
Extend the implemented common hard-completion constraint with rate and inventory
bounds; add a defensible transient impact model; calibrate on strictly prior data; compare
scenario-based planning with conditional-mean planning; audit adaptive deviations.

## Navigate the code in a review

- Start in `optimization.py` and reconcile its matrix expressions with the
  derivation, then read the independent optimizer tests.
- Read `equilibrium.py` and inspect a deliberately nonconvergent grid cell.
- Trace a single interval in `simulation.py`: decision, inventory, cash, price,
  observation correction, posterior, next prior.
- Inspect the finite-player cost derivative in `population.py` and its numerical test.
- Reproduce `results/published/report.md` from a fresh environment and compare
  all tables, including the cases where the proposed policy does not win.
- Reproduce `results/hard_liquidation/REPORT.md` and explain why its equal-completion
  results are less favorable to the filtered policy than the soft-terminal study.

