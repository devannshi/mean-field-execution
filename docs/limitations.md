# Assumptions, limitations and next experiments

## What the project establishes

The exact solver minimizes a stated discrete convex forecast objective. The direct
mean-field solve and residual-audited iteration agree when the iteration converges.
The simulator conserves cash and inventory under its timing convention. Filtering
and trading decisions are causal. Finite-player deviations are solved for the
specified deterministic open-loop game. The retained results quantify these
properties and policy behavior under a known simulator.

## What it does not establish

- **A full stochastic MFG solution.** The receding-horizon policy substitutes
  conditional-mean forecasts for future stochastic learning. It is not the FBSDE
  solution in Casgrain and Jaimungal, nor a general equilibrium existence proof.
- **Universal equilibrium stability.** An affine-map spectral radius diagnoses
  this discretization. Nonlinear impact, constraints, heterogeneous beliefs and
  general common-noise equilibria require different analysis.
- **A live execution advantage.** There is no exchange data, empirical parameter
  calibration, transaction-cost calibration, queue model, fill uncertainty,
  spread, venue fragmentation or latency. Costs are normalized simulation units.
- **A guaranteed adaptive epsilon-Nash bound.** The finite-player diagnostic holds
  peers' deterministic schedules fixed. It does not optimize arbitrary adaptive
  deviations or changes in other agents' beliefs.
- **A universal benefit from strategic execution.** The default soft-terminal
  study permits residual inventory and charges its marked value and penalty.
  A separate exact-completion supplement requires zero terminal inventory for
  every policy. In that supplement, filtered-minus-TWAP is -0.001241 with interval
  [-0.003270, 0.000788], which does not establish an improvement. The simpler
  AC-style policy has a lower mean objective, but filtered-minus-AC also has an
  interval spanning zero. A lower penalized objective is not necessarily lower
  unpenalized execution shortfall.
- **Constrained execution.** Signed rates can buy back shares or short inventory.
  There are no participation limits, nonnegativity constraints or borrowing costs.
- **Distribution-free uncertainty.** Intervals quantify finite Monte Carlo error
  conditional on the model. They exclude uncertainty about impact form, regime
  dynamics, parameters, market frictions and investment applicability.

## Information assumptions

The baseline filter observes aggregate flow and knows the permanent-impact
coefficient, noise level, latent drift levels and transition matrix. Those
parameters are supplied, not learned. All agents within a strategy share one
public belief process; type heterogeneity is in initial holdings, inventory
penalty and temporary friction. The informed comparator observes the current
regime but not future states. The misspecification case intentionally changes
several parameters together and is described as a combined stress.

All strategies also maintain an auxiliary diagnostic filter. Its stored priors
and posteriors are not the planner's decision information for the informed,
drift-blind, TWAP, or AC-style policies. The actual `decision_brier` is zero for
the informed policy, is the causal prior score for the three filter-driven
policies, and is unavailable for policies that do not use state probabilities.

## Discretization and convexity

Orders execute at the beginning-of-interval midprice and temporary impact.
Permanent flow impact then enters the next price. For a finite player, excessive
self-impact relative to temporary friction can destroy convexity in this discrete
convention. The deviation solver explicitly checks its Hessian rather than
returning an apparent optimum for such a case. The drift is piecewise constant
per grid interval; an exact chain transition does not make the integrated drift
an exact continuous-time simulation. Hard-completion finite-player deviations
also require a positive-definite full Hessian in this implementation, even though
positive definiteness only on the equality's tangent space would suffice in a
more general constrained solver.

## Useful extensions

1. Add rate, inventory, and participation bounds to the implemented hard terminal
   equality, using a constrained quadratic-program solver. Completion alone does
   not prevent buybacks or intermediate short positions.
2. Add a transient impact state and verify positive-definiteness/no-manipulation
   conditions before interpreting cost improvements.
3. Estimate impact and regime parameters on public historical data with a strict
   training/evaluation split; propagate calibration uncertainty into the policy.
4. Replace certainty-equivalent forecasts with a scenario-tree or least-squares
   Monte Carlo approximation that accounts for future filtering information.
5. Audit adaptive finite-player deviations on held-out common-noise paths rather
   than interpreting open-loop gains as a complete Nash certificate.
6. Quantify discretization error with coupled fine/coarse Brownian paths and a
   continuously switching integrated drift simulator.

