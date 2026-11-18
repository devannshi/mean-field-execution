# Mathematical specification and derivation

## 1. Timing and units

There are $n$ intervals of length $\Delta t=T/n$. Prices, inventory, and time use
normalized units; the default price level of 100 is a bookkeeping offset.
At the start of interval $t$, the agent knows current price and inventory and the
history of returns and aggregate flow through interval $t-1$. It chooses a signed
selling rate $v_t$ before observing $\Delta F_t$.

The two-state regime $z_t$ determines the drift over interval $t$; afterward it
transitions according to $P$. The transition is the exact sampled transition of a
symmetric chain with switching intensity $\lambda$,

$$P_{01}=P_{10}=\tfrac12(1-e^{-2\lambda\Delta t}).$$

The return model holds the selected regime constant within each interval. It is a
discrete hidden Markov model, not an exact simulation of integrated continuously
switching drift inside an interval.

$$\Delta F_t=a_{z_t}\Delta t+\sigma\sqrt{\Delta t}\epsilon_t,
\quad \epsilon_t\overset{iid}{\sim}N(0,1).$$

For type $k$ with population weight $w_k$,

$$m_t=\sum_k w_k v_t^k,\qquad
S_{t+1}=S_t+\Delta F_t-\kappa m_t\Delta t.$$

Positive aggregate selling lowers the subsequent midprice. This model contains
permanent aggregate impact and instantaneous quadratic execution cost; it does not
contain a transient resilience kernel or a limit order book.

## 2. Derive the cost from the cash ledger

The order executes at the pre-move price less temporary impact:

$$C_{t+1}=C_t+v_t\Delta t(S_t-\eta v_t),\qquad
q_{t+1}=q_t-v_t\Delta t.$$

Marked wealth $W_t=C_t+q_tS_t$ therefore obeys

$$W_{t+1}-W_t=-\eta v_t^2\Delta t+q_{t+1}(S_{t+1}-S_t).$$

Implementation shortfall relative to the initial marked inventory is

$$q_0S_0-C_T-q_TS_T=
\sum_t\eta v_t^2\Delta t-\sum_tq_{t+1}\Delta S_t.$$

The optimized deterministic forecast objective adds running and terminal inventory
penalties and replaces the return by its forecast:

$$J(v;m)=\eta\Delta t\sum_tv_t^2+\phi\Delta t\sum_tq_{t+1}^2
+Aq_T^2-\Delta t\sum_tq_{t+1}(\widehat a_t-\kappa m_t).$$

The running penalty is a chosen quadratic preference. It is not asserted to equal
the full variance of execution cost under regime uncertainty. The simulator
records temporary, exogenous-drift, permanent-impact, noise, running-risk and
terminal-penalty components separately. Their sum is checked against the cash ledger.

## 3. Exact convex best response

Let $L$ be the lower-triangular matrix of ones, $D=\Delta tL$, and
$q=q_0\mathbf1-Dv$ the vector of post-trade inventories. Let $d$ be the final row
of $D$ as a column vector. Writing $h=\widehat a-\kappa m$,

$$J(v;m)=\tfrac12v^\top Hv+f^\top v+c,$$

$$H=2\eta\Delta tI+2\phi\Delta tD^\top D+2Add^\top,$$

$$f=-2\phi\Delta t q_0D^\top\mathbf1-2Aq_0d+\Delta tD^\top h.$$

Since $\eta>0$ and $\phi,A\ge0$, $H$ is positive definite and
$v^*=-H^{-1}f$. The implementation uses Cholesky solves, not explicit matrix
inversion. Defining

$$b=-H^{-1}(-2\phi\Delta tD^\top\mathbf1-2Ad),\qquad
G=-H^{-1}\Delta tD^\top,$$

gives the affine response $v^*=bq_0+Gh$.

An analytic check sets $\phi=0$, $h=0$ and obtains the constant rate

$$v_t^*=\frac{Aq_0}{\eta+AT},\qquad q_T^*=\frac{\eta q_0}{\eta+AT}.$$

It approaches TWAP as $A\to\infty$ but retains inventory for finite $A$. With
$A=\phi=0$ and zero drift, the optimum is no trade. A one-interval closed-form
test, finite-difference gradients and an independent BFGS optimizer also verify
the solver.

### Exact terminal completion

An agent with `hard_liquidation=True` additionally requires
$d^\top v=q_0$, equivalently $q_T=0$. Let $v_{\mathrm{unc}}=-H^{-1}f$.
The Lagrangian stationarity equation is
$Hv+f+\lambda d=0$. Solving jointly with the equality gives

$$v^*=v_{\mathrm{unc}}+
H^{-1}d\frac{q_0-d^\top v_{\mathrm{unc}}}{d^\top H^{-1}d}.$$

The denominator is positive because $H$ is positive definite and $d\ne0$.
This is an exact KKT projection in the $H$ metric, not a penalty approximation.
The gradient at the constrained optimum is normal to the feasible hyperplane;
it need not be zero. Tests check completion, tangent stationarity, and agreement
with an independent SLSQP solve. With no drift or running inventory penalty, the
hard-completion solution is TWAP.

The projected response remains affine in $q_0$ and $h$, so the mean-field
consistency solve and its residual audit apply with the projected response
coefficients. Receding-horizon replanning applies the equality to the current
inventory and remaining horizon, preserving exact completion at the final step.
The supplement sets $A=0$: a terminal penalty is redundant on $q_T=0$ and is not
needed for numerical enforcement. No rate, inventory, or participation bounds
are imposed; trading remains signed.

## 4. Mean-field consistency and convergence

For a fixed forecast vector $\widehat a$ and type inventories $q_0^k$,

$$m=\sum_kw_k[b_kq_0^k+G_k(\widehat a-\kappa m)].$$

Let $\bar G=\sum_kw_kG_k$ and $c=\sum_kw_kb_kq_0^k+\bar G\widehat a$.
Then

$$m=c-\kappa\bar Gm,\qquad (I+\kappa\bar G)m=c.$$

An LU solve supplies an independent consistency reference whenever the equilibrium
matrix is nonsingular. Residuals are checked afterward; a large condition number
is reported, not interpreted as proof of accurate economics.

The damped iteration is

$$m^{r+1}=m^r+d\{c-\kappa\bar Gm^r-m^r\},\quad 0<d\le1.$$

Its iteration matrix is $J_d=(1-d)I-d\kappa\bar G$. Because this particular map is
affine, $\rho(J_d)<1$ gives convergence from any initial guess in exact arithmetic.
The stopping rule measures the *undamped* residual

$$R(m)=\frac{\|c-\kappa\bar Gm-m\|_2}{\max(1,\|m\|_2)}.$$

Using $\|m^{r+1}-m^r\|$ alone would let a tiny damping parameter masquerade as
equilibrium. The tests explicitly reject that false convergence. The final returned
iterate is the one whose residual was measured. The study keeps failures at the
iteration cap, rather than discarding difficult parameter cases.

## 5. Causal filtering

Let $p_t=P(z_t=\cdot\mid\mathcal F_t)$ be the prior used for interval-$t$ trading.
After seeing $\Delta S_t$ and observed mean flow, remove known impact:

$$y_t=\Delta S_t+\kappa m_t\Delta t.$$

The likelihood for regime $j$ is proportional to

$$\exp\left[-\frac{(y_t-a_j\Delta t)^2}{2\sigma^2\Delta t}\right].$$

Normalize log weights with log-sum-exp to obtain the posterior $\widetilde p_t$,
then predict $p_{t+1}=\widetilde p_tP$. The forecast over the remaining horizon is
$\widehat a_{t+j}=p_tP^ja$. The decision at $t$ never uses $y_t$ or a posterior
computed from it. A suffix-perturbation test confirms that changing future
observations cannot change earlier actions.

The correctly specified experiment assumes $\kappa$ and contemporaneously observed
aggregate flow are known. The uncorrected filter instead uses $y_t=\Delta S_t$.
The misspecified stress combines three deliberate errors; it is not an isolated
causal attribution of any one parameter.

## 6. From a forecast equilibrium to a stochastic policy

The filtered strategy repeatedly solves the finite-horizon deterministic game,
executes only its first action, observes the next return and replans. Current
inventories are updated for every representative type. This conditional-mean
approximation does not value future information acquisition or solve a coupled
stochastic forward-backward system. In particular, an open-loop equilibrium at one
time does not prove the collection of adaptive receding-horizon policies is a
stochastic Nash equilibrium.

The informed comparator sees the current $z_t$, then uses $P$ to forecast future
states. It does not know the future state sequence or Brownian innovations. TWAP
sells remaining inventory uniformly. The AC-style comparator uses the same
quadratic penalties and temporary impact, with zero alpha and zero aggregate
coupling in its planner; actual execution still happens in the impacted simulator.

## 7. Exact finite-player deviation audit

The deterministic audit draws $N$ finite agents with random type and bounded
inventory variation. They use their own $q_0^i$ with the infinite-population
equilibrium flow. For agent $i$, hold all peers' schedules fixed and denote their
sum by $r_{-i}$. Its actual mean flow under a candidate deviation $v$ is
$(r_{-i}+v)/N$.

The finite-player objective therefore has

$$H_N=H-\frac{\kappa\Delta t}{N}(D+D^\top),$$

$$f_N=f\left(q_0,\widehat a-\frac{\kappa r_{-i}}N\right)
+\frac{\kappa\Delta t}{N}q_0\mathbf1.$$

The extra terms internalize the deviator's permanent impact. A Cholesky
factorization checks $H_N\succ0$; invalid parameter combinations are rejected.
This check is essential because the chosen pre-trade execution convention can
admit nonconvex finite-player objectives at excessive impact and coarse grids.

Let $v_N^*=-H_N^{-1}f_N$. The exact gain is

$$J_N(v)-J_N(v_N^*)=\tfrac12(v-v_N^*)^\top H_N(v-v_N^*)\ge0.$$

This audit gives a numerical bound over deterministic open-loop deviations for a
specified finite population. It does not cover history-dependent policies,
strategic information effects or a theorem's uniform-in-$N$ assumptions.

For a hard-completion type, the finite-player optimum uses the same KKT projection
with $H_N$, $f_N$, and $d^\top v=q_0$. The transferred mean-field plan satisfies
the same equality. Therefore the first-order term in the difference of their
quadratic objectives vanishes along their feasible displacement, and the gain
remains $\tfrac12(v-v_N^*)^\top H_N(v-v_N^*)$. The implementation continues to
require $H_N\succ0$, which is a sufficient condition; it does not attempt the more
general case where only the Hessian restricted to the constraint's tangent space
is positive definite. Independent constrained minimization and completion checks
validate the finite-player extension.

## 8. Experimental uncertainty

Independent path seeds, not individual timestamps or agents exposed to common
noise, define the replication unit. All policies in a trial share its exogenous
state sequence and fundamental returns. The paired cost difference is formed
within the trial before computing the mean, standard error and Student-t interval.
Finite-population replications independently sample types and initial inventories.
Grid comparisons are prespecified and exploratory; no multiplicity-adjusted claim
or tuning on favorable cells is made.

## 9. Auxiliary filters and actual decision information

Every strategy maintains an auxiliary HMM for diagnostics. `beliefs[t]` is its
causal prior and `posteriors[t]` is its posterior for interval $t$; those stored
arrays retain the same meaning even when the planner ignores them.
`filter_prior_brier` scores the auxiliary prior for every strategy.
`posterior_brier` and `posterior_log_loss` likewise refer to the auxiliary filter.

`decision_brier` instead follows the information actually used in planning. It is
zero for the current-state-informed policy, whose regime belief is a point mass;
it equals the causal prior score for filtered, naive-filter, and misspecified
policies. It is unavailable (`NaN`) for TWAP, AC-style, and drift-blind policies,
which do not use a state probability. This distinction changes diagnostic labels,
not policy actions, cash flows, or costs.

## 10. Soft-terminal and hard-terminal evidence

The main study retains its finite terminal penalty and charges marked residual
inventory. The separately labeled hard-completion supplement uses 512 additional
paired paths, seed base 510000, and sets every type's terminal equality on.
It does not replace the original study or select a favorable seed after inspection.
The filtered-minus-TWAP penalized-objective difference is approximately -0.001241,
with paired interval [-0.003270, 0.000788]. Equal completion therefore does not
establish an advantage for the filtered policy on this setting. The AC-style
policy has a lower mean objective, while the filtered-minus-AC interval also spans
zero. Both the full raw outcomes and the comparison denominators are retained.

