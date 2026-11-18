# References and implementation provenance

## Primary mathematical sources

1. Philippe Casgrain and Sebastian Jaimungal, *Mean Field Games with Partial
   Information for Algorithmic Trading*, arXiv:1803.04094 (2018, revised 2019).
   [Paper](https://arxiv.org/abs/1803.04094). Establishes a heterogeneous trading
   game with latent factors, filtering, and permanent/temporary impact. Its
   stochastic FBSDE solution and theoretical finite-player guarantees are not
   reproduced or claimed by this discrete forecast implementation.
2. Philippe Casgrain and Sebastian Jaimungal, *Algorithmic Trading in Competitive
   Markets with Mean Field Games*, SIAM News (2019).
   [Author exposition](https://www.siam.org/publications/siam-news/articles/algorithmic-trading-in-competitive-markets-with-mean-field-games/).
   Provides the economic setting that motivates the model family.
3. Philippe Casgrain and Sebastian Jaimungal, *Mean-Field Games with Differing Beliefs
   for Algorithmic Trading*, Mathematical Finance 30(3), 995–1034 (2020).
   [Preprint](https://arxiv.org/abs/1810.06101),
   [publisher](https://doi.org/10.1111/mafi.12237). Motivates model-belief sensitivity.
   The present project compares shared-belief models across experiments; it does
   not implement that paper's within-game heterogeneous belief equilibrium.
4. Robert Almgren and Neil Chriss, *Optimal Execution of Portfolio Transactions*,
   Journal of Risk 3(2), 5–39 (2000/2001).
   [DOI](https://doi.org/10.21314/JOR.2001.041). Motivates the inventory-risk versus
   temporary-impact tradeoff. The baseline here is explicitly called **AC-style**,
   because it uses this project's discrete timing and soft terminal objective.

## What was developed here

- Original discretization and independently derived cash-ledger objective.
- Original exact best-response implementation and direct affine equilibrium audit.
- A stopping rule based on an undamped consistency residual with retained failures.
- Causal impact-corrected HMM implementation and no-lookahead tests.
- Finite-player self-impact Hessian and exact unilateral open-loop deviation audit.
- Paired simulation experiments, retained trial-level outputs, plots and reports.

The implementation was checked with
analytic benchmarks, independent numerical minimization, invariants and seeded
experiments. The repository's contribution is a reproducible research artifact;
established model ideas are attributed above. No claim of worldwide priority,
novel mathematical theorem or universally superior strategy is made.

The MIT license applies to original repository code and documentation. Linked
papers remain under their respective licenses and copyrights.

