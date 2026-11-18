# Executed validation

The final published source and both experiment suites were validated locally with
Python 3.12.6.

| Check | Result |
|---|---|
| `pytest -q` | 70 passed |
| `ruff check src tests scripts` | All checks passed |
| Editable package installation | Succeeded with current README metadata |
| Full soft-terminal study | 512 paired paths × 7 policies; 768 finite populations; 3,840 sensitivity policy/path observations |
| Full-study runtime | 48.77 seconds; machine-specific |
| Hard-completion supplement | 512 additional paired paths × 7 policies; all outcomes retained |
| Supplement runtime | 21.59 seconds; machine-specific |
| Maximum hard-completion residual | 6.4e-18 weighted terminal-inventory RMS |
| Maximum accounting residual | 1.75e-13 |
| Notebook | All 8 code cells executed in a Python kernel; outputs retained and schema validated |
| Figures | Original 6 research PNGs previously inspected; refreshed exact-completion figure inspected |
| Manifests | Every recorded source and artifact SHA-256 verified for both studies |
| Diagnostic correction | Planner-information scores verified; all prior policy/cost outcomes unchanged |
| Local Markdown links | All checked links resolve, including supplement and notebook links |

The tests cover analytic and numerical best responses, KKT equality projection,
independent constrained finite-player minimization, completion under all seven
policies, objective gradients, finite-player self-impact, direct/iterative
consistency, false-convergence rejection, filter Bayes updates, impact correction,
causal prefix invariance, paired paths, and cash/inventory conservation.

Regression tests distinguish actual decision information from auxiliary filter
priors: current-state informed has zero decision Brier error; filter-driven
policies use the causal prior score; TWAP, AC-style, and drift-blind have no such
score. Integer-valued floating state paths are normalized to integer indices,
while fractional state labels are rejected.

The final reruns preserve all previously published policy costs and auxiliary
posterior scores (verified through a common ten-significant-digit serialization).
The corrections change diagnostic semantics, not actions or economic outcomes.
Both soft-terminal and hard-terminal results remain available; no unfavorable
path, policy, or failed convergence cell was removed.

GitHub Actions is configured to repeat tests, lint including scripts, and an
end-to-end smoke run on Python 3.10 and 3.12. Local checks do not establish that
remote workflows have run; published workflow history supplies that evidence.
