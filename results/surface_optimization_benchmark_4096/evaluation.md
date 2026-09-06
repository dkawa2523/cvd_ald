# Evaluation of linear-rate and normalized Losses at 4,096 trials

The benchmark fixed the selected AIB equation and role assignment, fitted conditions
1, 2, 4, and 5, and reserved condition 3 as a no-refit audit. Ranking used the median
leave-one-identification-condition-out RMSE over three seeds. Each stochastic fit and
each of its four condition refits used 4,096 shape trials. Pattern search retained its
deterministic 1,010-evaluation design.

`mse` is the mean squared residual in the linear deposition-rate unit. The normalized
Losses also use linear rate residuals; they differ only in condition scaling. Positive
kinetic shape parameters use log10 search coordinates because their declared feasible
ranges span twenty decades. This coordinate transform does not make the response Loss
logarithmic.

| Loss | Best sampler | Condition-CV RMSE (nm/s) | Fixed-test RMSE (nm/s) | Interpretation |
| --- | --- | ---: | ---: | --- |
| Wafer-normalized MSE | Pattern | 0.000888134 | 0.000993829 | Best transfer score; selects the near-zero-inhibition boundary |
| Symmetric normalized MSE | Pattern | 0.000889261 | 0.000994608 | Nearly identical transfer, with numerical scale profiling |
| Linear-rate MSE | Pattern | 0.000894084 | 0.00104863 | Best condition CV within the dimensional Loss |
| Wafer-normalized MAE | Pattern | 0.000945823 | 0.000995128 | Robust point residual, but poorer condition transfer |

CMA-ES and DE reached a lower full-training objective than pattern search. For
linear-rate MSE, CMA-ES reduced the fitted objective by 0.135%, but condition-CV RMSE
increased by 0.245%; its fixed-test RMSE decreased by 2.80%. For wafer-normalized MSE,
CMA-ES reduced the fitted objective by 0.195%, while condition-CV RMSE increased by
0.506%. CMA-ES converged to the same parameters in all three seeds, and DE reached the
same basin with a small remaining spread.

This difference is physically and statistically relevant. Pattern search stopped at an
inhibition ratio of approximately 1.2e-10, while the converged CMA-ES solution used an
inhibition ratio near 0.011 for normalized MSE and 0.017 for linear MSE. Pattern's small
condition-CV advantage is consistent with implicit simplification caused by search
resolution, although the unrecorded fold parameters prevent assigning the CV difference
to that cause alone. Optimizer nonconvergence must not be used as regularization. If the
inhibitor effect is unsupported, the explicit no-inhibition reduction should be fitted
and compared under the same converged sampler.

Raising the trial budget materially improved differential evolution: its median
condition-CV RMSE fell by 71-81% across all four Losses relative to 1,024 trials. Lévy
flight improved by 27-72%, but retained large seed spread and frequent poor spatial
solutions. PSO was essentially unchanged and continued to enter a high-ratio basin.
CMA-MAE improved only in selected cases and remains inappropriate as the primary
minimum-Loss backend. TPE occasionally found the good basin, but its median was unstable
and it had the highest computational cost.

The fixed-test centered spatial R2 of the leading combinations remains near -0.015.
Neither additional trials nor the new samplers recover the measured wafer-scale spatial
variation. The optimization conclusions concern convergence and condition-level rate
transfer; they do not establish a microscopic role assignment or wafer correction.

For the present low-dimensional equation census, pattern search remains useful for fast
screening, CMA-ES is the strongest convergence audit, and DE is the faster global
alternative when at least 4,096 trials are affordable. Model adoption should use exact
reductions and condition CV after confirming that the selected sampler actually
minimized its declared Loss.
