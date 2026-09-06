# Wafer-centered R2 evaluation

For each condition, the observed and predicted condition means are removed before
computing R2. A value of zero is equivalent to predicting no spatial deviation from the
wafer mean. Negative values mean that the predicted spatial pattern has a larger squared
error than that zero-deviation reference.

## Median across three seeds

### Leave-one-identification-condition-out prediction

| Loss | Pattern | TPE | CMA-ES | DE | PSO | Lévy | CMA-MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear MSE | -0.140 | -0.804 | -0.139 | -0.139 | -2.143 | -0.242 | -0.914 |
| Wafer-normalized MSE | -0.140 | -0.810 | -0.139 | -0.139 | -2.341 | -0.797 | -0.916 |
| Wafer-normalized MAE | -0.138 | -1.586 | -0.144 | -0.143 | -2.418 | -0.816 | -0.970 |
| Symmetric normalized MSE | -0.140 | -0.143 | -0.139 | -0.139 | -2.372 | -0.790 | -0.758 |

### Fixed condition 3

| Loss | Pattern | TPE | CMA-ES | DE | PSO | Lévy | CMA-MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear MSE | -0.015 | -0.016 | -0.016 | -0.016 | -2.980 | -3.047 | -2.980 |
| Wafer-normalized MSE | -0.015 | -0.018 | -0.015 | -0.015 | -2.986 | -3.038 | -2.476 |
| Wafer-normalized MAE | -0.015 | -0.015 | -0.032 | -0.031 | -3.011 | -3.000 | -2.318 |
| Symmetric normalized MSE | -0.015 | -0.018 | -0.015 | -0.015 | -2.986 | -3.038 | -0.093 |

Pattern, CMA-ES, and DE converge to nearly the same spatial result for MSE-based Losses.
Their fixed-test values near -0.015 are only slightly below the flat centered reference,
but they do not demonstrate useful wafer-pattern prediction. PSO, most Lévy solutions,
and CMA-MAE produce materially worse spatial amplitudes or shapes. TPE can reach the
leading basin but is seed-unstable; the medians above should be read with the min/max
columns in `centered_r2_summary.csv`.

## Identification-condition detail for stable leading fits

| Loss and sampler | Condition 1 | Condition 2 | Condition 4 | Condition 5 |
| --- | ---: | ---: | ---: | ---: |
| Linear MSE, Pattern | -0.0803 | -0.0995 | 0.0326 | -0.1722 |
| Linear MSE, CMA-ES | -0.0814 | -0.0996 | 0.0306 | -0.1700 |
| Linear MSE, DE | -0.0814 | -0.0996 | 0.0307 | -0.1700 |
| Wafer-normalized MSE, Pattern | -0.0803 | -0.1005 | 0.0325 | -0.1722 |
| Wafer-normalized MSE, CMA-ES | -0.0805 | -0.0991 | 0.0316 | -0.1701 |
| Wafer-normalized MSE, DE | -0.0805 | -0.0991 | 0.0317 | -0.1703 |

Only condition 4 has a weak positive centered R2, about 0.03. Conditions 1, 2, 3, and 5
remain negative for every stable leading fit. Changing Loss or increasing the sampler
budget therefore improves condition-level rate transfer without recovering the measured
within-wafer spatial field. The limiting evidence is the present equation/input field
relationship, spatial alignment, or missing near-surface transport information rather
than insufficient optimization trials.
