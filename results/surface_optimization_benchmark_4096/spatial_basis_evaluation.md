# Transferable wafer-shape diagnostic

The negative centered R2 values from the reaction-role equations do not support wafer-map
prediction. A follow-up diagnostic tested whether the missing spatial signal is shared
between process conditions. Every spatial coefficient was fitted only to the remaining
identification conditions and transferred to the held-out condition without refitting.

Measured centered maps are strongly repeatable: pairwise correlations across conditions
range from 0.869 to 0.994. This indicates a common reactor or wafer-position response
that is not represented by the local bulk concentration fields.

| Shared spatial component | Condition 1 | Condition 2 | Fixed condition 3 | Condition 4 | Condition 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chemistry only | -0.0803 | -0.0995 | -0.0148 | 0.0326 | -0.1722 |
| Centered rho^2 | 0.1561 | 0.0799 | 0.2115 | 0.2000 | 0.1311 |
| Centered rho^2 and rho^4 | 0.7963 | 0.6994 | 0.8456 | 0.7704 | 0.7737 |
| Centered rho^2, rho^4, and rho^6 | 0.8156 | 0.7207 | 0.8637 | 0.7891 | 0.7906 |
| Shared value at every sampled coordinate | 0.9992 | 0.9699 | 0.9915 | 0.9790 | 0.9266 |

The coordinate-wise result is an upper-bound diagnostic with 49 spatial degrees of
freedom and is not suitable as the production model. The two-parameter radial component
captures most of the transferable spatial signal. Adding rho^6 gives only about 0.02
absolute R2 improvement and should remain a separately validated candidate.

A production spatial layer should be shared by all identification wafers and preserve
the mean rate predicted by the chemical equation. One positive form is

\[
\widehat v_{qn}=\overline v^{\mathrm{chem}}_q
\frac{f_{qn}(\phi)\exp[g(\rho_n,\theta_n;\gamma)]}
{\langle f_q(\phi)\exp(g)\rangle_q},
\qquad
g=\gamma_2(\rho^2-\langle\rho^2\rangle)
 +\gamma_4(\rho^4-\langle\rho^4\rangle).
\]

The normalization prevents the spatial layer from changing the condition mean, leaving
condition-scale chemistry and shared wafer shape as separate reported contributions.
Candidate spatial bases should be `none`, radial order 1, radial order 2, and a compact
low-order Zernike basis. Select them by whole-condition holdout, using the same linear or
wafer-normalized Loss, and report centered R2 as the shape diagnostic. Do not fit an
independent correction for each wafer.

This spatial layer can support wafer-uniformity prediction for conditions sharing the
same reactor geometry and wafer placement. It cannot identify a chemical pathway. Direct
surface concentrations or wall-normal species fluxes would still be required to replace
the empirical shared shape with a transport-based explanation.
