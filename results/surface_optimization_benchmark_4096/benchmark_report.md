# Loss and sampler benchmark

Fixed equation: `cvd:aib_qss:AIB:full:bulk_as_surface:A=idn_2,I=n2,B=adn_2`.

Combinations are ranked by the median leave-one-identification-condition-out RMSE. The fixed test condition is evaluated after ranking and is not used for selection.

Stochastic fits use 4,096 evaluations and 3 seeds. The current pattern/MSE reference uses 1,010 evaluations per full fit. Runs used 16 worker(s). Elapsed time covers one full fit and all condition-refits and may include concurrent CPU contention.

|Rank|Loss|Sampler|Condition CV RMSE (nm/s)|Seed range|Fixed test RMSE (nm/s)|Test centered R2|Median time (s)|
|---:|---|---|---:|---:|---:|---:|---:|
|1|Wafer-normalized MSE|pattern|0.000888134|0.000888134–0.000888134|0.000993829|-0.0153|0.779|
|2|Symmetric normalized MSE|pattern|0.000889261|0.000889261–0.000889261|0.000994608|-0.0153|5.878|
|3|Wafer-normalized MSE|cmaes|0.000892628|0.000892628–0.000892628|0.00100971|-0.0152|43.835|
|4|Wafer-normalized MSE|de|0.000892792|0.000892787–0.000893437|0.00100985|-0.0152|10.721|
|5|Symmetric normalized MSE|cmaes|0.000892809|0.000892809–0.000892809|0.00101072|-0.0153|77.415|
|6|Symmetric normalized MSE|de|0.000892976|0.000892891–0.000893795|0.00102039|-0.0153|35.745|
|7|Linear MSE|pattern|0.000894084|0.000894084–0.000894084|0.00104863|-0.0148|0.364|
|8|Linear MSE|de|0.000896204|0.000895241–0.000896378|0.00101872|-0.0159|11.179|
|9|Linear MSE|cmaes|0.000896275|0.000896275–0.000896275|0.00101927|-0.0161|43.516|
|10|Symmetric normalized MSE|tpe|0.000940213|0.00088991–0.00331216|0.00109461|-0.0179|419.025|
|11|Wafer-normalized MAE|pattern|0.000945823|0.000945823–0.000945823|0.000995128|-0.0146|0.798|
|12|Wafer-normalized MAE|de|0.000973107|0.000968774–0.000973921|0.000964359|-0.0314|10.824|
|13|Wafer-normalized MAE|cmaes|0.000973223|0.000973223–0.00110822|0.000964377|-0.0322|40.632|
|14|Wafer-normalized MAE|levy|0.0013608|0.00102051–0.00207811|0.00193026|-3.0003|133.908|
|15|Linear MSE|tpe|0.00181345|0.000889853–0.00577244|0.00119146|-0.0164|471.362|
|16|Wafer-normalized MSE|levy|0.00186225|0.00179192–0.00593699|0.00194036|-3.0381|130.727|
|17|Wafer-normalized MSE|tpe|0.00186491|0.00117798–0.00334729|0.00107941|-0.0179|467.962|
|18|Symmetric normalized MSE|levy|0.00186617|0.0017995–0.00376357|0.00194242|-3.0385|147.594|
|19|Wafer-normalized MAE|tpe|0.00220177|0.00159013–0.00327276|0.00101645|-0.0150|440.462|
|20|Linear MSE|levy|0.00253898|0.00174618–0.00484011|0.00207347|-3.0473|127.054|
|21|Wafer-normalized MAE|pso|0.00346753|0.00342593–0.00346852|0.00196289|-3.0114|10.319|
|22|Symmetric normalized MSE|pso|0.00348294|0.00331446–0.00363937|0.00189785|-2.9859|36.423|
|23|Wafer-normalized MSE|pso|0.00363961|0.00335711–0.00387195|0.00189708|-2.9856|10.322|
|24|Linear MSE|pso|0.0036605|0.00265906–0.0036638|0.00189317|-2.9801|10.648|
|25|Symmetric normalized MSE|cma_mae|0.00463836|0.00335223–0.00589241|0.00175629|-0.0933|39.128|
|26|Wafer-normalized MSE|cma_mae|0.00643337|0.00560375–0.00711148|0.00359879|-2.4763|19.411|
|27|Linear MSE|cma_mae|0.00656842|0.00652052–0.0066321|0.00189322|-2.9803|20.369|
|28|Wafer-normalized MAE|cma_mae|0.00810296|0.00353927–0.00816105|0.00283796|-2.3183|19.516|

The fixed test condition is condition 3. A lower transfer RMSE does not by itself establish wafer-pattern recovery; the centered R2 column assesses that separate question.
