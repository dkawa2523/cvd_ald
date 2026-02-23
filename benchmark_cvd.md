# AIBベンチマーク仕様（`benchmark_wafer2d`）

本書は `src/deposim_sim/benchmark_wafer2d.py` の現行仕様を記述する。
対象は `sim.model.name=aib_ode` のみであり、legacy `power_law/lhhw/root_solve` は本文の対象外とする。

## 1. 目的
- AIB-ODEのクラス差（A/AI/AB/AIB）を同一入出力契約で比較する。
- 実装の受入指標を `phi_B`, `f_I`, `CsA_over_CrefA`, `CsB_over_CrefB`, `residual_nm` に統一する。
- ランキング成果物 (`ranking.csv`, `class_compare.csv`) を毎回生成する。

## 2. 方程式（AIB-ODE）

### 2.1 被覆率補助変数
$$
\theta_* = \frac{1-\theta_A}{1 + K_I C_{ref,I}}
$$

### 2.2 Aの表面濃度
$$
C_{s,A} = \frac{k_{m,A}C_{ref,A} + \Gamma_s k_{des}\theta_A}{k_{m,A} + \Gamma_s k_{ads}\theta_*^{m_{ads}}}
$$

### 2.3 進化方程式
$$
\frac{d\theta_A}{dt} = k_{ads}C_{s,A}\theta_*^{m_{ads}} - k_{des}\theta_A - \nu_A r_{event}
$$

$$
\frac{dh}{dt} = \alpha_h\Gamma_s r_{event}
$$

### 2.4 B寄与（Bありクラスのみ）
$$
\phi_B = \frac{\Gamma_s k_{rxn}\theta_A^{p_A}\theta_*^{p_*}}{C_{B,scale}\,k_{m,B}}
$$

Bなしクラス（A, AI）では `phi_B`, `CsB_over_CrefB` は `NaN` を正とする。

## 3. 入力契約
- `sim.model.name=aib_ode` 必須。
- `sim.domain.kind=from_fluent_xy` を前提（再格子化は行わない）。
- `sim.roles` は `A` 必須、`I/B` は各0または1、かつ相互排他。
- Fluent入力は `xy:[n_pts,2]`, `cref:[n_pts,n_species]`（steady）を使用。

## 4. ケース設計（固定4ケース）
- `CASE-A`   : class=`A`
- `CASE-AI`  : class=`AI`
- `CASE-AB`  : class=`AB`
- `CASE-AIB` : class=`AIB`

各ケースは同一点群に対し `roles` と一部パラメータ（`K_I`, `k_rxn`, `C_B_scale`）のみ変更する。

## 5. 出力成果物
`results/<project>/runs/<run_id>/` 以下に最低限を出力する。

- `summary.json`
- `report.html`
- `outputs/benchmark_cases.npz`
- `outputs/benchmark_case_metrics.json`
- `outputs/ranking.csv`
- `outputs/class_compare.csv`
- （physviz有効時）`outputs/physviz_maps.npz`

`benchmark_case_metrics.json` の主キー:
- `case_id`, `class_id`
- `mean_h_nm`
- `mean_phi_B`
- `mean_f_I`
- `mean_CsA_over_CrefA`
- `mean_CsB_over_CrefB`
- `mean_abs_residual_nm`
- `score`

## 6. 合否判定
`summary.json -> trend_assertions` で次を判定する。

- `assert_class_coverage`
- `assert_ai_inhibition`（AIの `mean_f_I` がAより小さい）
- `assert_aib_inhibition_vs_ab`（AIBの `mean_h_nm` がABより小さい）
- `assert_ab_phi_b_finite`
- `assert_aib_phi_b_finite`
- `assert_a_phi_b_nan`
- `assert_ai_phi_b_nan`
- `overall_passed`（全assertのAND）

## 7. 実行コマンド
- `./scripts/commands.sh benchmark_wafer2d`
- `./scripts/commands.sh benchmark_wafer2d_physviz`

## 8. 履歴（legacy）
`power_law`, `lhhw_competition`, `root_solve`, `run_cvd_steady` を主経路とする旧ベンチ仕様は廃止済み。
必要時は Git 履歴を参照すること（実行時フォールバックは持たない）。
