# CVDベンチマーク設定と結果解釈（`benchmark_wafer2d`）

## 0. 今回の対象ラン
- 対象ランID: `benchmark_wafer2d_20260219T235827612318Z`
- 出力先: `results/runs/benchmark_wafer2d_20260219T235827612318Z`
- モード: `wafer_2d_polar`, `cvd_steady`（physviz付き）

## 1. 本ベンチマークで考慮した方程式

### 1.1 輸送-表面結合（定常）
各種 `i` について

$$
J_i = k_{m,i}\left(C_{\mathrm{ref},i} - C_{s,i}\right)
$$

$$
\nu_i R = J_i
$$

ここで `R` は表面進行速度（gross）、`k_{m,i}` は物質移動係数、`C_ref` は参照面濃度、`C_s` は表面濃度、`nu_i` は化学量論係数。

厚みは

$$
\frac{\partial h}{\partial t} = R_{\mathrm{net}}, \quad h(t)=\int_0^t R_{\mathrm{net}}(\tau)\,d\tau
$$

### 1.2 反応速度式（理論）
本ベンチでは以下を使用。

1. `power_law`

$$
R = k_0 \prod_i C_{s,i}^{n_i}
$$

2. `lhhw_competition`（`saturation_inhibition` クラス）

$$
R =
\frac{k_0 \prod_i C_{s,i}^{n_i^{(\mathrm{num})}}}
{\left(b_0 + \sum_i a_i C_{s,i}^{m_i}\right)^p}
$$

3. 温度依存（Arrhenius）

$$
k(T) = k_0 \exp\left(-\frac{E_a}{RT}\right)
$$

### 1.3 ネット式（堆積・エッチ・損失）
`dep_etch_loss` を用いるケースでは

$$
R_{\mathrm{net}} = R_{\mathrm{dep}} - R_{\mathrm{etch}} - R_{\mathrm{loss}}
$$

$$
R_{\mathrm{etch}} = f_{\mathrm{etch}}R_{\mathrm{dep}},\quad
R_{\mathrm{loss}} = f_{\mathrm{loss}}R_{\mathrm{dep}}
$$

### 1.4 入力分布の擬似時系列
physvizでは `C_ref` を時刻比 `t/T` で変調し、各時点の準定常解を積算して時系列可視化する。

$$
C_{\mathrm{ref}}(r,\theta,t) = C_{\mathrm{ref},0}(r,\theta)\left[1 + A\cdot W(r,\theta,t)\right]
$$

（実装では `W` は半径・角度・位相を含む波形）

## 2. ベンチマークケース
- `CASE-01_SYN_UNIFORM_RL`: 一様入力、反応律速寄り
- `CASE-02_SYN_UNIFORM_TL`: 一様入力、輸送律速寄り
- `CASE-03_SYN_RADIAL_GRAD`: 中心高濃度の半径勾配
- `CASE-04_FILE_THETA_PATTERN`: file入力（角度変調）
- `CASE-05_FILE_EDGE_DEPLETED`: file入力（外周希薄）
- `CASE-06_SYN_SEEDED_LHHW_NET`: seededムラ + LHHW + dep/etch/loss
- `CASE-07_FILE_COMPLEX_LHHW_NET`: 複合2D file入力 + LHHW + dep/etch/loss

## 3. 妥当性評価（今回ランの具体値）

### 3.1 合否
- `overall_passed = true`
- `assert_solver_health = true`（全ケース `root_failure_fraction = 0`）

### 3.2 基本トレンドが妥当な点
1. 反応律速/輸送律速の分離が出ている
   - `mean(Cs/Cref)` 差: `0.1983566`（CASE-01 > CASE-02）
   - `mean(Da_proxy)` 差: `0.1983566`（CASE-02 > CASE-01）
2. 半径トレンドが正しい
   - CASE-03: `center_mean=1.1855 > edge_mean=0.7212`
3. 入力分布の転写が強い
   - CASE-04 `theta_transfer_corr = 0.9999999998`（正相関）
4. 外周低濃度ケースの中心優勢
   - CASE-05: `center_mean=1.1821 > edge_mean=0.3849`

### 3.3 複雑ケース（LHHW+net）の解釈
1. CASE-06（代表ケース）で反応重要度上位が多項化
   - `ea_j_mol`: `importance_score=1.1959`（最大、符号マイナス）
   - `k0`: `0.3002`
   - `denominator_base`: `0.2077`
   - `denominator_power`: `0.2074`
   - `denominator_coeffs.precursor`: `0.1893`
2. `dep_etch_loss`寄与は設定と整合
   - ネット寄与平均: `etch_fraction_of_dep=0.10`, `loss_fraction_of_dep=0.03`

上記は「単一 `k0/order` だけでなく、LHHW分母側パラメータも効いている」ことを示し、今回の目的（複雑理論項の可視化）に対して妥当。

## 4. 出力画像の意味

### 4.1 入力分布（新規）
- `physviz_input_cref_<species>_tXX.png`
  - 時刻比 `t/T=XX%` における入力 `C_ref` の空間分布
- `physviz_input_delta_cref_<species>.png`
  - 隣接時刻の入力差分の最大絶対値マップ

### 4.2 時間×空間
- `physviz_cvd_thickness_tXX.png`
  - 時刻ごとの厚みマップ
- `physviz_cvd_delta_thickness_step.png`
  - 時間ステップ間の厚み差分（最大絶対値）
- `physviz_cvd_linearity_residual.png`
  - 線形積算近似からの偏差

### 4.3 輸送項
- `physviz_transport_capacity_<species>.png`
  - $k_m C_{ref}$（輸送可能量）
- `physviz_reaction_demand_<species>.png`
  - $\nu R$（反応要求量）
- `physviz_depletion_ratio_<species>.png`
  - $1-C_s/C_{ref}$（枯渇度）
- `physviz_utilization_<species>.png`
  - $\nu R/(k_m C_{ref}+\varepsilon)$（利用率、Da解釈）

### 4.4 反応項重要度
- `physviz_reaction_sensitivity_<param>.png`
  - $S_p=\partial\log h/\partial\log p$ の空間分布
- `physviz_reaction_ablation_<term>.png`
  - 項OFF時の厚み差分マップ
- `physviz_reaction_importance_rank.png`
  - 面積重み付き統合スコア順位

### 4.5 ネット項重要度
- `physviz_net_dep_rate.png`
  - 堆積速度
- `physviz_net_etch_rate.png`
  - エッチ速度
- `physviz_net_loss_rate.png`
  - ロス速度
- `physviz_net_contribution_rank.png`
  - `etch_fraction_of_dep`, `loss_fraction_of_dep` の寄与順位

## 5. 重要度変数（param/term）の意味

### 5.1 反応パラメータ
- `k0`: 反応前因子（反応速度スケール）
- `ea_j_mol` / `ea`: 活性化エネルギー（温度感度）
- `order`, `orders.*`: 濃度冪指数（power-law）
- `numerator_orders.*`: LHHW分子側の濃度指数
- `denominator_coeffs.*`: LHHW分母の吸着強度係数
- `denominator_orders.*`: LHHW分母の濃度指数
- `denominator_power`: 分母全体の冪
- `denominator_base`: 分母定数項
- `pattern_loading`: パターン負荷係数（有効化学量論への空間スケーリング）

### 5.2 ネット寄与変数
- `etch_fraction_of_dep`: 堆積に対するエッチ比率
- `loss_fraction_of_dep`: 堆積に対する損失比率

## 6. 重要度スコア定義

$$
\mathrm{score}_{\mathrm{sens}}(p)=\mathrm{mean}_w\left(|S_p|\right)
$$

$$
\mathrm{score}_{\mathrm{abl}}(p)=
\frac{\mathrm{mean}_w\left(|h_{\mathrm{base}}-h_{p,\mathrm{off}}|\right)}
{\mathrm{mean}_w\left(|h_{\mathrm{base}}|+\varepsilon\right)}
$$

$$
\mathrm{score}_{\mathrm{total}}=0.5\,\mathrm{score}_{\mathrm{sens}}+0.5\,\mathrm{score}_{\mathrm{abl}}
$$

ここで重み `w` は `area_weights_mm2` と `edge_mask` を使用。

## 7. PDF変換時にLaTeX数式を表示する方法

本ファイルは `$...$`, `$$...$$` で数式記述済み。Pandoc利用時の例:

1. LaTeXエンジンで直接PDF化
   - `pandoc benchmark_cvd.md -o benchmark_cvd.pdf --pdf-engine=xelatex`
2. HTML経由（MathJax）
   - `pandoc benchmark_cvd.md -o benchmark_cvd.html --mathjax`

※ 環境依存で表示が崩れる場合は、Pandocの数式拡張（`tex_math_dollars`）が有効か確認すること。
