# 参照面から表面までの輸送方針

## 輸送収支

役割 \(j\in\{A,B\}\) に対する局所膜輸送収支は

\[
J_j=k_{m,j}(C_{j,\mathrm{ref}}-C_{j,s})
\]

である。反応・状態モデルは壁面需要を与え、\(C_{j,s}\) を解く。輸送供給方式は
\(k_{m,j}\) を供給し、濃度位置を記録する。反応式がCFD場の意味を暗黙に書き換えないよう、
両者の責務を分離する。

## 現在の供給方式

以下は動的状態モデルの輸送閉包で使う。定常CSV網羅評価では、
`transport_capacity_flux` を正規化した `direct_flux` 反応ドライバーとして選ぶことも
できる。その場合、コードは \(k_m\) や \(C_s\) を推定せず、与えられた空間供給場に膜応答
を当てはめる。両者は同じ入力場を利用できるが、答える問いは異なる。

| `km_source` | 必須入力 | 意味 | 主な限界 |
| --- | --- | --- | --- |
| `direct_surface` | 壁面／表面濃度 | 与えられた \(C_s\) を使い、内部の局所膜抵抗をゼロへ近づける | 輸送による濃度低下は推定しない |
| `fit_scalar` | スカラーまたは場の `km_A`、`km_B` | 指定またはフィットした独立膜係数を使う | \(k_m\) が反応誤差や形状誤差を吸収し得る |
| `from_cfd_flux_sink` | `flux_sink`、参照濃度、定義済み境界濃度 | 空間分布をもつ輸送容量係数を推定する | 容量フラックスとしての意味、単位、符号が必要 |

CFD容量フラックスに対しては

\[
k_{m,j}^{\mathrm{CFD}}=
\frac{J_{j,\mathrm{cap}}}
{C_{j,\mathrm{ref}}-C_{j,\mathrm{boundary}}},
\qquad
k_{m,j}=\operatorname{clip}
\left(\gamma_j k_{m,j}^{\mathrm{CFD}},k_{m,\min},k_{m,\max}\right)
\]

を使う。`flux_semantics` は `transport_capacity` とする。
`flux_negative_policy` は `error`、`clip_to_zero`、`allow` のいずれかであり、
本番の既定値は `error` とする。濃度駆動力が非正であるにもかかわらず正のフラックスが
与えられた場合はエラーとする。

## \(k_m\) の補助式

物質移動レジストリには、静止膜に対する

\[
k_m=\frac{D_{\mathrm{eff}}}{\delta_{\mathrm{eff}}}
\]

と、Levich型回転円板に対する

\[
k_m=C_kD_{\mathrm{eff}}^{2/3}\omega^{1/2}\nu^{-1/6}
\]

を実装している。後者では \(\omega=0\) の方針を明示し、エラーまたは設定済み静止膜への
代替処理を選ぶ。

分子拡散抵抗とKnudsen拡散抵抗を組み合わせる場合は

\[
\frac{1}{D_{\mathrm{eff}}}=\frac{1}{D_m}+\frac{1}{D_K}
\]

とする。これらの補助式は \(k_m\) を計算するだけであり、役割モデルの処理系は前節の
3 供給方式のいずれかを使用する。

## 診断量

動的CVD・ALD出力には、条件がそろう場合に次を含める。

- `CsA_over_CrefA`、`CsB_over_CrefB`
- `J_A_surface`、`J_B_surface`
- `J_A_transport`、`J_B_transport`
- CFDから求めた \(k_m\) と実際に使用した \(k_m\) の場
- 境界濃度と濃度駆動力
- 互換性のある観測フラックスがある場合のフラックス閉包残差

輸送滞留時間の指標は秒単位で

\[
\tau_{j,s}=\frac{z_{\mathrm{ref}}\times10^{-3}}{k_{m,j}}
\]

とする。ここで \(z_{\mathrm{ref}}\) はmm、\(k_m\) はm s\(^{-1}\) で設定する。出力名は
`tau_A_s`、`tau_B_s`、マップ診断名は `tau_A_s_map`、`tau_B_s_map` とする。

ALDの被覆蓄積と転化は、サイト密度 \(\Gamma_s\) を掛けて初めてモルフラックスになる。
したがって
\(J_{A,s}=\Gamma_s(r_{\mathrm{store},A}-r_{\mathrm{release},A})\)、
\(J_{B,s}=\Gamma_s\nu_Br_{\mathrm{conv}}\) である。絶対フラックスの解釈には、物理的に
校正したサイト密度が必要である。

CVD AIBモデルにおけるB輸送競合の指標は

\[
\phi_B=
\frac{\Gamma_s\nu_B k_{\mathrm{rxn}}
\theta_A^{p_A}\theta_*^{p_*}}
{C_{B,\mathrm{scale}}k_{m,B}}
\]

である。\(\phi_B\ll1\) はスカラー膜モデルでの局所濃度低下が小さいことを示し、大きな
\(\phi_B\) は輸送需要が膜供給と同程度以上であることを示す。これは縮約したDamköhler型
診断量であり、反応器全体の輸送数ではない。

## 現在の限界

- 現行供給方式はAとBに独立なスカラー膜則を適用する。
- Stefan流、交差拡散、熱拡散、圧力拡散、組成依存の多成分連成は未実装である。
- 回転円板式は相関式であり、反応器形状と流動領域に対する妥当性確認が必要である。
- 現行定常CSV解析の `bulk_as_surface` は壁面変換も絶対フラックス計算も行わない。

化学種が希薄でない場合、正味モル流束が大きい場合、交差拡散が壁面組成を変える場合には
Maxwell–Stefan輸送が必要になる。実装には二成分拡散係数、全組成、温度、圧力、壁面境界
条件、整合したモル平均速度またはフラックス規約が必要である。参考文献は
[THEORY.md](THEORY.md)、実装を再開する証拠条件は [GAPS.md](GAPS.md) を参照する。
