# 反応役割方程式と物理的解釈

## 目的と主張できる範囲

本モデル群は、Fluentの化学種場を、検証可能な少数の表面反応役割へ変換する。役割 \(A\) は、その到達または表面貯蔵が成膜に関係する化学種、\(B\) は気相または吸着状態で変換・再生に関わる相手、\(I\) は競争阻害種を表す。これらの記号は生の化学種列に割り当てる仮説であり、化学種の同定名ではない。

定常方程式は観測可能な縮約モデルである。当てはめパラメータが表すのは、与えた条件下における成膜速度応答の形状と尺度である。表面濃度、サイト密度、量論、温度依存性、仮定した素過程が独立に確立された場合に限り、素反応速度定数として解釈できる。

## 記号と単位

| 記号 | 意味 | 実装上の単位 |
| --- | --- | --- |
| \(C_{j,\mathrm{ref}}\) | 指定した参照面におけるFluent濃度 | kmol m\(^{-3}\) |
| \(C_{j,s}\) | 反応壁面に隣接する濃度 | kmol m\(^{-3}\) |
| \(k_{m,j}\) | 膜物質移動係数 | m s\(^{-1}\) |
| \(J_j\) | 壁面法線方向モルフラックス | kmol m\(^{-2}\) s\(^{-1}\) |
| \(X_j\) | 明示的に選んだ定常反応駆動量：\(C_{j,\mathrm{ref}}\)、\(C_{j,s}\)、または独立な \(J_{j,\mathrm{cap}}\) | 入力modeが宣言する濃度またはフラックス単位 |
| \(X_{j,0}\) | 同定データから推定する正の基準駆動量 | \(X_j\) と同じ |
| \(u_j=X_j/X_{j,0}\) | 正規化した定常反応駆動量 | 1 |
| \(\theta_j\) | モデル化したサイトまたは容量プールの占有率 | 1 |
| \(\theta_*\) | 空サイト率 | 1 |
| \(\chi\) | Mars–van Krevelen型容量プールの酸化状態率 | 1 |
| \(\Gamma_s\) | 表面サイトまたは酸化還元容量密度 | kmol m\(^{-2}\) |
| \(r\) | モデル化したサイト・容量当たりの事象速度 | s\(^{-1}\) |
| \(v\) | 成膜速度 | nm s\(^{-1}\) |
| \(h\) | 膜厚 | nm |
| \(R\) | 定常縮約式でプロファイル消去する成膜速度尺度 | nm s\(^{-1}\) |

現在のCSV網羅比較に含まれる濃度は、バルクまたは参照面濃度である。同じ定常式に、`direct_surface` で与えた壁面濃度、または `direct_flux` で独立に計算したウェハー供給フラックスを入力できる。化学モデルを列挙する前に、一つの入力modeへ固定する。正規化後の代数形が同じでも、フラックス応答群と濃度応答群は異なる物理解釈を持つ。

## モデル一覧

### 表面反応・状態モデル

| 実装モデル | 中心となる応答 | 物理的な問いと仮定する反応 | 適した観測量と用途 | 利点 | 限界 | 主な文献 |
| --- | --- | --- | --- | --- | --- | --- |
| 定数基準 | \(v=R\) | 全マップを条件に依存しない一速度で説明できるか。化学種の役割は仮定しない。 | すべての定常比較に必要な基準。 | 複雑なモデルが、定数でも説明できる傾向によって評価されることを防ぐ。 | 化学的意味も空間応答も持たない。 | 統計基準であり、速度論的帰属はない。 |
| 全濃度補助基準 | \(v=R(C_{\mathrm{tot}}/C_{\mathrm{tot},0})^n\) | 化学種を選ばず、共通濃度尺度だけで条件間転移を説明できるか。 | 全濃度が変化する定常濃度・速度データ。定数基準と併記する。 | 希釈または圧力尺度だけによる改善を検出し、反応役割への誤帰属を防ぐ。 | 化学種・機構の意味を持たず、指数が複数の操作変化を吸収し得る。 | 経験的な次元解析基準であり、速度論的帰属はない。 |
| 単一 \(A\) / \(AI\) 飽和 | \(v=R u_A/[u_A+\lambda(1+\kappa u_I)]\) | 一つの吸着・貯蔵種が飽和応答を与え、阻害種が任意に抑制するか。 | \(A\) を低応答域から飽和域まで独立に変えた定常CVD。\(AI\) 形では \(I\) の変動も必要。 | 最小の解釈可能な飽和モデルであり、阻害なしの厳密縮約を持つ。 | 必須共反応物を表せず、吸着と別の飽和律速を識別できない。 | Langmuir [1]。 |
| 逐次 `aib_qss` | \(v=R u_A b u_B/[u_A+(\delta+b u_B)(1+\kappa u_I)]\) | \(A\) が表面を占有し、気相 \(B\) が吸着 \(A\) を変換し、\(I\) が空サイトプールを阻害するLangmuir–Rideal型仮説。 | \(A/B\) 条件が低 \(B\) 域を含み、応答が準定常である場合の既定CVD比較。 | 飽和、逐次依存、任意の阻害、有限損失の厳密縮約を少数パラメータで表す。 | 阻害のない定常応答は、再パラメータ化後に二つの生の化学種を交換しても不変。当てはめ群は素反応定数ではない。 | Langmuir [1]、Eley and Rideal [2]。 |
| 並列 `parallel_a_ab_qss` | \(v=R u_A(c+b u_B)/[u_A+(\delta+c+b u_B)(1+\kappa u_I)]\) | 吸着 \(A\) が、\(A\) 単独経路と気相 \(B\) を伴う追加経路の二つで変換される。 | \(B\to0\) でも成膜が残り得て、かつ \(B\) 応答域を含むCVD。 | \(A\) 単独経路と \(A+B\) 経路に割り当てた分率を分離して出力する。 | \(B\) の変化が小さいと \(c\) と \(b u_B\) は交絡する。対応する独立の動的状態実装はない。 | Langmuir [1]、Eley and Rideal [2]。 |
| `langmuir_hinshelwood_qss` | \(v=R(a u_A)(b u_B)/(1+a u_A+b u_B+\kappa u_I)^2\) | \(A\) と \(B\) が一つのサイトプールへ競争吸着し、吸着種同士で反応する。 | 両反応物を低被覆率から飽和まで独立に変えた探索的定常CVD比較、または \(B\) の吸着・保持根拠がある場合。 | 異なる分母形を試し、\(\theta_A,\theta_B,\theta_I\)、空サイトを明示する。 | 吸着パラメータも交換すると \(A/B\) 交換に対称。一つの膜速度マップだけでは共吸着を確立できない。 | Langmuir [1]、標準的Langmuir–Hinshelwood速度論 [3]。 |
| 動的 `role_cvd_aib` | \(d\theta_A/dt=r_{\mathrm{ads}}-r_{\mathrm{des}}-\nu_A r_{\mathrm{event}}\) | 継続的に更新される吸着 \(A\) 状態と任意の \(B\) 補助変換で過渡CVD挙動を再現できるか。 | 時間分解したFluent濃度履歴と膜厚または速度観測。 | 表面被覆率を独立な \(A/B\) 輸送閉包と結合し、表面・輸送フラックスの整合を示す。 | 一つの貯蔵状態では、複数サイト、再構成、核生成、詳細生成物ネットワークを表せない。 | Langmuir [1]、Eley and Rideal [2]。 |
| 動的 `role_cvd_mvk` | \(d\chi/dt=r_{\mathrm{reg}}-r_{\mathrm{red}}\) | \(A\) が酸化された表面・格子リザーバーを消費し、\(B\) が再生するか。 | A/B切替え、パルスまたはステップ応答。酸化状態観測があることが望ましい。 | リザーバーの履歴効果を表し、還元、再生、緩和時間を分けて出力する。 | 定常では二反応物応答が逐次無損失形に縮退する。定常膜速度だけではリザーバーを同定できない。 | Mars and van Krevelen [4]。 |
| 動的 `role_ald_state` | \(d\theta_A/dt=r_{\mathrm{store}}-r_{\mathrm{release}}-r_{\mathrm{convert}}\)、\(d\theta_I/dt=r_{I,\mathrm{store}}-r_{I,\mathrm{release}}\) | ドーズ、パージ、共反応物曝露が、貯蔵前駆体と阻害状態を介して作用するか。 | 過渡ALDドーズ・パージ・サイクルデータと最終膜厚またはGPC。状態感度を持つ観測で識別性が高まる。 | 化学種名を先に固定した機構を導入せず、自己制限貯蔵、パージ履歴、変換、阻害を表す。 | 有界陽的小刻みステップには小さい時間刻みを要する場合がある。最終膜厚だけでは貯蔵、脱離、変換速度を分けにくい。 | Puurunen [5]、George [6]。 |

出力表の “運用” は、通常の定常比較にその方程式系が参加することを意味する。対応する化学機構が真であるという根拠ではない。Langmuir–Hinshelwood系は探索的位置づけだが、`--models all` に含める。

### 輸送層と正味膜成長層

| 層 | 実装式 | 適切な用途 | 利点 | 限界 |
| --- | --- | --- | --- | --- |
| `direct_surface` | \(C_s\) を直接与える。局所膜閉包では \(k_m\to\infty\) に相当 | Fluentの壁面または壁面近傍濃度が速度論境界を表す場合 | 輸送係数の当てはめを避ける | 整合するフラックス場もなければ絶対フラックスは求まらない |
| 定常 `direct_flux` 入力 | \(u_j=J_{j,\mathrm{cap}}/J_{j,0}\) | Fluentが、当てはめる壁面反応と独立に計算した非負の到達・輸送容量フラックスを与える場合 | 条件固有のウェハー供給分布を直接保持し、濃度からフラックスへの恣意的変換を避ける | 当てはめ群はフラックス応答に条件づけられる。実現反応フラックスを入れると循環論になる |
| `fit_scalar` | \(J=k_m(C_{\mathrm{ref}}-C_s)\)。\(k_m\) はスカラーまたは与えた場 | 参照面濃度と、独立に選択または当てはめる膜係数がある場合 | 単純な結合で、輸送感度の検討に使える | 反応と輸送が交絡し得る。当てはめ \(k_m\) は膜近似に条件づけられる |
| `from_cfd_flux_sink` | \(k_{m,\mathrm{CFD}}=J_{\mathrm{cap}}/(C_{\mathrm{ref}}-C_b)\)、\(k_m=\gamma k_{m,\mathrm{CFD}}\) | 境界条件を明記したCFD輸送容量フラックスがある場合 | CFDの空間輸送構造を保ち、\(\gamma\) で較正できる | 実現反応フラックスを輸送容量として再利用してはならない。単位と符号が必要 |
| 停滞膜補助計算 | \(k_m=D_{\mathrm{eff}}/\delta_{\mathrm{eff}}\) | 拡散係数と有効膜厚が既知 | 透明性の高い限界推定 | \(\delta_{\mathrm{eff}}\) はモデル量。スカラー Fick拡散は多成分連成を省く |
| 回転円板補助計算 | \(k_m=C_kD^{2/3}\omega^{1/2}\nu^{-1/6}\) | 拡散係数・動粘度が既知の層流回転円板尺度 | 回転数を物理的な輸送尺度へ結びつける | 形状・流れの仮定が反応器に合わない場合がある。\(\omega=0\) には明示的代替処理が必要 |
| Bosanquet拡散選択肢 | \(D_{\mathrm{eff}}^{-1}=D_m^{-1}+D_K^{-1}\) | 分子拡散抵抗とKnudsen抵抗が直列に働く場合 | 細孔輸送の有用な縮約推定 | Maxwell–Stefan多成分壁面モデルを代替しない |
| `deposition_only` | \(v_{\mathrm{net}}=v_{\mathrm{dep}}\) | エッチングまたは損失経路の観測がない場合 | 符号規約が明確 | 正味除去を説明できない |
| `dep_etch_loss` | \(v_{\mathrm{net}}=v_{\mathrm{dep}}-v_{\mathrm{etch}}-v_{\mathrm{loss}}\) | 独立したエッチング・損失速度、または根拠のある分率がある場合 | 膜収支を表面機構選択から分離する | 独立観測のない分率は帳尻合わせであり、同定した経路ではない |

輸送補助計算は候補 \(k_m\) 場を計算する。実際の反応役割処理系は `direct_surface`、`fit_scalar`、`from_cfd_flux_sink` を受け付ける。完全なMaxwell–Stefan拡散とStefan流の結合は未実装であり、希薄な独立膜輸送が不十分な場合の既知の拡張課題である [7,8]。

\(B\) を消費するAIB事象について、実装した輸送要求比は

\[
\phi_B=
\frac{\Gamma_s\nu_B k_{\mathrm{rxn}}
\theta_A^{p_A}\theta_*^{p_*}}
{C_{B,\mathrm{scale}}k_{m,B}},
\qquad
\frac{C_{B,s}}{C_{B,\mathrm{ref}}}=\frac{1}{1+\phi_B}
\]

である。\(\phi_B\ll1\) はスカラー膜をまたぐ枯渇が小さいことを、\(\phi_B\gg1\) はこの閉包内で輸送要求が大きいことを示す。入口供給のうち反応した分率ではなく、局所の無次元収支である。阻害種の利用可能率

\[
f_I=\frac{1}{1+K_I C_{I,\mathrm{ref}}}
\]

は、AIBモデルが仮定する空サイト抑制を表す。表面フラックスと輸送フラックスは分けて出力する。

\[
J_{j,\mathrm{transport}}=k_{m,j}(C_{j,\mathrm{ref}}-C_{j,s}),
\qquad
J_{j,\mathrm{surface}}=\Gamma_s\nu_j r_j.
\]

両者の一致は局所閉包の検査になる。現在の定常CSV経路には較正済み \(k_m\)、サイト密度、壁面濃度がないため、どちらも推定できない。

## 定常の観測可能な縮約式

### 正規化と振幅のプロファイル消去

各化学種 \(j\) について、同定集合から明示的に選んだ局所駆動量の基準を

\[
X_{j,0}=\operatorname{median}_{n\in\mathcal T} X_{j,n},
\qquad u_{j,n}=\frac{X_{j,n}}{X_{j,0}}
\]

と定める。この正規化により、非線形形状パラメータから入力尺度の任意性を除く。濃度入力では \(X=C\)、独立なウェハー供給フラックスでは \(X=J_{\mathrm{cap}}\) である。定常候補は

\[
\hat v_n=R f(\mathbf u_n;\boldsymbol\phi)
\]

と書ける。\(R\ge0\) はnm s\(^{-1}\) の単位を持ち、\(\boldsymbol\phi\) は正の無次元形状パラメータである。\(\boldsymbol\phi\) を固定したとき、重み付き最小二乗の最適値を解析的に求める。

\[
R^*(\boldsymbol\phi)=\max\left[
0,\frac{\sum_n w_n f_n v_n}{\sum_n w_n f_n^2}
\right].
\]

\(R\) のプロファイル消去により非線形探索を一次元減らし、条件付き最適値を厳密に得る。一方で \(R\) は、独立に測定していないサイト密度、膜変換係数、速度定数尺度を吸収する。

二つの補助応答により、役割割当ての価値に下限を置く。定数基準は \(f=1\) である。全濃度基準は

\[
f_{\mathrm{tot}}=
\left(\frac{C_{\mathrm{tot}}}{C_{\mathrm{tot},0}}\right)^n,
\qquad 0.01\le n\le10
\]

とする。全濃度一定の組成変化には不変なので、役割方程式がこの基準より改善した部分は、共通の全濃度傾向だけでは説明できない。指数は経験的補助パラメータであり、素過程の反応次数として報告しない。

### 単一化学種の飽和と競争阻害

最小の吸着 \(A\) 収支は

\[
\frac{d\theta_A}{dt}=k_{\mathrm{ads}}C_{A,s}\theta_*
-k_{\mathrm{loss}}\theta_A,
\qquad
\theta_*+\theta_A+\theta_I=1
\]

である。阻害種が速やかに平衡化すると仮定すると、

\[
\theta_I=K_I C_{I,s}\theta_*,
\qquad
\theta_*=\frac{1-\theta_A}{1+K_I C_{I,s}}.
\]

\(d\theta_A/dt=0\) とし、成膜速度が \(\theta_A\) に比例すると、

\[
v=R\frac{u_A}{u_A+\lambda(1+\kappa u_I)}
\]

へ縮約される。\(\lambda\) は有効半飽和・損失比、\(\kappa\) は阻害被覆率の尺度である。\(\kappa=0\) が単一 \(A\) の厳密縮約となる。阻害効果は、親モデルが条件再当てはめ全体で阻害なし縮約より改善する場合だけ採用する。

### 逐次AIB準定常モデル

一次吸着と、吸着 \(A\)・気相 \(B\) 間の事象について、

\[
\begin{aligned}
r_{\mathrm{ads}} &= k_{\mathrm{ads}}C_{A,s}\theta_*,\\
r_{\mathrm{des}} &= k_{\mathrm{des}}\theta_A,\\
r_{AB} &= k_{\mathrm{rxn}}\theta_A\frac{C_{B,s}}{C_{B,\mathrm{scale}}},\\
\frac{d\theta_A}{dt} &= r_{\mathrm{ads}}-r_{\mathrm{des}}-\nu_A r_{AB}.
\end{aligned}
\]

上の阻害種関係を使い、被覆率を準定常とすると、

\[
\theta_A=
\frac{k_{\mathrm{ads}}C_{A,s}}
{k_{\mathrm{ads}}C_{A,s}+
\left(k_{\mathrm{des}}+\nu_A k_{\mathrm{rxn}}C_{B,s}/C_{B,\mathrm{scale}}\right)
(1+K_I C_{I,s})}
\]

を得る。実行する簡潔な形は

\[
v=R\frac{u_A b u_B}
{u_A+(\delta+b u_B)(1+\kappa u_I)}
\]

であり、この式を実装上の定義とする。無次元群のおおよその対応は

\[
\delta\sim\frac{k_{\mathrm{des}}}{k_{\mathrm{ads}}C_{A,0}},\qquad
b\sim\frac{\nu_A k_{\mathrm{rxn}}C_{B,0}}
{k_{\mathrm{ads}}C_{A,0}C_{B,\mathrm{scale}}},\qquad
\kappa\sim K_I C_{I,0}
\]

で、残る尺度を \(R\) が吸収する。定常当てはめ処理は \(R,\delta,b,\kappa\) を直接推定し、\(\Gamma_s,\alpha_h,\nu_A\) や表面濃度を知らないため、この対応は条件付きである。

`no_desorption` 縮約では \(\delta=0\) とする。報告書では、有限な非生産損失群を除いたモデルと記述する。性能差だけから、その損失を物理的な脱離と同定することはできない。不可逆損失、失活、欠落した経路でも同じ定常効果を生じ得る。

阻害のないAB形では、\(u_A\) と \(u_B\) の交換に尺度・パラメータ変換を組み合わせると応答族が変わらない。そのため、阻害種、過渡状態、外部化学情報のいずれかが対称性を破らない限り、コードはこの組を無向として報告する。

### 並列AおよびA+Bモデル

吸着 \(A\) が二経路で変換されるとする。

\[
r_A=k_A\theta_A,\qquad
r_{AB}=k_{AB}\theta_A C_{B,s}/C_{B,\mathrm{scale}}.
\]

正規化した準定常応答は

\[
v=R\frac{u_A(c+b u_B)}
{u_A+(\delta+c+b u_B)(1+\kappa u_I)}
\]

である。コードが出力する経路分率は

\[
f_A=\frac{c}{c+b u_B},\qquad
f_{AB}=\frac{b u_B}{c+b u_B},\qquad f_A+f_{AB}=1
\]

となる。厳密縮約は、適用可能な場合に \(\delta=0\)、\(c=0\)、\(B\) の除去、\(I\) の除去を試験する。\(A\) 単独経路の根拠には \(u_B=0\) 近傍のデータが必要である。そうでなければ、\(c\) と \(b u_B\) の効果がほぼ同じになり、信頼して分離できない。

### 二吸着種Langmuir–Hinshelwoodモデル

一様な一サイトプールへの競争吸着を仮定する。

\[
\theta_A=K_A C_{A,s}\theta_*,\quad
\theta_B=K_B C_{B,s}\theta_*,\quad
\theta_I=K_I C_{I,s}\theta_*.
\]

サイト収支から

\[
\theta_*=\frac{1}{1+K_A C_{A,s}+K_B C_{B,s}+K_I C_{I,s}}
\]

となる。二分子表面反応 \(r_{AB}=k\theta_A\theta_B\) を正規化すると、

\[
v=R\frac{(a u_A)(b u_B)}
{(1+a u_A+b u_B+\kappa u_I)^2}
\]

を得る。分母の二乗は、同じ空サイト分母を持つ二被覆率を掛けることから直接生じる。本モデルは、吸着平衡、一種類のサイト、横方向相互作用なし、表面反応律速を仮定する。吸着量論や複数サイトプールが異なれば、指数と分母も変わる。したがって、実装モデルが示すのはこの応答形に対する根拠であり、微視的Langmuir–Hinshelwood機構の証明ではない。

## 動的CVDモデル

### 輸送閉包を持つAIB被覆率モデル

動的CVDモデルは \(\theta_A\) を状態として保持し、

\[
\frac{d\theta_A}{dt}=
k_{\mathrm{ads}}C_{A,s}\theta_*^{m}
-k_{\mathrm{des}}\theta_A
-\nu_A k_{\mathrm{rxn}}\theta_A^{p_A}\theta_*^{p_*}
\left(\frac{C_{B,s}}{C_{B,\mathrm{scale}}}\right)^{\mathbb 1_B}
\]

を積分する。ただし、

\[
\theta_*=\frac{1-\theta_A}{1+K_I C_{I,\mathrm{ref}}},\qquad
\frac{dh}{dt}=\alpha_h\Gamma_s r_{\mathrm{event}}.
\]

局所膜収支は代数的に

\[
C_{A,s}=
\frac{C_{A,\mathrm{ref}}+(\Gamma_s k_{\mathrm{des}}\theta_A)/k_{m,A}}
{1+(\Gamma_s k_{\mathrm{ads}}\theta_*^m)/k_{m,A}},
\]

\[
C_{B,s}=
\frac{C_{B,\mathrm{ref}}}
{1+\Gamma_s\nu_B k_{\mathrm{rxn}}\theta_A^{p_A}\theta_*^{p_*}/
(C_{B,\mathrm{scale}}k_{m,B})}
\]

として解く。これらは、モデル化した正味壁面要求と \(k_m(C_{\mathrm{ref}}-C_s)\) を整合させる。多成分Stefan–Maxwell境界層は実装していない。

### Mars–van Krevelen酸化還元リザーバー

状態 \(\chi\) は、\(A\) による還元に利用できるモデル化酸化還元容量の分率である。

\[
r_{\mathrm{red}}=k_{\mathrm{red}}C_{A,s}\chi,\qquad
r_{\mathrm{reg}}=k_{\mathrm{reg}}C_{B,s}(1-\chi),
\]

\[
\frac{d\chi}{dt}=r_{\mathrm{reg}}-r_{\mathrm{red}},\qquad
\frac{dh}{dt}=\alpha_h\Gamma_s r_{\mathrm{red}}.
\]

独立な膜閉包は

\[
C_{A,s}=\frac{C_{A,\mathrm{ref}}}
{1+\Gamma_s k_{\mathrm{red}}\chi/k_{m,A}},
\qquad
C_{B,s}=\frac{C_{B,\mathrm{ref}}}
{1+\Gamma_s\nu_B k_{\mathrm{reg}}(1-\chi)/k_{m,B}}
\]

である。表面濃度を固定して速度論的定常状態をとると、

\[
r_{\mathrm{red}}=r_{\mathrm{reg}}
=\frac{k_{\mathrm{red}}C_{A,s}\,k_{\mathrm{reg}}C_{B,s}}
{k_{\mathrm{red}}C_{A,s}+k_{\mathrm{reg}}C_{B,s}}
\]

となる。正規化後は \(\delta=0\) の逐次ABと同じ関数族なので、定常網羅比較では追加の一票を与えず、一つの代表モデルとして扱う。MvKを識別するには、

\[
\tau_{\mathrm{redox}}=
\left(k_{\mathrm{red}}C_{A,s}+k_{\mathrm{reg}}C_{B,s}\right)^{-1}
\]

で表されるリザーバー履歴が応答を変える過渡条件、または \(\chi\) の独立測定が必要である。

## 動的ALD貯蔵モデル

ALDモデルは

\[
\theta_*=1-\theta_A-\theta_I,
\]

\[
\frac{d\theta_A}{dt}=
k_{\mathrm{store},A}C_{A,s}\theta_*
-k_{\mathrm{release},A}\theta_A-r_{\mathrm{conv}},
\]

\[
\frac{d\theta_I}{dt}=
k_{\mathrm{store},I}C_{I,s}\theta_*
-k_{\mathrm{release},I}\theta_I
\]

を用いる。変換経路は \(B\) の有無で選ぶ。

\[
r_{\mathrm{conv}}=
\begin{cases}
k_{\mathrm{convert},A}\theta_A, & B\text{ なし},\\
k_{\mathrm{convert},AB}C_{B,s}\theta_A, & B\text{ あり}.
\end{cases}
\qquad
\frac{dh}{dt}=\alpha_h r_{\mathrm{conv}}.
\]

ここで \(r_{\mathrm{conv}}\) は被覆率 s\(^{-1}\)、\(\alpha_h\) は変換被覆率当たりのnmを単位とする。サイト密度 \(\Gamma_s\,[\mathrm{kmol\,m^{-2}}]\) を用いると、絶対的な役割フラックスは

\[
J_{A,s}=\Gamma_s\left(
k_{\mathrm{store},A}C_{A,s}\theta_*
-k_{\mathrm{release},A}\theta_A\right),
\qquad
J_{B,s}=\Gamma_s\nu_B r_{\mathrm{conv}}
\]

となる。貯蔵と脱離は、変換を二重計上せずに \(A\) の膜閉包へ入る。

\[
C_{A,s}=
\frac{k_{m,A}C_{A,\mathrm{ref}}+
\Gamma_s k_{\mathrm{release},A}\theta_A}
{k_{m,A}+\Gamma_s k_{\mathrm{store},A}\theta_*}.
\]

\(B\) の吸込みは

\[
C_{B,s}=
\frac{k_{m,B}C_{B,\mathrm{ref}}}
{k_{m,B}+\Gamma_s\nu_Bk_{\mathrm{convert},AB}\theta_A}
\]

を与える。これらは \(k_m(C_{\mathrm{ref}}-C_s)=J_s\) を \(\mathrm{kmol\,m^{-2}\,s^{-1}}\) で満たす。明示的な \(\Gamma_s\) により、被覆率速度をモル輸送フラックスと直接比較する誤りを防ぐ。正規化検討では \(\Gamma_s=1\) を許容するが、その場合のフラックス絶対値は正規化値である。

これは反応役割同化のための最小潜在状態モデルである。貯蔵、脱離、変換を記述するが、固有名を持つ配位子交換系列や特定の表面終端を主張しない。

## 推定・識別・可視化に用いる量

### ウェハー全体の損失関数

条件 \(k\) が観測 \(y_{ki}\)、予測 \(\hat y_{ki}\)、非負の点重み \(q_{ki}\) を含むとする。実装では、まず各ウェハー内で

\[
w_{ki}=\frac{q_{ki}}{\sum_iq_{ki}},
\qquad \sum_iw_{ki}=1
\]

と重みを正規化し、その後で条件損失関数を平均する。したがって、マップの点数が異なっても、同定に使う各ウェハーは一票ずつ持つ。当てはめ係数は同定ウェハー全体で共通であり、ウェハーごとに独立した運動論式を当てはめない。

| CLI名 | 条件損失関数 \(L_k\) | 用途 | 主な限界 |
| --- | --- | --- | --- |
| `mse` | \(\sum_iw_{ki}(\hat y_{ki}-y_{ki})^2\) | 次元を持つ線形速度の当てはめ。成膜速度絶対誤差の物理的費用を保つ | 高成膜速度条件が数値尺度を支配し得る |
| `wafer_normalized_mse` | \(\sum_iw_{ki}(\hat y_{ki}-y_{ki})^2/s_k^2\)、\(s_k^2=\sum_iw_{ki}y_{ki}^2\) | 低速・高速ウェハーに同程度の相対的影響を持たせる | 同じ比率誤差を同じ費用とし、nm\(^2\) s\(^{-2}\) の単位を失う |
| `wafer_normalized_mae` | \(\sum_iw_{ki}|\hat y_{ki}-y_{ki}|/s_k\) | 相対尺度で、孤立残差の影響を抑える | 0で微分不能で、系統的な小残差構造を軽視し得る |
| `symmetric_normalized_mse` | \(2\sum_iw_{ki}(\hat y_{ki}-y_{ki})^2/\sum_iw_{ki}(y_{ki}^2+\hat y_{ki}^2)\) | 観測値・予測値の一方だけを分母尺度にしたくない場合 | 尺度が予測に依存し、物理的な直接性が弱い |

全目的関数は \(L=K^{-1}\sum_k L_k\) である。任意の半径方向不確かさモデルは、宣言した中心対エッジの標準不確かさ比を通して点重みに掛かる。不確かさまたは反復分散が根拠を与えるときだけ、根拠に基づく重みづけになる。それ以外は感度解析として扱う。異なる損失関数定義の目的関数値を直接比較しない。候補と最適化器の比較は、次元を持つ条件交差検証RMSEとホールドアウトRMSEへ戻して行う。

### 予測指標とウェハー形状指標

\(N\) 点の一つのホールドアウトウェハーについて、残差を \(e_i=\hat y_i-y_i\)、観測平均を \(\bar y\)、予測平均を \(\bar{\hat y}\) とする。通常の速度指標は

\[
\operatorname{RMSE}=\sqrt{\frac1N\sum_i e_i^2},\qquad
\operatorname{bias}=\frac1N\sum_i e_i,
\qquad
\operatorname{relative\ RMSE}=\frac{\operatorname{RMSE}}{|\bar y|}
\]

である。条件平均の転移とウェハー形状を、中心化によって分離する。

\[
e_i^{\circ}=(\hat y_i-\bar{\hat y})-(y_i-\bar y),
\]

\[
\operatorname{RMSE}_{\mathrm{centered}}=
\sqrt{\frac1N\sum_i(e_i^{\circ})^2},
\qquad
R^2_{\mathrm{centered}}=
1-\frac{\sum_i(e_i^{\circ})^2}{\sum_i(y_i-\bar y)^2}.
\]

面内中心化 \(R^2\) が負なら、予測分布は、正しい測定平均を全点へ一様に割り当てる予測より悪い。条件平均だけがよく合い、面内振幅または位相が合わない場合、通常RMSEが小さくても負になり得る。空間相関は中心化後の位相一致を、予測・観測範囲比は振幅捕捉を測る。いずれも中心化誤差の代わりにはならない。

### 反応役割の重要度と割当て安定性

割当て役割 \(j\) について、選択モデルの予測を \(\hat y_i\)、その役割の局所入力を同定基準 \(X_{j,0}\) に置き換えた予測を \(\hat y_i^{(-j)}\) とする。コードは二乗差を条件均衡化する。

\[
S_j=\left[
\frac1K\sum_{k=1}^K\frac1{N_k}
\sum_{i\in k}(\hat y_i-\hat y_i^{(-j)})^2
\right]^{1/2}.
\]

これは一度に一役割だけを変える予測感度である。非線形相互作用があるため、\(S_j\) を足しても成膜速度や1にはならない。役割 \(j\) に同じ生の化学種が選ばれた外側条件再当てはめの割合を \(f_j\)、選択モデルの固定ホールドアウトRMSEを \(E\) とする。無次元尺度

\[
Q_j=\frac{S_j}{E}
\]

によって、実務上異なる二種類の非同定を分ける。低い \(f_j\) と \(Q_j\ll1\) の組合せは、不安定だが試験範囲では予測への影響が小さい。低い \(f_j\) と \(Q_j\gtrsim1\) は、影響が大きい未解決割当てである。\(Q_j=1\) は図上の目安であり、普遍的な統計棄却閾値ではない。

### 代替方程式系の予測差

方程式系 \(m\) の最良候補によるホールドアウト予測を \(\hat y_{m,i}\)、同じ座標における選択系予測を \(\hat y_{\star,i}\) とする。モデル条件付き予測分離は

\[
D_m=\sqrt{\frac1N\sum_i
(\hat y_{m,i}-\hat y_{\star,i})^2},
\qquad H_m=\frac{D_m}{E}
\]

である。\(H_m\) が小さければ、当てはめた反応解釈が変わっても試験予測への影響は小さい。大きければ、方程式系の曖昧さが予測リスクにもなる。これは機構確率を与えず、当てはめた観測可能式から一つを選ぶ影響を示す。

### 運動論比の局所感度と部分損失関数断面

正の当てはめ形状パラメータ \(p_j\) について、点 \(i\) における局所対数感度を

\[
g_{ij}=\frac{\partial\ln \hat y_i}{\partial\ln p_j}
\]

とする。実装は

\[
G_j=\sqrt{\frac1N\sum_i g_{ij}^2}
\]

と、中心化した列 \(g_{\cdot j}\)、\(g_{\cdot\ell}\) 間のPearson相関を出力する。小さい \(G_j\) は局所的に不活性な方向を示す。相関絶対値が1に近ければ、二方向が符号と尺度を除いてほぼ同じ空間応答を作るため、個別値の実用的情報が弱い。この微分設計は局所情報の診断であり、それだけで大域的不確かさ区間は得られない [12,14]。

図示する損失関数断面では、一つのパラメータ \(p_j\) を対数格子上で固定し、他の形状パラメータを当てはめ値に保ち、分離可能な非負速度尺度 \(R\) だけを再プロファイルする。

\[
\widetilde L_j(p)=\min_{R\ge0}
L\{R f(\mathbf u;p,\hat{\boldsymbol\phi}_{-j})\}.
\]

広く平坦な断面は、選択式の下で現在の観測がその方向をほとんど制約しない直接的根拠となる。\(\boldsymbol\phi_{-j}\) を再最適化しないため、これは完全なプロファイル尤度ではなく部分断面である。正式な尤度区間には、ノイズモデルと同時再プロファイルが必要である [14]。

### 選択後の空間残差応答

任意の空間段は、化学モデルと係数を固定した後にだけ開始する。\(\rho_i\) を正規化ウェハー半径とし、基底に \(\rho^2\)、または \((\rho^2,\rho^4)\) を用いる。各基底列を同定条件内で中心化して \(\Phi_{ki}\) とする。当てはめ対象は中心化対数残差である。

\[
d_{ki}=\{\ln y_{ki}-\overline{\ln y}_k\}
-\{\ln\hat y^{\mathrm{chem}}_{ki}
-\overline{\ln\hat y^{\mathrm{chem}}}_k\},
\]

\[
\hat{\boldsymbol\beta}
=\arg\min_{\boldsymbol\beta}
\frac1K\sum_k\frac1{N_k}
\sum_{i\in k}(d_{ki}-\Phi_{ki}\boldsymbol\beta)^2.
\]

生の補正係数は \(g_i=\exp(\Phi_i\hat{\boldsymbol\beta})\) である。適用する各ウェハー上で

\[
\hat y_i^{\mathrm{corr}}=
\hat y_i^{\mathrm{chem}}g_i
\frac{\overline{\hat y^{\mathrm{chem}}}}
{\overline{\hat y^{\mathrm{chem}}g}},
\qquad
\overline{\hat y^{\mathrm{corr}}}
=\overline{\hat y^{\mathrm{chem}}}
\]

となるよう再尺度化する。したがって、空間応答は化学モデルの条件平均を修復できず、役割や方程式系の選択も変えられない。係数が表すのは転移可能な半径残差基底である。対応する測定場がなければ、温度、輸送、チャンバー形状、その他の物理原因は同定しない。欠落した空間物理を速度論へ誤帰属しないためにも、モデル不一致を較正パラメータから分離する必要がある [15]。

### 根拠を示す図

| 図の分類 | 主な数値出典 | 図から支持できる結論 |
| --- | --- | --- |
| 最適化収束 | `optimization_history.csv` | 方程式系ごとの最良候補の数値的進展 |
| 方程式比較と反応経路 | `role_ranking.csv`、方程式系登録表 | 条件間転移誤差、選択安定性、仮定した反応構造 |
| 代替モデル間の一致 | `reaction_model_predictions.csv` | 方程式系の曖昧さが予測へ与える影響 |
| 役割の安定性と重要度 | `role_stability.csv`、`role_importance_and_stability.csv` | 割当て不確かさが予測上軽微か重大か |
| 状態・経路分率 | `reaction_model_states.csv`、`reaction_state_summary.csv` | 選択モデルに条件づけた占有率と速度配分 |
| パラメータ感度と損失関数断面 | `parameter_sensitivity_correlations.csv`、`parameter_loss_slices.csv` | 局所的に弱い方向と連成した方向 |
| ホールドアウトマップと半径分布 | `test_predictions.csv` | 平均の転移、空間位相、振幅、残差構造 |
| 空間応答図 | `spatial_response_summary.csv`、`spatial_response_coefficients.csv` | 分離した残差基底の予測上の効果 |

反応経路の矢印、当てはめ分率、空間基底は、宣言した方程式を説明する図である。独立な表面測定ではない。したがって、図の解釈は、データ分割、出典成果物、単位、観測量の意味に従う。

## 近似階層と解釈

実装モデルは異なる階層にあり、整合する観測量なしに一列の順位へ並べてはならない。

1. 定常方程式の網羅比較は、観測可能な膜速度応答形状を比較する。
2. 動的CVD・ALDモデルは、時間分解観測に対して状態履歴を比較する。
3. 輸送閉包は、Fluentで与えた位置を表面へどう結ぶかを決める。
4. 正味膜モデルは、独立に支持された堆積、エッチング、損失速度を結合する。

異なる物理機構が同じ定常代数形を共有することがある。逆に、同じ化学でも、輸送、温度、未観測表面状態が変われば、見かけ上異なる式に従うことがある。小さい交差検証誤差が確立するのは、試験領域内での予測適合性である。機構を採用するには、さらに役割安定性、厳密縮約の根拠、十分な入力変動、機構固有の観測量が必要である。

## 参考文献

1. I. Langmuir, “The Adsorption of Gases on Plane Surfaces of Glass, Mica and Platinum,” *Journal of the American Chemical Society* **40** (1918) 1361–1403. [doi:10.1021/ja02242a004](https://doi.org/10.1021/ja02242a004).
2. D. D. Eley and E. K. Rideal, “The Catalysis of the Parahydrogen Conversion by Tungsten,” *Proceedings of the Royal Society A* **178** (1941) 429–451. [doi:10.1098/rspa.1941.0066](https://doi.org/10.1098/rspa.1941.0066).
3. C. N. Hinshelwood, *The Kinetics of Chemical Change in Gaseous Systems*, Oxford University Press, 1926.
4. P. Mars and D. W. van Krevelen, “Oxidations Carried Out by Means of Vanadium Oxide Catalysts,” *Chemical Engineering Science*, Special Supplement **3** (1954) 41–59. [doi:10.1016/S0009-2509(54)80005-4](https://doi.org/10.1016/S0009-2509(54)80005-4).
5. R. L. Puurunen, “Surface Chemistry of Atomic Layer Deposition: A Case Study for the Trimethylaluminum/Water Process,” *Journal of Applied Physics* **97** (2005) 121301. [doi:10.1063/1.1940727](https://doi.org/10.1063/1.1940727).
6. S. M. George, “Atomic Layer Deposition: An Overview,” *Chemical Reviews* **110** (2010) 111–131. [doi:10.1021/cr900056b](https://doi.org/10.1021/cr900056b).
7. R. Krishna and J. A. Wesselingh, “The Maxwell-Stefan Approach to Mass Transfer,” *Chemical Engineering Science* **52** (1997) 861–911. [doi:10.1016/S0009-2509(96)00458-7](https://doi.org/10.1016/S0009-2509(96)00458-7).
8. J. A. Wesselingh and R. Krishna, “Stefan-Maxwell Mass Transport,” *Chemical Engineering Science* **64** (2009) 4796–4803. [doi:10.1016/j.ces.2009.07.002](https://doi.org/10.1016/j.ces.2009.07.002).
9. V. G. Levich, *Physicochemical Hydrodynamics*, Prentice-Hall, Englewood Cliffs, 1962.
10. E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed., Springer, 1996. [doi:10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7).
11. S. Varma and R. Simon, “Bias in Error Estimation When Using Cross-Validation for Model Selection,” *BMC Bioinformatics* **7** (2006) 91. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91).
12. G. Franceschini and S. Macchietto, “Model-Based Design of Experiments for Parameter Precision: State of the Art,” *Chemical Engineering Science* **63** (2008) 4846–4872. [doi:10.1016/j.ces.2007.11.034](https://doi.org/10.1016/j.ces.2007.11.034).
13. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall/CRC, 1993. [doi:10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9).
14. A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmüller, and J. Timmer, “Structural and Practical Identifiability Analysis of Partially Observed Dynamical Models by Exploiting the Profile Likelihood,” *Bioinformatics* **25** (2009) 1923–1929. [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358).
15. M. C. Kennedy and A. O'Hagan, “Bayesian Calibration of Computer Models,” *Journal of the Royal Statistical Society: Series B* **63** (2001) 425–464. [doi:10.1111/1467-9868.00294](https://doi.org/10.1111/1467-9868.00294).
