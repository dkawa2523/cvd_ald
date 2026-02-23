以下は、**今回の大幅改良（AIB‑ODE 1本化／role縮約 A・I・B／I・B≤1／B次数0/1固定／ODE主役）**と、**本来目的（実測1D/2D膜厚マップに整合する“物理的に妥当な”係数・モデル選択を自動で行う）**に対して、**現場導入が成功しやすい最適化（＝同化＋モデル選択）**を、調査（Optuna/BO/DA）も踏まえて **実装に落ちる粒度**で整理した提案です。

---

# 0) 結論：この改良に最適な“最適化の型”

今回のAIB‑ODEは、最適化問題が **「混合離散＋連続」** で、さらに **物理妥当性（拘束・識別性）** が重要です。
したがって、最も壊れにくく、第三者が理解しやすい “型” は次です。

## 推奨（P0～P1の本命）

**二段（階層）最適化＋多忠実度（coarse→fine）＋プルーニング**：

1. **離散（モデル構造）**：role割当（A/AI/AB/AIB）＋整数次数（pA,p*,m_ads、Bは0/1固定）を列挙
2. **連続（物理パラメータ）**：各候補に対して Optuna で最小化（TPE/CMA‑ES）
3. **高速化**：coarse評価で `trial.report()` → pruner（Median/Hyperband）で早期停止
4. **モデル選択**：データ適合＋複雑さ罰則＋物理診断（phi_B, f_I, solver健全性）でランキング
5. **識別不能を明示**：上位候補が拮抗する場合は「Bは必要だがspeciesは特定不能」等を出力（重要）

Optuna側のサンプラー・プルーナーは、TPEやCMA‑ES、Median/Hyperband 等が公式に提供されます。([docs-ja.optuna.org][1])

---

# 1) 問題定式化：今回のAIB‑ODEで最適化すべき対象

## 1.1 観測（実測）

* 条件 (c\in{1,\dots,C})
* 実測膜厚マップ（2D、ただし1D radialに落とす運用もあり）
  [
  h^{(c)}_{\mathrm{meas}}(x,y)
  ]

## 1.2 入力（Fluent）

* (C_{\mathrm{ref},s}^{(c)}(x,y,t))（species (s)、最大10種程度）
* steadyなら (t) なし、transient/ALDなら (t) あり

## 1.3 モデル構造（離散）

* role割当：Aは必須1種、I/Bは None または 1種（disjoint）、unused許可
* クラス：A / AI / AB / AIB
* 反応次数（整数）：

  * (m_{\mathrm{ads}}\in{0,1,2,\dots})
  * (p_A, p_*\in{0,1,2,\dots})
  * (m_B\in{0,1})（Bの有無で固定）
  * **総次数制約**（例）：(p_A+p_*+m_B \le 3)

## 1.4 連続パラメータ（例）

AIB‑ODEの範囲で “黒箱化しやすいが物理意味がある”ものに限定しておくのが成功確率が高いです。

* 反応・吸着系：(k_{\mathrm{ads}}, k_{\mathrm{des}}, k_{\mathrm{rxn}}, K_I)
* 輸送のまとめ：(k_{m,A})（条件別推奨）、(k_{m,B})（Bがあるときのみ、条件別推奨）
* 変換・スケール：(C_{B,\mathrm{scale}})、(\alpha_h\Gamma_s)（まとめても可）
* 化学量論：(\nu_A)（固定推奨。自由にすると同定崩壊しやすい）

---

# 2) 最適化の全体設計：混合（離散＋連続）を“壊れずに解く”

## 2.1 なぜ「1つのOptuna studyに全部（roleも含めて）入れない」方が良いか

Optunaは `suggest_categorical` でカテゴリを扱えます。([docs-ja.optuna.org][2])
しかし今回の role割当には **disjoint制約（A/I/B重複禁止）** があり、カテゴリをそのまま入れると

* 無効組合せが大量に出る（trialの多くが無駄）
* “無効回避ロジック”がobjective側に散り、第三者が読めなくなる

→ だから **離散は列挙で外出し**し、各候補に対して “連続だけOptuna” が最短で堅牢です。

---

## 2.2 推奨パイプライン（実装に直結）

### Step A：候補生成（離散）

* roles候補：最大 (10\times(1+9)\times(1+8)=910) 程度（制約で減る）
* orders候補：ユーザがYAMLで小さな候補集合を指定（例：10～30個）

**出力**：`CandidateModel(structure_id, roles, orders, class_id, n_params)` のリスト

---

### Step B：coarseスクリーニング（高速に落とす）

候補が多いので、いきなりフル2D＋フル時間で回すと重いです。

coarseでやること（例）：

* 空間を間引き（例：全点→25点）
* 時間を粗く（dt大きめ）
* 目的関数を簡易化（1D radial, あるいはdownsampleした2D）

**この段階で落とすべき候補**

* solver健全性が悪い（bracket_fail多、θクリップ多）
* Iありなのに (f_I\approx 1) しか出ない（実質不要）
* Bありなのに (\phi_B\ll 1) で (C_{s,B}/C_{ref,B}\approx 1) ばかり（寄与薄い）

※これは “排除” というより **「上位Kだけ精密化」**に使うのが安全です。

---

### Step C：精密フィット（連続）＝Optuna

Optunaは

* サンプラー（TPE、CMA‑ES、NSGA‑II等）([optuna.readthedocs.io][3])
* プルーナー（Median、Hyperband等）([optuna.readthedocs.io][4])
  を持ち、途中値 `trial.report()` を使って早期停止できます。([docs-ja.optuna.org][5])

---

### Step D：モデル選択（ランキング）

最終的には “物理妥当性” が必要なので、
**単純な最小損失ではなく**、以下のスコアで比較します。

[
\text{score}
= \underbrace{\mathcal{L}*{\text{data}}}*{\text{実測適合}}

* \lambda_{\text{complex}}\underbrace{#(\text{free params})}_{\text{複雑さ}}
* \lambda_{\text{role}}(\mathbb{1}[I]+\mathbb{1}[B])
* \lambda_{\text{num}}\underbrace{\text{penalty(solver)}}_{\text{数値健全性}}
* \lambda_{\text{prior}}\underbrace{\mathcal{R}(\theta)}_{\text{物理事前}}
  ]

ここで

* (\mathcal{L}_{\text{data}})：HuberやL1（外れ・測定ノイズに強い）
* `penalty(solver)`：bracket_fail, clip率, NaN発生など
* (\mathcal{R})：パラメータの事前（例：log正則化、範囲逸脱を罰する）

---

# 3) 連続最適化：Optunaで何を使うべきか（TPE vs CMA‑ES vs それ以外）

## 3.1 基本：TPESampler（最初のデフォルト）

* Optunaの代表的サンプラーで、条件分岐を含む探索空間に強い（define‑by‑run）。([optuna.readthedocs.io][6])
* あなたのケースでは “候補ごとに連続空間が変わる（I/B有無でパラメータ次元が変わる）” ので、扱いやすい

**推奨理由**

* まず “確実に回る”
* パラメータが 6～15 次元程度なら十分戦える
* logスケール（正値パラメータ）に素直に対応可能([docs-ja.optuna.org][2])

---

## 3.2 反応係数が連続で滑らかなら：CmaEsSampler（候補が絞れた後に強い）

CMA‑ESは連続最適化で強力で、Optunaにも `CmaEsSampler` として提供されます。([optuna.readthedocs.io][7])

**向く条件**

* 目的関数が比較的滑らか（AIB‑ODEは暗黙解なので滑らか寄りになりやすい）
* 次元は中程度（≲20くらいまでが現実的）
* 既に「rolesとordersはほぼ決まった」状態での詰め

**使い方（おすすめ）**

* まずTPEで荒く当てる → 上位候補でCMA‑ESに切替、が成功率が高い

---

## 3.3 多目的（精度 vs 物理健全性 vs 複雑さ）を“本当に”やるなら：NSGA‑II

Optunaは多目的サンプラーとして `NSGAIISampler` を持ちます。([optuna.readthedocs.io][8])
ただし、Optunaのprunerは単一目的前提という注意が明記されています。([docs-ja.optuna.org][5])

**結論（現場導入向け）**

* P0/P1では **単一目的（重み付き和）**のままにしておくのが簡単で堅牢
* NSGA‑IIはP2で「Pareto frontが必要」になったら導入

---

## 3.4 “サンプル効率”をさらに上げたい（試行回数が高い）なら：BoTorchSampler（オプション）

Optuna Integration / OptunaHub には BoTorchSampler（GP‑based BO）が提供されていますが、追加依存が増えます。([OptunaHub][9])
BoTorch側は制約付き・多目的BOも強力です。([botorch.org][10])

**このプロジェクトでの位置づけ**

* “候補が絞れた後の高価な精密評価” に向く
* ただし依存追加・GPU/PyTorch環境が絡むので、**P2以降の拡張**が現実的

---

# 4) 高速化の核：プルーニング＋多忠実度（coarse→fine）の設計

Optunaのプルーニングは、中間値を `trial.report(value, step)` で渡し、`trial.should_prune()` で止めます。([docs-ja.optuna.org][5])
プルーナーには MedianPruner / HyperbandPruner などがあります。([optuna.readthedocs.io][4])

## 4.1 あなたの問題に“そのまま適用できる”多忠実度ステップ

「step＝忠実度段階」にして報告するのが最も実装が簡単です。

例（1 trial の中で段階的に評価）：

* step 0：空間25点＋粗dt（超高速）
* step 1：空間100点＋中dt
* step 2：全点（最大数百）＋本dt＋（必要なら）全条件

各段階で

* `loss_data`
* `penalty_num`
* `penalty_phys`
  を合算して中間スコアをreportする

→ 悪いtrialは step 0/1 で落ちる。

---

## 4.2 Hyperbandを使う意味

Hyperbandは限られた予算を複数設定に配分する考え方で、Optunaで `HyperbandPruner` が使えます。([docs-ja.optuna.org][11])
今回の “忠実度（点数・dt・条件数）＝予算” はHyperbandと相性が良いです。

---

# 5) 複数条件（multi-condition）同化：何が難しく、どう設計すると壊れにくいか

## 5.1 難しさ

* 条件ごとに流れ・温度が違うため、**kmなどは条件依存が強い**
* 一方で反応速度定数を条件ごとに自由にすると **ただ当てるだけの黒箱**になりやすい

## 5.2 推奨：global / per_condition を明示し、階層正則化で暴走を抑える

（実装の複雑さを増やさず、物理妥当性も守りやすい）

* **global推奨**：(k_{ads}, k_{des}, k_{rxn}, K_I, C_{B,scale}, \alpha_h\Gamma_s)
* **per_condition推奨**：(k_{m,A}^{(c)}, k_{m,B}^{(c)})（Bありのとき）

  * ここが装置条件（流量・圧力・回転など）の吸収ノブになりやすい

さらに “per_condition の自由度を抑える” ために
[
k_{m,A}^{(c)} = \bar{k}_{m,A}\exp(\delta^{(c)}),\quad
\delta^{(c)}\sim \mathcal{N}(0,\sigma^2)
]
のような **log-offset＋L2罰則**を入れる（MAP推定として実装が簡単）。

---

## 5.3 条件の重み付け（現場的に重要）

* “重要条件” を強めに当てる（量産条件）
* “探索条件” は弱め（学習用）

[
\mathcal{L}_{data}=\sum_c w_c,\mathcal{L}^{(c)}
]

---

# 6) 物理妥当性を測る：最適化に組み込むべき「物理ペナルティ」と「出力診断」

今回のAIB‑ODEでは、旧来の “見かけ次数” よりも、

* (f_I=1/(1+K_I C_{ref,I}))（Iが効いているか）
* (\phi_B)（Bが枯渇して支配しているか）
* (C_{s,A}/C_{ref,A})（輸送枯渇）
* θの張り付き（0/1）・bracket_fail

が、**物理妥当性の説明に直結**します。

### 推奨：最適化スコアに入れる「軽い物理ペナルティ」

* AB/AIBが勝つのに (\phi_B) が小さすぎる → 過学習疑い
* AI/AIBが勝つのに (f_I\approx1) → 実質不要
* θが広範囲で0/1張り付き → パラメータが不自然
* bracket_fail多発 → 数値が崩れている（解釈禁止）

→ これを “ランキングの透明性” として report に必ず出す。

---

# 7) 実装の“対応部分”：どこに何を書くと第三者が改造しやすいか

あなたの新設計（sim/opt分離）に対して、最適化は次の責務分割が最小で堅牢です。

## 7.1 deposim_opt 側（最適化パッケージ）

* `enumerate_roles.py`：A必須、I/B≤1、disjoint、unused許可
* `enumerate_orders.py`：整数次数候補＋3次制約
* `objective.py`：

  * 入力：Candidate（roles+orders）＋ trial（連続パラメータ）
  * 出力：score（単一目的）
  * 中で `simulate()` を呼び、coarse→fine で `trial.report()`
* `fit_optuna.py`：

  * sampler/pruner/storage の生成（YAMLで選択）
  * studyのresume（SQLite等）
* `ranking.py`：

  * 候補ごとの最良スコアをまとめて `ranking.csv`
* `class_compare.py`：

  * A/AI/AB/AIBのbest比較＋差分＋“識別不能”判定

Optunaの保存・再開はSQLite等のRDB backendが公式に推奨されます。([optuna.readthedocs.io][12])

## 7.2 deposim_sim 側（前向き計算）

* `simulate.py`：steady/transient統一（同じforwardを回す）
* `diagnostics.py`：phi_B, f_I, Cs/Cref, solver健全性
* `metrics.py`：Huber loss, NU%, radial等

---

# 8) YAMLでの最適化方式の選択（“迷わない最小”）

ここは要点だけ（詳細YAMLは既に前段で詰めている前提）：

* `opt.sampler: tpe | cmaes | nsgaii | botorch (P2)`
* `opt.pruner: median | hyperband | none`
* `opt.fidelity.levels: [coarse, mid, fine]`（点数/dt/条件数を指定）
* `opt.score.penalties: {complexity, role, solver, prior}`
* `opt.storage: sqlite:///...`（resume必須）([optuna.readthedocs.io][12])

---

# 9) この方針が「今回の改良」と最も整合する理由

1. **AIB‑ODEは構造が固定**なので、旧MSのような “式のON/OFF地獄” を回避できる
2. しかし **species→role** と **整数次数** が離散なので、混合最適化が必須
3. 離散は外出し列挙、連続はOptuna、という分離は **実装が短く、レビューしやすい**
4. 物理妥当性は、AIB‑ODE固有の診断（(\phi_B,f_I,C_s/C_{ref})）を **スコアとレポートに直結**させるのが最も強い
5. プルーニング＋多忠実度は Optunaの設計思想と合い、実装も軽い([optuna.readthedocs.io][4])

---

# 10) 追加で検討すべき “データ同化寄り” 手法（P2以降の候補）

将来「不確かさ（UQ）まで欲しい」「同定の安定性を上げたい」となった場合の候補です。

## 10.1 EKI/EnKF系（導関数不要・アンサンブルで逆問題）

EnKF/EnKI/Ensemble Kalman inversion は、導関数を使わずに逆問題を解く枠組みとして広く知られています。([math.umd.edu][13])
ただし専用ライブラリがプロジェクト依存になりやすいので、**P2で“必要になったら最小実装”**が現実的です。

## 10.2 ベイズ最適化（GP/BO）

BoTorchは制約付き・多目的BOの実装が豊富で、理論的にも強いです。([botorch.org][10])
Optunaから呼べる BoTorchSampler もありますが、依存が増えるため運用判断が必要です。([optuna-integration.readthedocs.io][14])

---

# 参考文献・ドキュメント（URL付き）

（URLは指示どおりコードブロックで列挙）

```text
[Optuna: サンプラー概要]
https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
https://docs-ja.optuna.org/reference/samplers/index.html

[Optuna: TPESampler / CmaEsSampler]
https://docs-ja.optuna.org/reference/samplers/generated/optuna.samplers.TPESampler.html
https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html

[Optuna: NSGA-II（多目的）]
https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html

[Optuna: Pruners（Median/Hyperband）]
https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html
https://docs-ja.optuna.org/reference/generated/optuna.pruners.HyperbandPruner.html
https://docs-ja.optuna.org/tutorial/10_key_features/003_efficient_optimization_algorithms.html

[Optuna: Studyの保存/再開（SQLite/RDB）]
https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
https://optuna.readthedocs.io/en/v3.6.2/faq.html

[OptunaHub / Optuna Integration: BoTorchSampler]
https://hub.optuna.org/samplers/botorch_sampler/
https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.BoTorchSampler.html

[BoTorch: 制約付き・多目的BO]
https://botorch.org/docs/tutorials/constrained_multi_objective_bo/
https://botorch.org/docs/multi_objective/

[EnKF/データ同化（参考：同化・逆問題の背景）]
https://www.math.umd.edu/~slud/RITF17/enkf-tutorial.pdf
https://www.ecmwf.int/sites/default/files/elibrary/2003/9756-ensemble-based-data-assimilation.pdf
```

---

## 次に（実装に直結する提案）

もし次のステップとして「すぐ実装できる形」に落とすなら、私は

1. **opt YAMLの“sampler/pruner/fidelity”最小テンプレ**（CVD steady / CVD transient / ALD transient）
2. objective関数の **coarse→fine報告設計（report stepの定義）**
3. ranking/class_compare 出力仕様（識別不能判定のルール含む）

を、あなたの既存 `deposim_opt` 構造に合わせて “コピペ可能な仕様” として提示できます。

[1]: https://docs-ja.optuna.org/tutorial/10_key_features/003_efficient_optimization_algorithms.html?utm_source=chatgpt.com "効率的な最適化アルゴリズム"
[2]: https://docs-ja.optuna.org/tutorial/10_key_features/002_configurations.html?utm_source=chatgpt.com "Pythonic な探索空間 — Optuna 4.4.0.dev documentation"
[3]: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html?utm_source=chatgpt.com "optuna.samplers - Read the Docs"
[4]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html?utm_source=chatgpt.com "optuna.pruners.MedianPruner - Read the Docs"
[5]: https://docs-ja.optuna.org/reference/pruners.html?utm_source=chatgpt.com "optuna.pruners — Optuna 4.4.0.dev documentation"
[6]: https://optuna.readthedocs.io/?utm_source=chatgpt.com "Optuna: A hyperparameter optimization framework — Optuna ..."
[7]: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html?utm_source=chatgpt.com "optuna.samplers.CmaEsSampler - Read the Docs"
[8]: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html?utm_source=chatgpt.com "optuna.samplers.NSGAIISampler - Read the Docs"
[9]: https://hub.optuna.org/samplers/botorch_sampler/?utm_source=chatgpt.com "BoTorch Sampler"
[10]: https://botorch.org/docs/tutorials/constrained_multi_objective_bo/?utm_source=chatgpt.com "Constrained multi-objective optimization with qNEHVI and ..."
[11]: https://docs-ja.optuna.org/reference/generated/optuna.pruners.HyperbandPruner.html?utm_source=chatgpt.com "optuna.pruners.HyperbandPruner"
[12]: https://optuna.readthedocs.io/en/v3.6.2/faq.html?utm_source=chatgpt.com "FAQ — Optuna 3.6.2 documentation"
[13]: https://www.math.umd.edu/~slud/RITF17/enkf-tutorial.pdf?utm_source=chatgpt.com "Understanding the Ensemble Kalman Filter"
[14]: https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.BoTorchSampler.html?utm_source=chatgpt.com "optuna_integration.BoTorchSampler - Optuna-Integration"
