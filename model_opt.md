目的を数式で言い換えると、あなたがやりたいことは **(A) 係数推定（データ同化 / 逆問題）**だけでなく、同時に **(B) モデル構造の選択（どの理論項を入れるか）**まで含む、いわゆる

* **階層化された逆問題（hierarchical inverse problem）**
* **モデル選択（model selection / structure learning）**
* **物理妥当性を壊さない正則化（physics‑aware regularization）**

の組合せです。

以下では「単一条件/複数条件」「1D/2D膜厚」「複数の数値モデル（MS-01〜15 + 項のON/OFF）」を前提に、**自動で（半自動ではなく）推定・選択するために必要な手法**と、**Pythonでの関連技術（ライブラリ）**を、実務で回る順序でまとめます。

---

## 1) まず問題設定：最適化すべき対象は3階層ある

### 観測（実測）

条件 (d=1,\dots,D)（流量/温度/圧力/回転数/…）ごとに、
膜厚マップ (y_d)（1Dでも2Dでも良い、欠損maskあり）を観測。

### 前向きモデル（あなたの本コード）

候補モデル（モデルセット + 項ON/OFF）を (M) とし、パラメータを (\theta) とすると、
[
h_d^{(M)} = \mathcal{F}^{(M)}(x,y;; \theta,; \text{inputs}_d)
]
ここで (\mathcal{F}^{(M)}) は MS-01〜15 の組合せ（TC結合、km、RateCore、State、Net、…）です。

### 実測と比較する観測作用素

実測座標への変換（dx,dy,回転、補間、mask、edge exclusion）を (\mathcal{H}) とすると、
[
\hat y_d^{(M)} = \mathcal{H}!\left(h_d^{(M)}\right)
]

---

## 2) 「係数推定」だけなら最小二乗で済むが、あなたの目的では足りない理由

あなたが指摘した通り、

* 理論を増やすほどパラメータ数が増える
* ブラックボックス係数を増やすほど“合わせられる”が外挿で壊れる
* どの理論を採用すべきかで **同定対象の次元が変わる**（モデルごとに (\theta) の次元が違う）

ので、単に
[
\min_\theta \sum_d | y_d-\hat y_d(\theta)|^2
]
では **モデルが複雑なほど勝つ**（過学習）構造になり、物理妥当性も測れません。

したがって必要なのは以下のどれか（実務では併用が多い）です：

1. **複雑さに罰則を入れる（正則化 / 情報量規準 / MDL）**
2. **学習と評価を分ける（条件ホールドアウト/クロスバリデーション）**
3. **確率モデル化して比較する（ベイズ的モデル比較：証拠、LOO/WAIC）**

---

## 3) 必要な手法（全体像）：連続パラメータ推定 × モデル選択 × 物理妥当性

### 3.1 連続パラメータ推定（固定モデルMのもとで (\theta) を推定）

以下の3系統を準備しておくと、モデルの性質に応じて切り替えられます。

#### (A) 勾配法（高速・高精度、ただし微分可能性が鍵）

* 目的関数：負の対数尤度 + 正則化
  [
  \min_\theta ;; \underbrace{\sum_d \rho!\left(W_d\left(y_d-\hat y_d(\theta)\right)\right)}_{\text{データ適合}}

  * \underbrace{\lambda \Omega(\theta)}_{\text{物理/複雑性の正則化}}
    ]

  - (\rho)：L2だけでなくHuber/Student‑tなどのロバスト損失も重要（外れや欠損に強い）
  - (W_d)：maskや信頼度重み

* 実装上の要点：
  あなたの前向き計算は **root（TCのR解）**や、ALDなら **ODE**を含むので、勾配を使うなら
  **（i）自動微分**＋（ii）root/ODEの微分（暗黙微分）
  が必要になります。

* JAX/JAXopt なら、root solver（Bisection等）と暗黙微分を扱える設計が取りやすいです（後述）。([jaxopt.github.io][1])

**使いどころ**

* MS-01/02/03/04/05 のような「TC+代数的root中心」のCVDに非常に相性が良い
* DOEで大量に回しても高速に収束しやすい（ただしJITオーバーヘッドは別）

#### (B) 勾配なし最適化（非滑らかでも動く、ただし試行回数が増えがち）

* CMA‑ES / DE / NG系など
* 非滑らか（ガード、piecewise、離散スイッチが多い）でも動く

例：Nevergrad は勾配不要の最適化ライブラリです。([Facebook Research][2])

**使いどころ**

* “モデルの切替・ガード・フォールバック”が多い探索初期
* まず粗い当たりを取りたいとき
* ただし forward が高価だと計算が重い

#### (C) アンサンブル型（EKI/EKI系）：並列しやすく、微分不要

* 逆問題（パラメータ推定）を、EnKFの更新則に近い形で反復して解く
* 微分不要、並列に強い、パラメータ数が増えてもある程度耐える
* 研究系では **Ensemble Kalman Inversion (EKI)** がよく使われます([arXiv][3])

**注意**

* Pythonで“定番”のEKIライブラリは分野ごとに散っており、実務では自前実装になることも多い（概念は論文/実装例に基づきやすい）。
* EnKF自体は Python でも実装例があり（FilterPyなど）、考え方の参考にはなります。([filterpy.readthedocs.io][4])

**使いどころ**

* パラメータが多く、勾配が取りづらい/信用できない場合
* DOEを並列に回せる環境がある場合（GPUでなくてもCPU並列で効く）

---

### 3.2 モデル選択（どの理論項を入れるか）＝「離散＋連続の混合最適化」

あなたのMS-01〜15や、RateLawのterm ON/OFF、Transform ON/OFFは **離散変数**です。
これを自動化する手法は大きく3つです。

#### (1) 列挙＋スコアリング（実務で最も堅い）

候補モデル集合 ({M_k}) を（物理的に意味のある範囲で）有限個に絞り、
各モデルで (\theta) を最適化してから、モデルを比較します。

比較方法（どれか1つではなく併用推奨）：

* **BIC/AIC 的な複雑さペナルティ**（パラメータ数に罰則）
* **条件ホールドアウト**（条件Dの一部を検証用にして外挿性能を見る）
* **残差マップの構造**（リング状や方位角パターンが残る等）

これは実務で壊れにくいです（ただし候補数が大きいと重い）。

#### (2) “超モデル（super‑model）”＋スパース化（項選択を連続化する）

各理論項 (T_k) にゲート (g_k\in[0,1]) を掛けて
[
r = r_{\text{core}}(\cdot)\times \prod_k f_k(\cdot)^{g_k}
\quad\text{or}\quad
r = r_{\text{core}}(\cdot) + \sum_k g_k T_k(\cdot)
]
のようにし、(\sum |g_k|)（L1）やグループLassoで **不要項を0へ収縮**させます。

**利点**

* 「項のON/OFF」と「係数推定」を同時に回しやすい
* ブラックボックス係数を増やしても“収縮”で暴れにくい

**注意**

* 物理的に意味が変わる排他（例：RateCoreの種類）は連続化しにくい
  → ここは後述の“カテゴリ変数探索”と併用が現実的

#### (3) ハイパーパラメータ最適化（HPO）として探索する（混合変数探索）

モデル構造（カテゴリ変数）＋連続パラメータをまとめて探索する枠組みです。

* **Optuna**：define‑by‑runで探索空間を柔軟に書け、pruning（見込みの薄い試行の早期打ち切り）も持ちます。([Optuna][5])
* **BoTorch**：ベイズ最適化（BO）で surrogate を使い、混合連続/離散の最適化の考え方を提供しています（fixed features など）。([BoTorch][6])
  さらに multi‑fidelity（粗い計算→精密計算）と離散fidelityの扱いに関するチュートリアルもあります。([BoTorch][7])

**使いどころ**

* 候補モデルが多く、列挙が現実的でない
* DOEを回すのが比較的安い（あなたの格子点数なら十分現実的）
* 途中段階（粗い格子/簡易診断）で “早期打ち切り” を効かせたい

---

### 3.3 「物理妥当性」を定量化して最適化に組み込む

ここがあなたの最重要点です。

ポイントは **“物理妥当性をスコア化して、最適化/選択の評価軸に入れる”**ことです。
大きく2つのやり方があります。

#### (A) 制約・事前分布（priors）として入れる（最も綺麗）

* 例：

  * (k_0>0)（logパラメータ化）
  * (0<s(\theta)\le1)（シグモイド/ベータ分布）
  * (K_i>0)
  * (E_a) が非現実的に大きい/小さいならペナルティ
  * (\delta_\mathrm{eff}) が z_ref から見ておかしいならペナルティ
* さらに「条件間で共通であるべき係数」を共有化（階層化）する

  * 例：反応の (k_0,E_a) は **全条件で共通**
  * (\delta_\mathrm{eff}) やアライメントは **条件依存のノイズ（nuisance）** として弱く許す
    → ブラックボックス係数が増えても“勝手に条件ごとに暴れない”

#### (B) 多目的最適化（fit vs complexity vs plausibility のパレート最適）

* 目的1：データ適合（RMSE/robust loss）
* 目的2：モデル複雑さ（有効パラメータ数、ONの項数）
* 目的3：物理妥当性（ペナルティスコア、レジーム矛盾、過度な負次数など）

こうすると「当たりはするが物理が怪しい」モデルが、パレート面の端（極端）に押し出されます。
多目的最適化の枠組みとして **pymoo（NSGA‑II等）**が使えます。([pymoo.org][8])

---

## 4) ベイズ的にやると何が得か（“物理妥当性”の評価と相性が良い）

ベイズ流に書くと：
[
p(\theta,M\mid D)\propto p(D\mid \theta,M),p(\theta\mid M),p(M)
]

* (p(\theta\mid M))：物理妥当性（範囲・スケール）を **事前分布**で与えられる
* (p(M))：複雑なモデルを事前に不利にできる
* モデル比較：

  * **LOO/WAIC**（予測性能ベース）
  * **Evidence（周辺尤度）**（“複雑さ”を積分で自動的に罰する）

### 4.1 予測性能（LOO/WAIC）でモデル比較

ArviZ は LOO/WAIC の比較機能を提供します。([python.arviz.org][9])
（PyMCと組み合わせた例でも同様の説明があります。([pymc.io][10])）

### 4.2 Evidence（周辺尤度）でモデル比較

Evidence を評価したい場合、nested sampling が有力で、
**dynesty** は Bayesian posterior と evidence 推定のための dynamic nested sampling パッケージです。([dynesty.readthedocs.io][11])

### 4.3 ベイズ推定（NUTS等）を回す基盤

**NumPyro** はJAX上の確率プログラミングで、JAXの自動微分/JITでCPU/GPUを使える設計です。([NumPyro][12])

> 実務的には：
>
> * まずは **MAP推定（最適化）**で当たりを付ける
> * 上位モデルだけ **NumPyro/PyMC/dynesty** で不確かさ・モデル比較
>   が現実的です。

---

## 5) 計算を回すための数値・計算技術（GPUが有用かも含む）

### 5.1 GPUが有用になりやすい条件

* DOEで **100〜1000条件**を同じグリッド形状で回す
* forwardがベクトル化できる（wafer grid × cases）
* 同じコードを繰り返すので **JITの初回コストを償却**できる

このとき JAXの `jit/vmap` は効きやすいです（JAXはJIT・自動微分・並列化のための変換を提供）。([jax.dev][13])

一方、

* 単発条件（1ケース）
* 形状が毎回変わる
* guard分岐が多い
  場合は、JITコンパイルが支配的になり逆効果もあり得ます。
  → したがって **“GPUが有用かは run_strategy で固定せずユーザー選択”**が正しい方針です（あなたの方針と一致）。

### 5.2 root（TCのR解）を含むモデルで“勾配”を使うなら

TC結合のRは root で求めます。これを勾配ベース最適化に入れるには、

* 反復をそのまま微分する（遅い/不安定になりやすい）
* **暗黙微分（Implicit differentiation）**を使う
  のどちらかです。

JAXoptは solver の暗黙微分をサポートし、Bisectionなどのroot solverも提供しています。([jaxopt.github.io][14])

---

## 6) 実務で回る「推奨アーキテクチャ」（あなたのコード設計に沿う）

あなたのパッケージ分割（数値計算部と最適化/ML部を分離、YAML/Hydraで選択、model registry）を前提に、最適化側に最低限必要な部品は次です。

### 6.1 deposim_opt（最適化パッケージ）側に必要なモジュール

1. **ModelSpace**

* MS-01〜15 + 各項ON/OFFを「探索空間」として表現
* 互換性（requires/excludes/time_mode）を使って無効構成を排除（P0で必須）

2. **Parameterization**

* 物理パラメータの変換

  * 正値：log/softplus
  * 0〜1：sigmoid
* 条件共通 vs 条件固有（階層）を分離

  * 例：(k_0,E_a,K_i,\beta) は共通
  * (\delta_\mathrm{eff},C_k) 等は条件ごと or 弱い階層

3. **Objective / Likelihood**

* 2Dマップ誤差（mask/edge exclusion）
* ロバスト損失（外れ値/欠損）
* 正則化（物理妥当性・複雑性）

4. **Search Engine**

* 連続推定：SciPy最適化 or JAXopt/Optax
* 構造探索：Optuna（カテゴリ＋連続＋pruning）やBoTorch（BO）
* 多目的：pymoo（NSGA‑II等）
* 仕上げ：NumPyro/PyMC/dynesty + ArviZ（LOO/WAIC）

---

## 7) 具体的に「何のライブラリが必要か」一覧（目的別）

### 7.1 高速な連続最適化（勾配あり/なし）

* **SciPy optimize**（まずは堅牢な標準）
* **JAX + JAXopt**（root/implicit diff/JITで高速化）([jax.dev][13])
* **Optax**（JAX上の最適化アルゴリズム群：Adam等）
* **Nevergrad**（勾配不要の探索）([Facebook Research][2])

### 7.2 構造（理論項）選択：混合離散＋連続

* **Optuna**（カテゴリ・連続・pruning・define-by-run）([Optuna][5])
* **BoTorch**（BO、混合連続/離散、multi-fidelity設計も参考になる）([BoTorch][6])

### 7.3 “物理妥当性を測る”ためのモデル比較（予測性能/複雑性）

* **ArviZ**（LOO/WAICでモデル比較）([python.arviz.org][9])
* **dynesty**（evidence推定：nested sampling）([dynesty.readthedocs.io][11])
* **NumPyro**（JAX上の確率推論、CPU/GPUで動く）([NumPyro][12])

### 7.4 多目的最適化（fit/complexity/plausibilityのパレート）

* **pymoo**（NSGA‑II等）([pymoo.org][8])

### 7.5 アンサンブル系（微分不要の逆問題）

* EKI/EKI系：論文ベースでの自前実装を推奨（概念・発展は文献が豊富）([arXiv][3])
* EnKF参考：FilterPyにEnsembleKalmanFilter実装例あり([filterpy.readthedocs.io][4])

---

## 8) 最後に：あなたの用途での「現実的な自動同定フロー」推奨（手順）

### Step 0：モデル空間を“物理的に意味ある範囲”に絞る

* 例：CVD（連続）が本命なら、まず
  **MS-01/02/03/04/05/06/07/13**
  の範囲で、項ON/OFFを含む空間を作る
* Validatorで「意味のない組合せ」を排除する（最重要）

### Step 1：粗いスクリーニング（高速・頑健）

* Optunaで

  * モデル選択（カテゴリ）
  * term ON/OFF
  * 連続パラメータ
    をまとめて探索
* 途中で pruning（粗い誤差が悪い試行を早期終了）([Optuna][15])

### Step 2：上位モデルを精密最適化（MAP）

* SciPy or JAXoptで連続パラメータを詰める
* JAXoptでrootを含む問題を暗黙微分で扱えるようにしておくと、反復回数を抑えて高速化しやすい([jaxopt.github.io][14])

### Step 3：物理妥当性＆外挿性で評価

* 条件ホールドアウト（別条件で再現できるか）
* 残差の空間構造（リング/方位角の系統誤差が残らないか）
* パラメータの“飛び”（条件間で物理係数が変わっていないか）

### Step 4：不確かさ・モデル比較（必要なら）

* 上位2〜3モデルだけ

  * NumPyroで事後分布（不確かさ）([NumPyro][12])
  * ArviZでLOO/WAIC比較([python.arviz.org][9])
  * dynestyでevidence比較（時間に余裕がある場合）([dynesty.readthedocs.io][11])

---

## 9) 「ブラックボックス係数が多いほど合わせられる」問題への、実務的な解決策まとめ

**結論だけ言うと**、ブラックボックス係数を増やすなら、同時に以下を必須化すると安全です：

1. **階層化（shared vs per-condition nuisance）**
   → 条件ごとに勝手に暴れない
2. **縮小（shrinkage）正則化**（L2/L1/ARD）
   → “本当に必要な自由度だけ”残る
3. **モデル複雑性ペナルティ**（BIC/MDL/モデル事前）
   → 複雑なモデルが自動的に不利
4. **条件ホールドアウト評価**
   → 外挿性で弾ける

これを入れると、「合わせられるが物理が怪しい」モデルは自然に落ちます。


[1]: https://jaxopt.github.io/stable/_autosummary/jaxopt.Bisection.html?utm_source=chatgpt.com "jaxopt.Bisection — JAXopt 0.8 documentation"
[2]: https://facebookresearch.github.io/nevergrad/?utm_source=chatgpt.com "Nevergrad - A gradient-free optimization platform"
[3]: https://arxiv.org/abs/2206.02451?utm_source=chatgpt.com "Component-wise iterative ensemble Kalman inversion for ..."
[4]: https://filterpy.readthedocs.io/en/latest/kalman/EnsembleKalmanFilter.html?utm_source=chatgpt.com "EnsembleKalmanFilter — FilterPy 1.4.4 documentation"
[5]: https://optuna.readthedocs.io/en/latest/?utm_source=chatgpt.com "Optuna: A hyperparameter optimization framework"
[6]: https://botorch.readthedocs.io/en/latest/optim.html?utm_source=chatgpt.com "botorch.optim — BoTorch documentation"
[7]: https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/?utm_source=chatgpt.com "Multi-fidelity Bayesian optimization with discrete ..."
[8]: https://pymoo.org/?utm_source=chatgpt.com "pymoo: Multi-objective Optimization in Python — pymoo: Multi ..."
[9]: https://python.arviz.org/en/v0.20.0/api/generated/arviz.compare.html?utm_source=chatgpt.com "arviz.compare — ArviZ 0.20.0 documentation"
[10]: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html?utm_source=chatgpt.com "Model comparison — PyMC 5.27.1 documentation"
[11]: https://dynesty.readthedocs.io/?utm_source=chatgpt.com "dynesty — dynesty 3.0.0 documentation"
[12]: https://num.pyro.ai/en/stable/getting_started.html?utm_source=chatgpt.com "Getting Started with NumPyro"
[13]: https://jax.dev/?utm_source=chatgpt.com "JAX: High performance array computing — JAX documentation"
[14]: https://jaxopt.github.io/stable/implicit_diff.html?utm_source=chatgpt.com "Implicit differentiation — JAXopt 0.8 documentation"
[15]: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html?utm_source=chatgpt.com "Efficient Optimization Algorithms — Optuna"
目的を数式で言い換えると、あなたがやりたいことは **(A) 係数推定（データ同化 / 逆問題）**だけでなく、同時に **(B) モデル構造の選択（どの理論項を入れるか）**まで含む、いわゆる

* **階層化された逆問題（hierarchical inverse problem）**
* **モデル選択（model selection / structure learning）**
* **物理妥当性を壊さない正則化（physics‑aware regularization）**

の組合せです。

以下では「単一条件/複数条件」「1D/2D膜厚」「複数の数値モデル（MS-01〜15 + 項のON/OFF）」を前提に、**自動で（半自動ではなく）推定・選択するために必要な手法**と、**Pythonでの関連技術（ライブラリ）**を、実務で回る順序でまとめます。

---

## 1) まず問題設定：最適化すべき対象は3階層ある

### 観測（実測）

条件 (d=1,\dots,D)（流量/温度/圧力/回転数/…）ごとに、
膜厚マップ (y_d)（1Dでも2Dでも良い、欠損maskあり）を観測。

### 前向きモデル（あなたの本コード）

候補モデル（モデルセット + 項ON/OFF）を (M) とし、パラメータを (\theta) とすると、
[
h_d^{(M)} = \mathcal{F}^{(M)}(x,y;; \theta,; \text{inputs}_d)
]
ここで (\mathcal{F}^{(M)}) は MS-01〜15 の組合せ（TC結合、km、RateCore、State、Net、…）です。

### 実測と比較する観測作用素

実測座標への変換（dx,dy,回転、補間、mask、edge exclusion）を (\mathcal{H}) とすると、
[
\hat y_d^{(M)} = \mathcal{H}!\left(h_d^{(M)}\right)
]

---

## 2) 「係数推定」だけなら最小二乗で済むが、あなたの目的では足りない理由

あなたが指摘した通り、

* 理論を増やすほどパラメータ数が増える
* ブラックボックス係数を増やすほど“合わせられる”が外挿で壊れる
* どの理論を採用すべきかで **同定対象の次元が変わる**（モデルごとに (\theta) の次元が違う）

ので、単に
[
\min_\theta \sum_d | y_d-\hat y_d(\theta)|^2
]
では **モデルが複雑なほど勝つ**（過学習）構造になり、物理妥当性も測れません。

したがって必要なのは以下のどれか（実務では併用が多い）です：

1. **複雑さに罰則を入れる（正則化 / 情報量規準 / MDL）**
2. **学習と評価を分ける（条件ホールドアウト/クロスバリデーション）**
3. **確率モデル化して比較する（ベイズ的モデル比較：証拠、LOO/WAIC）**

---

## 3) 必要な手法（全体像）：連続パラメータ推定 × モデル選択 × 物理妥当性

### 3.1 連続パラメータ推定（固定モデルMのもとで (\theta) を推定）

以下の3系統を準備しておくと、モデルの性質に応じて切り替えられます。

#### (A) 勾配法（高速・高精度、ただし微分可能性が鍵）

* 目的関数：負の対数尤度 + 正則化
  [
  \min_\theta ;; \underbrace{\sum_d \rho!\left(W_d\left(y_d-\hat y_d(\theta)\right)\right)}_{\text{データ適合}}

  * \underbrace{\lambda \Omega(\theta)}_{\text{物理/複雑性の正則化}}
    ]

  - (\rho)：L2だけでなくHuber/Student‑tなどのロバスト損失も重要（外れや欠損に強い）
  - (W_d)：maskや信頼度重み

* 実装上の要点：
  あなたの前向き計算は **root（TCのR解）**や、ALDなら **ODE**を含むので、勾配を使うなら
  **（i）自動微分**＋（ii）root/ODEの微分（暗黙微分）
  が必要になります。

* JAX/JAXopt なら、root solver（Bisection等）と暗黙微分を扱える設計が取りやすいです（後述）。([jaxopt.github.io][1])

**使いどころ**

* MS-01/02/03/04/05 のような「TC+代数的root中心」のCVDに非常に相性が良い
* DOEで大量に回しても高速に収束しやすい（ただしJITオーバーヘッドは別）

#### (B) 勾配なし最適化（非滑らかでも動く、ただし試行回数が増えがち）

* CMA‑ES / DE / NG系など
* 非滑らか（ガード、piecewise、離散スイッチが多い）でも動く

例：Nevergrad は勾配不要の最適化ライブラリです。([Facebook Research][2])

**使いどころ**

* “モデルの切替・ガード・フォールバック”が多い探索初期
* まず粗い当たりを取りたいとき
* ただし forward が高価だと計算が重い

#### (C) アンサンブル型（EKI/EKI系）：並列しやすく、微分不要

* 逆問題（パラメータ推定）を、EnKFの更新則に近い形で反復して解く
* 微分不要、並列に強い、パラメータ数が増えてもある程度耐える
* 研究系では **Ensemble Kalman Inversion (EKI)** がよく使われます([arXiv][3])

**注意**

* Pythonで“定番”のEKIライブラリは分野ごとに散っており、実務では自前実装になることも多い（概念は論文/実装例に基づきやすい）。
* EnKF自体は Python でも実装例があり（FilterPyなど）、考え方の参考にはなります。([filterpy.readthedocs.io][4])

**使いどころ**

* パラメータが多く、勾配が取りづらい/信用できない場合
* DOEを並列に回せる環境がある場合（GPUでなくてもCPU並列で効く）

---

### 3.2 モデル選択（どの理論項を入れるか）＝「離散＋連続の混合最適化」

あなたのMS-01〜15や、RateLawのterm ON/OFF、Transform ON/OFFは **離散変数**です。
これを自動化する手法は大きく3つです。

#### (1) 列挙＋スコアリング（実務で最も堅い）

候補モデル集合 ({M_k}) を（物理的に意味のある範囲で）有限個に絞り、
各モデルで (\theta) を最適化してから、モデルを比較します。

比較方法（どれか1つではなく併用推奨）：

* **BIC/AIC 的な複雑さペナルティ**（パラメータ数に罰則）
* **条件ホールドアウト**（条件Dの一部を検証用にして外挿性能を見る）
* **残差マップの構造**（リング状や方位角パターンが残る等）

これは実務で壊れにくいです（ただし候補数が大きいと重い）。

#### (2) “超モデル（super‑model）”＋スパース化（項選択を連続化する）

各理論項 (T_k) にゲート (g_k\in[0,1]) を掛けて
[
r = r_{\text{core}}(\cdot)\times \prod_k f_k(\cdot)^{g_k}
\quad\text{or}\quad
r = r_{\text{core}}(\cdot) + \sum_k g_k T_k(\cdot)
]
のようにし、(\sum |g_k|)（L1）やグループLassoで **不要項を0へ収縮**させます。

**利点**

* 「項のON/OFF」と「係数推定」を同時に回しやすい
* ブラックボックス係数を増やしても“収縮”で暴れにくい

**注意**

* 物理的に意味が変わる排他（例：RateCoreの種類）は連続化しにくい
  → ここは後述の“カテゴリ変数探索”と併用が現実的

#### (3) ハイパーパラメータ最適化（HPO）として探索する（混合変数探索）

モデル構造（カテゴリ変数）＋連続パラメータをまとめて探索する枠組みです。

* **Optuna**：define‑by‑runで探索空間を柔軟に書け、pruning（見込みの薄い試行の早期打ち切り）も持ちます。([Optuna][5])
* **BoTorch**：ベイズ最適化（BO）で surrogate を使い、混合連続/離散の最適化の考え方を提供しています（fixed features など）。([BoTorch][6])
  さらに multi‑fidelity（粗い計算→精密計算）と離散fidelityの扱いに関するチュートリアルもあります。([BoTorch][7])

**使いどころ**

* 候補モデルが多く、列挙が現実的でない
* DOEを回すのが比較的安い（あなたの格子点数なら十分現実的）
* 途中段階（粗い格子/簡易診断）で “早期打ち切り” を効かせたい

---

### 3.3 「物理妥当性」を定量化して最適化に組み込む

ここがあなたの最重要点です。

ポイントは **“物理妥当性をスコア化して、最適化/選択の評価軸に入れる”**ことです。
大きく2つのやり方があります。

#### (A) 制約・事前分布（priors）として入れる（最も綺麗）

* 例：

  * (k_0>0)（logパラメータ化）
  * (0<s(\theta)\le1)（シグモイド/ベータ分布）
  * (K_i>0)
  * (E_a) が非現実的に大きい/小さいならペナルティ
  * (\delta_\mathrm{eff}) が z_ref から見ておかしいならペナルティ
* さらに「条件間で共通であるべき係数」を共有化（階層化）する

  * 例：反応の (k_0,E_a) は **全条件で共通**
  * (\delta_\mathrm{eff}) やアライメントは **条件依存のノイズ（nuisance）** として弱く許す
    → ブラックボックス係数が増えても“勝手に条件ごとに暴れない”

#### (B) 多目的最適化（fit vs complexity vs plausibility のパレート最適）

* 目的1：データ適合（RMSE/robust loss）
* 目的2：モデル複雑さ（有効パラメータ数、ONの項数）
* 目的3：物理妥当性（ペナルティスコア、レジーム矛盾、過度な負次数など）

こうすると「当たりはするが物理が怪しい」モデルが、パレート面の端（極端）に押し出されます。
多目的最適化の枠組みとして **pymoo（NSGA‑II等）**が使えます。([pymoo.org][8])

---

## 4) ベイズ的にやると何が得か（“物理妥当性”の評価と相性が良い）

ベイズ流に書くと：
[
p(\theta,M\mid D)\propto p(D\mid \theta,M),p(\theta\mid M),p(M)
]

* (p(\theta\mid M))：物理妥当性（範囲・スケール）を **事前分布**で与えられる
* (p(M))：複雑なモデルを事前に不利にできる
* モデル比較：

  * **LOO/WAIC**（予測性能ベース）
  * **Evidence（周辺尤度）**（“複雑さ”を積分で自動的に罰する）

### 4.1 予測性能（LOO/WAIC）でモデル比較

ArviZ は LOO/WAIC の比較機能を提供します。([python.arviz.org][9])
（PyMCと組み合わせた例でも同様の説明があります。([pymc.io][10])）

### 4.2 Evidence（周辺尤度）でモデル比較

Evidence を評価したい場合、nested sampling が有力で、
**dynesty** は Bayesian posterior と evidence 推定のための dynamic nested sampling パッケージです。([dynesty.readthedocs.io][11])

### 4.3 ベイズ推定（NUTS等）を回す基盤

**NumPyro** はJAX上の確率プログラミングで、JAXの自動微分/JITでCPU/GPUを使える設計です。([NumPyro][12])

> 実務的には：
>
> * まずは **MAP推定（最適化）**で当たりを付ける
> * 上位モデルだけ **NumPyro/PyMC/dynesty** で不確かさ・モデル比較
>   が現実的です。

---

## 5) 計算を回すための数値・計算技術（GPUが有用かも含む）

### 5.1 GPUが有用になりやすい条件

* DOEで **100〜1000条件**を同じグリッド形状で回す
* forwardがベクトル化できる（wafer grid × cases）
* 同じコードを繰り返すので **JITの初回コストを償却**できる

このとき JAXの `jit/vmap` は効きやすいです（JAXはJIT・自動微分・並列化のための変換を提供）。([jax.dev][13])

一方、

* 単発条件（1ケース）
* 形状が毎回変わる
* guard分岐が多い
  場合は、JITコンパイルが支配的になり逆効果もあり得ます。
  → したがって **“GPUが有用かは run_strategy で固定せずユーザー選択”**が正しい方針です（あなたの方針と一致）。

### 5.2 root（TCのR解）を含むモデルで“勾配”を使うなら

TC結合のRは root で求めます。これを勾配ベース最適化に入れるには、

* 反復をそのまま微分する（遅い/不安定になりやすい）
* **暗黙微分（Implicit differentiation）**を使う
  のどちらかです。

JAXoptは solver の暗黙微分をサポートし、Bisectionなどのroot solverも提供しています。([jaxopt.github.io][14])

---

## 6) 実務で回る「推奨アーキテクチャ」（あなたのコード設計に沿う）

あなたのパッケージ分割（数値計算部と最適化/ML部を分離、YAML/Hydraで選択、model registry）を前提に、最適化側に最低限必要な部品は次です。

### 6.1 deposim_opt（最適化パッケージ）側に必要なモジュール

1. **ModelSpace**

* MS-01〜15 + 各項ON/OFFを「探索空間」として表現
* 互換性（requires/excludes/time_mode）を使って無効構成を排除（P0で必須）

2. **Parameterization**

* 物理パラメータの変換

  * 正値：log/softplus
  * 0〜1：sigmoid
* 条件共通 vs 条件固有（階層）を分離

  * 例：(k_0,E_a,K_i,\beta) は共通
  * (\delta_\mathrm{eff},C_k) 等は条件ごと or 弱い階層

3. **Objective / Likelihood**

* 2Dマップ誤差（mask/edge exclusion）
* ロバスト損失（外れ値/欠損）
* 正則化（物理妥当性・複雑性）

4. **Search Engine**

* 連続推定：SciPy最適化 or JAXopt/Optax
* 構造探索：Optuna（カテゴリ＋連続＋pruning）やBoTorch（BO）
* 多目的：pymoo（NSGA‑II等）
* 仕上げ：NumPyro/PyMC/dynesty + ArviZ（LOO/WAIC）

---

## 7) 具体的に「何のライブラリが必要か」一覧（目的別）

### 7.1 高速な連続最適化（勾配あり/なし）

* **SciPy optimize**（まずは堅牢な標準）
* **JAX + JAXopt**（root/implicit diff/JITで高速化）([jax.dev][13])
* **Optax**（JAX上の最適化アルゴリズム群：Adam等）
* **Nevergrad**（勾配不要の探索）([Facebook Research][2])

### 7.2 構造（理論項）選択：混合離散＋連続

* **Optuna**（カテゴリ・連続・pruning・define-by-run）([Optuna][5])
* **BoTorch**（BO、混合連続/離散、multi-fidelity設計も参考になる）([BoTorch][6])

### 7.3 “物理妥当性を測る”ためのモデル比較（予測性能/複雑性）

* **ArviZ**（LOO/WAICでモデル比較）([python.arviz.org][9])
* **dynesty**（evidence推定：nested sampling）([dynesty.readthedocs.io][11])
* **NumPyro**（JAX上の確率推論、CPU/GPUで動く）([NumPyro][12])

### 7.4 多目的最適化（fit/complexity/plausibilityのパレート）

* **pymoo**（NSGA‑II等）([pymoo.org][8])

### 7.5 アンサンブル系（微分不要の逆問題）

* EKI/EKI系：論文ベースでの自前実装を推奨（概念・発展は文献が豊富）([arXiv][3])
* EnKF参考：FilterPyにEnsembleKalmanFilter実装例あり([filterpy.readthedocs.io][4])

---

## 8) 最後に：あなたの用途での「現実的な自動同定フロー」推奨（手順）

### Step 0：モデル空間を“物理的に意味ある範囲”に絞る

* 例：CVD（連続）が本命なら、まず
  **MS-01/02/03/04/05/06/07/13**
  の範囲で、項ON/OFFを含む空間を作る
* Validatorで「意味のない組合せ」を排除する（最重要）

### Step 1：粗いスクリーニング（高速・頑健）

* Optunaで

  * モデル選択（カテゴリ）
  * term ON/OFF
  * 連続パラメータ
    をまとめて探索
* 途中で pruning（粗い誤差が悪い試行を早期終了）([Optuna][15])

### Step 2：上位モデルを精密最適化（MAP）

* SciPy or JAXoptで連続パラメータを詰める
* JAXoptでrootを含む問題を暗黙微分で扱えるようにしておくと、反復回数を抑えて高速化しやすい([jaxopt.github.io][14])

### Step 3：物理妥当性＆外挿性で評価

* 条件ホールドアウト（別条件で再現できるか）
* 残差の空間構造（リング/方位角の系統誤差が残らないか）
* パラメータの“飛び”（条件間で物理係数が変わっていないか）

### Step 4：不確かさ・モデル比較（必要なら）

* 上位2〜3モデルだけ

  * NumPyroで事後分布（不確かさ）([NumPyro][12])
  * ArviZでLOO/WAIC比較([python.arviz.org][9])
  * dynestyでevidence比較（時間に余裕がある場合）([dynesty.readthedocs.io][11])

---

## 9) 「ブラックボックス係数が多いほど合わせられる」問題への、実務的な解決策まとめ

**結論だけ言うと**、ブラックボックス係数を増やすなら、同時に以下を必須化すると安全です：

1. **階層化（shared vs per-condition nuisance）**
   → 条件ごとに勝手に暴れない
2. **縮小（shrinkage）正則化**（L2/L1/ARD）
   → “本当に必要な自由度だけ”残る
3. **モデル複雑性ペナルティ**（BIC/MDL/モデル事前）
   → 複雑なモデルが自動的に不利
4. **条件ホールドアウト評価**
   → 外挿性で弾ける

これを入れると、「合わせられるが物理が怪しい」モデルは自然に落ちます。

[1]: https://jaxopt.github.io/stable/_autosummary/jaxopt.Bisection.html?utm_source=chatgpt.com "jaxopt.Bisection — JAXopt 0.8 documentation"
[2]: https://facebookresearch.github.io/nevergrad/?utm_source=chatgpt.com "Nevergrad - A gradient-free optimization platform"
[3]: https://arxiv.org/abs/2206.02451?utm_source=chatgpt.com "Component-wise iterative ensemble Kalman inversion for ..."
[4]: https://filterpy.readthedocs.io/en/latest/kalman/EnsembleKalmanFilter.html?utm_source=chatgpt.com "EnsembleKalmanFilter — FilterPy 1.4.4 documentation"
[5]: https://optuna.readthedocs.io/en/latest/?utm_source=chatgpt.com "Optuna: A hyperparameter optimization framework"
[6]: https://botorch.readthedocs.io/en/latest/optim.html?utm_source=chatgpt.com "botorch.optim — BoTorch documentation"
[7]: https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/?utm_source=chatgpt.com "Multi-fidelity Bayesian optimization with discrete ..."
[8]: https://pymoo.org/?utm_source=chatgpt.com "pymoo: Multi-objective Optimization in Python — pymoo: Multi ..."
[9]: https://python.arviz.org/en/v0.20.0/api/generated/arviz.compare.html?utm_source=chatgpt.com "arviz.compare — ArviZ 0.20.0 documentation"
[10]: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html?utm_source=chatgpt.com "Model comparison — PyMC 5.27.1 documentation"
[11]: https://dynesty.readthedocs.io/?utm_source=chatgpt.com "dynesty — dynesty 3.0.0 documentation"
[12]: https://num.pyro.ai/en/stable/getting_started.html?utm_source=chatgpt.com "Getting Started with NumPyro"
[13]: https://jax.dev/?utm_source=chatgpt.com "JAX: High performance array computing — JAX documentation"
[14]: https://jaxopt.github.io/stable/implicit_diff.html?utm_source=chatgpt.com "Implicit differentiation — JAXopt 0.8 documentation"
[15]: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html?utm_source=chatgpt.com "Efficient Optimization Algorithms — Optuna"


了解です。ここでは **「実装が複雑になりすぎない」**ことを最優先にしつつ、後から第三者が

* 最適化指標（損失・KPI・物理妥当性ペナルティ）
* 最適化手法（探索アルゴリズム、pruning、multi-objective）
* モデル構造（MS-01〜15＋項ON/OFF）

を **安全に追加・改良できる**形に落とします。

加えてご指定どおり：

* **ブラックボックス最適化は Optuna を採用**（SciPyのDE等を直接使わない）
* Optunaは基本 **勾配法ではない**（TPE/CMA-ES/GA等のサンプラー）ので、勾配法を使うなら **Optunaの外（後段）で任意**にする

という方針でまとめます。Optunaの代表サンプラー（TPE/CMA-ES）と multi-objective の NSGA-II は公式ドキュメントにあります。 ([Optuna][1])

---

# 1) まず「一番シンプルに目的を達成する」戦略

あなたの目的は **(1) 係数推定（逆問題）** と **(2) モデル構造選択（理論項の有無）** の同時達成です。
実装複雑化を避けるなら、私は次の “二段構え” を推奨します。

## 推奨フロー（シンプルで現場導入しやすい）

### Stage A：モデル候補（MS）ごとに Optuna で係数推定（黒箱）

* 候補モデル集合 ( {M_k} )（例：MS-01/02/03/04/05/06/07/08/09/10/11/12/13 から必要分）を用意
* 各モデル (M_k) に対して Optuna で連続パラメータ (\theta) を推定
* **同じ objective 定義**で “best score” を取る

Optunaは define-by-run で探索空間を動的に構成できるため（条件付き探索空間が書きやすい）、「モデルにより最適化パラメータ数が変わる」問題を扱いやすいです。 ([Optuna][2])

### Stage B：モデル比較（“当てやすさ”＋“物理妥当性”＋“複雑さ”）でランキング

* best trial を **同一の比較指標**で評価し、モデルを選択
* 過学習を避けるために、可能なら **条件ホールドアウト（LOCO等）**を P1 で導入

> これにより、**「パラメータが多いほど勝つ」問題を、複雑度罰則＋外挿評価で抑える**ことができます。

---

# 2) 実装を複雑にしないためのコア設計（最小インターフェース）

最適化側（`deposim_opt`）は、数値計算側（`deposim_core`）に対して **たった1つの安定I/F**だけ要求します：

```python
SimulationResult = simulate(sim_cfg, case_inputs, params_dict)
# SimulationResult.thickness_map  (1D/2D)
# SimulationResult.diagnostics    (Cs/Cref, n_app, solver_stats, regime_numbers ...)
```

最適化側は「simulateを呼んで、測定に合わせて、スコア返す」だけ。
これが **実装を増やしても破綻しない最短**です。

---

# 3) `deposim_opt` のパッケージ構成（第三者が触る場所を限定）

“追加・改良しやすい” ために **拡張点を3つに絞ります**：

1. **Objective（指標）**：loss term を追加
2. **ModelCatalog（モデル候補）**：MSや項ON/OFFの候補を追加
3. **Optimizer（手法）**：Optunaの sampler/pruner を差し替え（基本は設定追加のみ）

## 推奨ディレクトリ（最小）

```
deposim_opt/
  dataset/               # 条件/測定の読み込みと整形
    observation.py       # 1D/2D + mask + weights を統一表現
  model_catalog/         # 候補モデル（MS）とパラメータ空間
    catalog.py           # CandidateModelの集合
    param_space.py       # ParamSpec: bounds/transform/shared_scope
  objective/
    objective.py         # LossTermを合成してスコア化
    terms/
      data_misfit.py
      kpi_misfit.py
      prior_penalty.py
      complexity_penalty.py
      regime_penalty.py
  optim/
    optuna_runner.py     # Optunaでstudyを回す（唯一の最適化実装）
  eval/
    protocol.py          # train/val split（P0はall、P1でholdout）
  reports/
    opt_report.py        # ランキング、best再計算、可視化
```

> 追加実装のとき第三者が触るのは、基本 `objective/terms/*` と `model_catalog/*` と `configs/opt/*` だけになります（壊れにくい）。

---

# 4) Objective を “合成可能な部品” にする（指標の改良が容易）

Optunaの objective は **スカラー**（または multi-objective ならベクトル）を返します。
ここを複雑にしないために、**LossTerm の足し算**で作るのが一番メンテしやすいです。

## 4.1 推奨スコア構造（単一目的の標準形）

[
J(M,\theta)=
w_\text{data}J_\text{data}+
w_\text{kpi}J_\text{kpi}+
w_\text{prior}J_\text{prior}+
w_\text{complex}J_\text{complex}+
w_\text{regime}J_\text{regime}
]

* **J_data**：膜厚マップ残差（1D/2D、mask、ロバスト損失）
* **J_kpi**：NU%等のKPI一致
* **J_prior**：物理妥当性（係数の範囲・符号・スケール）を “弱い罰則” で入れる
* **J_complex**：有効パラメータ数・有効項数を罰する（過学習抑制）
* **J_regime**：レジーム矛盾（例：ω=0で回転相関、Kn高いのに連続体等）を罰する（※P0は軽く、P1で強化）

## 4.2 各LossTermが返すもの（スコア＋詳細）

各 term は以下だけ実装すればよい、にします：

```python
class LossTerm:
    name: str
    def compute(self, pred, obs, sim_diag, params, model_meta) -> dict:
        return {"value": float, "details": {...}}
```

Objective合成側は `value` を重み付きで足すだけ。
`details` は **Optunaの trial.user_attrs** に入れて後で分析できます（第三者が指標を追加しやすい）。

---

# 5) モデル構造選択（理論項の有無）を “簡単に・安全に” 扱う方法

「モデルにより推定係数の数が異なる」「ブラックボックス係数が多いほど合わせられる」という問題を同時に扱うには、**探索空間を野放しにしない**のが重要です。

## 5.1 一番壊れにくい方法：ModelCatalog（候補MS）を “有限集合” として管理

* MS-01〜15全部を自由にON/OFF探索すると爆発しがち
* 代わりに “実務で意味がある組合せ” を CandidateModel として列挙

  * 例：CVD優先なら

    * MS-01, 02, 03, 04, 05, 06, 07, 13
  * ALDなら

    * MS-08, 09, 10, 11, 12

こうすると第三者が候補を追加しても、**Validatorに通る形で増やせます**（既にあなたが設計した requires/excludes が効く）。

## 5.2 “項ON/OFF”は、候補内の ParamSpace に閉じ込める

たとえば sat_inh の分母項（阻害種）を入れる/入れないは、モデル候補の中で

* `term_set` を categorical にする（推奨：探索空間を抑える）
* or boolean enable を少数だけ持つ（5個程度なら許容）

のどちらかにします。
これで探索次元が爆発しにくい。

---

# 6) Optuna の使い方（黒箱最適化）を “実務向けに固定”

あなたの制約に合わせて「SciPyの黒箱最適化の代わり」を Optunaで統一します。

## 6.1 サンプラー選定（設定で差し替え可能）

* **混合（連続＋カテゴリ）**があるので、まずは **TPE** が標準（Optunaの代表） ([Optuna][1])
* 連続中心で局所が難しいなら **CMA-ES** を使える（Optuna samplerとして提供） ([Optuna][3])
* 「物理妥当性 vs 当てやすさ vs 複雑さ」を分けて見たいなら、Optunaの **NSGA-II sampler** で multi-objective（後述） ([Optuna][4])

## 6.2 pruning（計算を軽くする必須テク）

複数条件を回す objective では、途中で見込みがない trial を切るだけで劇的に軽くなります。

* 条件 (d) を順に評価し、途中スコアを `trial.report(score, step=d)` で報告
* `trial.should_prune()` が True なら `optuna.TrialPruned` を投げて終了

Optunaはこの pruned trial のために `TrialPruned` 例外と `Trial.should_prune()` を提供しています。 ([Optuna][5])

> これを入れるだけで「複数モデル×複数条件」でも現実的に回るようになります（実装も小さい）。

---

# 7) 「Optunaは勾配法が使えない」前提での、現実的な“高精度化”オプション

ご指摘どおり、Optunaの標準運用は **サンプリング型（黒箱）**で、勾配法の置き換えではありません（TPE/CMA-ES/NSGA-IIなど）。 ([Optuna][1])

ただし、精度や収束速度を上げたくなる局面は必ず来ます。
そこで実装を複雑にしないために、次の形に固定するのが良いです。

## オプション：**「後段」だけ勾配系で局所再最適化（Refine）**

* Optunaで best trial を取る（グローバル探索）
* best付近だけを、別ルーチンで局所最適化（任意）

これを **Optunaの外**でやれば、「Optunaは勾配法でない」前提と矛盾しません。

### 参考：JAXopt を使う場合（root/暗黙微分を扱いやすい）

あなたの forward は TC root（Bisection等）を含むので、勾配法に持ち込むなら “暗黙微分” が要点になります。
JAXopt には root finding（Bisection）と、その周辺（implicit diff含む）設計要素が揃っています。 ([jaxopt.github.io][6])

> ただしこの refine は **P2の高度化**として、P0/P1はOptuna単独で十分回る設計にしておくのが安全です。

---

# 8) 物理妥当性を“測る／保つ”ために最低限必要な仕掛け（簡単に入る）

ブラックボックス係数を増やして“合わせる”だけなら簡単ですが、あなたは **物理妥当性も測りたい**。
このために、実装が重くならず効くものを優先順で挙げます。

## (A) 共有パラメータ vs 条件別パラメータの分離（最重要）

* 反応本質（例：(k_0, E_a, K_i, \alpha,\beta)）は **全条件で共有**
* 条件依存になりやすいもの（例：(\delta_\mathrm{eff})、アライメント微調整）は **nuisance** として弱く許す（または固定）

これだけで “条件ごとに係数が暴れて当たる” を大きく抑えられます。

## (B) Complexity penalty（有効パラメータ数・有効項数に罰則）

[
J_\text{complex} = \lambda_p N_\text{params} + \lambda_t N_\text{terms}
]

* 項ON/OFFが増えた場合の過学習を抑えます
* 実装が非常に軽い

## (C) Prior penalty（物理範囲を soft constraint）

例：正値制約・範囲制約を「変換＋弱い罰則」で入れるだけで十分効きます。
（厳密ベイズにしない）

---

# 9) Hydra/YAML設計（第三者が“設定だけで”指標/手法を差し替えられる形）

最小で次の3ファイルに分離すると運用が綺麗です。

## 9.1 `configs/opt/optimizer/optuna_tpe.yaml`

* sampler / pruner / n_trials / storage（SQLite推奨）等

例（概念）：

```yaml
optimizer:
  name: optuna
  sampler: tpe        # 変更したければ cmaes など
  pruner: percentile  # 例：PercentilePruner
  n_trials: 200
  n_jobs: 4
  storage: "sqlite:///runs/project/optuna/study.db"
```

TPE・CMA-ES・pruner・TrialPruned などは Optuna公式に記載があります。 ([Optuna][1])

## 9.2 `configs/opt/objective/mapfit_standard.yaml`

* loss_terms の list（enabled + weight）

```yaml
objective:
  mode: scalar
  terms:
    - {enabled: true,  name: data_misfit_map, weight: 1.0, params: {...}}
    - {enabled: true,  name: kpi_misfit,      weight: 0.2, params: {...}}
    - {enabled: true,  name: prior_penalty,   weight: 0.1, params: {...}}
    - {enabled: true,  name: complexity,      weight: 0.05, params: {...}}
    - {enabled: false, name: regime_penalty,  weight: 0.05, params: {...}}
```

## 9.3 `configs/opt/model_catalog/cvd_priority.yaml`

* 候補MS一覧と、それぞれの ParamSpace（bounds/transform/shared_scope）

```yaml
model_catalog:
  candidates:
    - name: MS-01_CVD_basic
      sim_model_set: MS-01
      params:
        - {name: log10_k0,  type: float, low: -8, high: 2,  transform: pow10, shared: global}
        - {name: Ea_kJmol,  type: float, low: 0,  high: 300, transform: identity, shared: global}
        - {name: delta_mm,  type: float, low: 0.2, high: 10, transform: mm_to_m, shared: per_condition}
    - name: MS-02_CVD_sat_inh
      sim_model_set: MS-02
      params:
        - {name: log10_k0, ...}
        - {name: beta, type: float, low: 0.5, high: 3.0, shared: global}
        - {name: K_A, type: float, low: 1e-6, high: 1e2, transform: log, shared: global}
```

---

# 10) P0/P1/P2 タスク（最適化側だけに絞って、現場導入しやすい順）

## P0（まず “回って、比較できて、モデル比較できる”）

1. `deposim_opt.dataset`：Observation（1D/2D+mask+weights）統一
2. `model_catalog`：候補MSを有限集合で管理（CVD中心：MS-01/02/03/04/05/06/07/13）
3. `objective`：LossTerm合成（data_misfit + complexity + prior）
4. `optuna_runner`：TPEで最適化、trial.user_attrにKPI/診断を保存
5. `pruning`：条件ループ途中で `trial.report` / `should_prune` / `TrialPruned` を入れる ([Optuna][7])
6. `report`：モデルランキング（best score + complexity + plausibility内訳）

**Decision（P0で止めるべき）**

* 係数の物理範囲（priorの範囲）をどう置くか（材料/プロセスで変わるのでYAML化）

## P1（ALD/複数条件で “外挿耐性” を上げる）

1. eval_protocol：条件ホールドアウト（train/val）を導入
2. objective：KPIの重み付け、残差構造（リング等）を optional term として追加
3. optimizer：CMA-ES/NSGA-IIなど sampler 切替を設定で可能に

   * NSGA-IIは Optuna samplerとして提供 ([Optuna][4])
4. model_catalog：ALD候補（MS-08/09/10/11/12）を追加

## P2（精密化：勾配・不確かさ）

1. post_refine：Optuna best を起点に、任意でJAXopt等の局所最適化（外部ステップ）

   * JAXopt の root finding（Bisection）や implicit diff の枠が参考になる ([jaxopt.github.io][6])
2. 不確かさ推定（上位モデルだけ）を別系統で実施（必要になってから）

---

# 11) ここまでの設計が「複雑にしないのに目的を達成」できる理由

* **モデル構造選択**は、無限のON/OFF探索ではなく **候補MS集合（ModelCatalog）**に閉じ込める
* 係数推定は **Optuna（黒箱）**で統一し、sampler/pruner差し替えは設定だけ
* 指標は **LossTerm合成**で、第三者が1ファイル追加＋YAML追記で拡張できる
* “物理妥当性”は、いきなりベイズにしなくても

  * 共有/条件別の分離
  * complexity penalty
  * prior penalty
  * 条件ホールドアウト
    の4点でかなり担保できる


[1]: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html?utm_source=chatgpt.com "optuna.samplers - TPESampler"
[2]: https://optuna.readthedocs.io/?utm_source=chatgpt.com "Optuna: A hyperparameter optimization framework — Optuna ..."
[3]: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html?utm_source=chatgpt.com "optuna.samplers.CmaEsSampler - Read the Docs"
[4]: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html?utm_source=chatgpt.com "optuna.samplers.NSGAIISampler - Read the Docs"
[5]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.TrialPruned.html?utm_source=chatgpt.com "optuna.TrialPruned — Optuna 4.7.0 documentation"
[6]: https://jaxopt.github.io/stable/root_finding.html?utm_source=chatgpt.com "Root finding — JAXopt 0.8 documentation"
[7]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?utm_source=chatgpt.com "optuna.trial.Trial — Optuna 4.7.0 documentation"
