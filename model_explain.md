以下は、**（これまで議論してきた）基盤に含めるもの＝「実装済み／実装対象（P0/P1/P2で確定）」＋追加検討した理論**を **すべて**含めて、第三者（物理研究者／数値計算研究者／半導体装置エンジニア）が

* 何をどの用途で使うべきか
* その方程式が何を仮定しているか
* メリット／デメリット（同定性・外挿性・数値安定性）
* どのモデルセット（MS）で採用されるか
* どの分岐（選択）で切り替えるか

を理解できるように整理したものです。

---

# 0. まず前提：この基盤の「共通コア」と「モデルセット差分」

## 共通の目的（全MS共通）

* CFDで得られるのは **z=z_ref** の参照面での

  * 濃度 (C_{\mathrm{ref},i}(x,y,t))
  * 速度代表量 (|u|(x,y,t))
  * 温度 (T(x,y,t))（スカラーでも可）
    のみ（厳密組成・表面反応の全機構は無理）
* そこで **輸送–反応結合（TC）**＋**縮約表面反応運動論（Reduced surface kinetics）** で

  * **ウェハ上膜厚2D分布 (h(x,y))** を予測
  * 多次数・阻害・飽和・ALD phase も扱える枠を用意
* さらに実務で必須の

  * **実測2Dマップ比較（座標整合）**
  * **KPI/レポート**
  * **不正な式の組合せを止める互換性バリデータ**
    を共通スタックとして標準装備

## モデルセット（MS-01〜15）の役割

* “物理の差分”は主に以下の選択で生まれます：

  1. **InputTransform**（参照面入力補正：気相ロスなど）
  2. **MassTransfer**（(k_m) のモデル：停滞膜／回転ディスク／低圧ブリッジ／Stefan補正）
  3. **Surface kinetics core**（Power-law／分母形／LHHW／ER／sticking）
  4. **Surface state**（被覆率θ、被毒θI、核生成η…：none/代数/ODE/定常解）
  5. **Net model**（dep only / dep–etch–loss）
  6. **Time mode**（steady/transient/phases）

---

# 1. 共通スタック（全モデルセットで“標準”にするもの）

モデルセット差分の前に、**現場導入成功**のために全MSに共通で入れる（べき）ものをまとめます。

| 共通スタックID                     | 内容                                                      | 数式/定義（代表）                                      | メリット                   | デメリット/注意                            | 主要用途            |
| ---------------------------- | ------------------------------------------------------- | ---------------------------------------------- | ---------------------- | ----------------------------------- | --------------- |
| **CS-01 Validator**          | 互換性バリデータ（排他・必要条件・時間モード）                                 | `requires/excludes/governing_class/time_modes` | “あり得ない組合せ”を事前に排除（事故防止） | ルール設計が必要（最初に丁寧に）                    | 研究/現場の両方で必須     |
| **CS-02 Diagnostics**        | 物理診断：(C_s/C_{ref})、見かけ次数 (n_\mathrm{app})、Da proxy、反復統計 | (n_\mathrm{app}=\partial\ln r/\partial\ln C_s) | 物理妥当性の説明ができる           | 数式があるモデルのみ (n_\mathrm{app}) を厳密に出せる | 研究レビュー、外挿の安全性確認 |
| **CS-03 Regime numbers**     | レジーム推定（Re/Sc/Pe/Kn の近似）                                 | 例：(Re=\rho UL/\mu), (Sc=\nu/D)                 | km/低圧モデル選択の根拠が出る       | 入力が足りない場合は推定＋注記                     | “どのモデルが妥当か”説明   |
| **CS-04 MeasurementAdapter** | 実測2D膜厚マップと座標整合（dx,dy,rot,mask,補間）                       | ((x',y')=R(\phi)(x-x_0,y-y_0))                 | 実務で最頻のズレ要因を排除          | 測定座標仕様が未決だと仮決めが必要                   | 現場導入の最大の壁       |
| **CS-05 KPI/Metrics**        | NU%、C/E差、ring統計、規格外面積率など                                | NU%例：((\max-\min)/(2\bar h)\cdot100)           | “使える指標”で比較・DOEが可能      | 指標定義を固定しないと比較がぶれる                   | 装置エンジニア向け       |
| **CS-06 Report**             | index.html固定入口、図・表・KPI・差分マップ                            | —                                              | ファイル迷子防止、レビュー容易        | HTML/プロット整備が必要                      | DOE/比較/説明資料     |

> 以降の MS は「物理モデル差分」中心にまとめますが、**実行時は上のCSが必ず併走**します。

---

# 2. モデルセット（用途）別の概要テーブル（MS-01〜MS-15）

## 記号（全MS共通）

* **Cref**：CFD参照面 (z=z_\mathrm{ref}) の濃度 (C_{\mathrm{ref},i}(x,y,t))
* **Cs**：表面近傍濃度（TC結合で決める）
* **km**：物質移動係数 (k_{m,i}(x,y,t))
* **R**：反応進行度（面積反応速度、rootの未知数）
* **θ**：表面状態（被覆率など）
* **S(x,y)**：パターンローディング（有効表面積倍率など）

---

## 2.1 MS一覧テーブル（概要）

| MS-ID（用途）                      | 主要方程式セット（要約）               | 何を近似しているか（各項）                                                                     | メリット                           | デメリット/注意           | 典型用途                           | Ref key |
| ------------------------------ | -------------------------- | --------------------------------------------------------------------------------- | ------------------------------ | ------------------ | ------------------------------ | ------- |
| **MS-01 CVD_steady_basic**     | TC＋Power-law（多次数）＋steady積算 | (J_i=k_m(C_{ref}-C_s))、(C_s=C_{ref}-\nu R/k_m)、(F(R)=R-r(C_s)=0)、(h=h_0+\dot h t) | 最短で回る、DOE向き、拘束（Cs≥0）を保ちやすい     | 飽和/阻害/競合が強いと次数漂流   | CVD初期モデル、狭い条件で当てる              | R1      |
| **MS-02 CVD_steady_sat_inh**   | MS-01＋分母形（飽和/阻害）           | (r=k\frac{\prod (K C)^\alpha}{(1+\sum K C)^\beta})                                | 見かけ次数遷移（一次→零次・負次数）を自然表現        | パラメータ増、同定性が課題      | 生成物阻害、飽和、均一化ノブ                 | R2      |
| **MS-03 CVD_rotating_disk_km** | MS-01/02＋回転ディスクkm（ω依存）     | (k_m\sim C_k D^{2/3}\omega^{1/2}\nu^{-1/6})                                       | 回転single waferの輸送差を反映          | ω=0で不適切→ガード必須      | 回転装置で中心-外周差                    | R3      |
| **MS-04 CVD_gas_phase_loss**   | MS-01/02＋入力補正（気相ロス）        | (C_{eff}=C_{ref}e^{-k_g t_{res}})                                                 | 上流分解/気相反応を少数パラメータで吸収→外挿安定      | (t_{res}) 推定が要る    | 温度/流量/滞留でズレるCVD                | R4      |
| **MS-05 lowP_bridge**          | MS-01/02＋低圧ブリッジ            | (1/D_{eff}=1/D_m+1/D_K)、(k_m=D_{eff}/\delta)                                      | 低圧で連続体拡散だけだと崩れるのを抑制            | D_K推定/適用判定が要る      | 低圧CVD/ALDの輸送補正                 | R5      |
| **MS-06 pattern_loading**      | MS-01/02＋S(x,y)            | (C_s=C_{ref}-\nu R S/k_m)                                                         | 実測2Dムラ（パターン密度）を説明しやすい、root構造維持 | S(x,y)入力が要る（または推定） | microloading / pattern loading | R6      |
| **MS-07 dep_etch_loss**        | dep–etch–loss（多チャネル）       | (\dot h=\alpha r_{dep}-\alpha_e r_{etch}-r_{loss})                                | 競合反応・符号反転を表現                   | 同定性が悪化（切り分け実験推奨）   | PECVD/同時エッチ/損失                 | R1      |
| **MS-08 ALD_basic_coverage**   | phases＋被覆率θ（ODE/代数）        | (\dot\theta=A(C_s,T)(1-\theta)^m-B\theta)、(\dot h=\alpha r(\theta))               | ALDの核（自己終端、露光依存）               | stiff化、purge無視でズレ  | ALD基本、GPCの露光依存                 | R7      |
| **MS-09 ALD_purge_residual**   | MS-08＋purge残留              | purge中 (C(t)=C_0e^{-t/\tau})                                                      | purge不完全→CVD混入を表現              | τが条件依存する場合あり       | purge時間依存が強いALD                | R7      |
| **MS-10 ALD_sticking_flux**    | MS-08＋分子運動論フラックス＋sticking  | (\Gamma=\alpha p/\sqrt{2\pi m k_BT})、(r=s(\theta)\Gamma)                          | 低圧・衝突律速を自然に表現                  | Cs→p変換、s(θ)同定      | 低圧ALD/ラジカル衝突                   | R8      |
| **MS-11 incubation/JMAK**      | 核生成/初期遅れ状態η                | (Y=1-e^{-Kt^n}) 又は (\dot\eta=kC^n(1-\eta))、(\dot h=\eta r)                        | 初期だけ合わない問題を救う                  | 現象論で誤解釈注意          | 初期成長遅れ、基板依存                    | R9      |
| **MS-12 poisoning_state**      | 被毒状態θI＋反応抑制                | (\dot\theta_I=k_{ads}C_I(1-\theta_I)-k_{des}\theta_I)、(r=r_0(1-\theta_I)^m)       | 分母形より履歴（遅れ/回復）を表現              | stiff化、データ要求       | 生成物被毒、長時間ドリフト                  | R10     |
| **MS-13 LHHW/ER_mech**         | 機構寄り（LHHW/ER）              | (r\propto \frac{K_AC_AK_BC_B}{(1+\sum KC)^m})、(r\approx kP_B\theta_A)             | 負次数/競合を物理で説明                   | パラメータ多、簡約が必要       | 次数変化を物理で説明                     | R2,R11  |
| **MS-14 Stefan_flow_corr**（任意） | Stefan流補正                  | (N_A=-D,dC/dz+y_AN_T)                                                             | 強消費時の輸送非線形を補正                  | 適用範囲限定、複雑化         | 強消費CVD等                        | R12     |
| **MS-15 smoothing_PDE**（任意）    | 膜厚平滑化PDE                   | (\partial_t h=R_{dep}-\kappa\nabla^2h)（簡略）                                        | “ムラが残らない”形状緩和                  | PDE導入で重い/同定難       | 高温表面拡散支配                       | R13     |

---

## 2.2 参考文献（URL付き）Ref key 一覧

※URLはルール上 **コードブロック内**にまとめます（表には直書きしません）。

```text
R1: 反応＋質量移動（輸送律速/反応律速の整理、TC結合の基本）
https://public.websites.umich.edu/~elements/5e/14chap/Fogler_Web_Ch14.pdf

R2: LHHW/分母形（競合吸着・阻害・見かけ次数の遷移）
https://www.sciencedirect.com/science/article/pii/S006980400480013X/pdf

R3: 回転ディスク（RDE）輸送相関の概説（Levichスケール）
https://pineresearch.com/support-article/rotating-disk-electrode-rde-theory/

R4: CVDで気相反応＋表面反応を含む数理モデル例（Coltrin系の文書）
https://bdt.semi.ac.cn/download/0.4862375146410606.pdf

R5: 低圧/Knudsen/Bosanquetを含む輸送・ALDの不完全被覆モデル論文例
https://www.sciencedirect.com/science/article/pii/S0038110122003562

R6: パターン依存 microloading の例（JVST B）
https://pubs.aip.org/avs/jvb/article/23/6/2340/945287/Pattern-dependent-microloading-and-step-coverage

R7: ALD（露光/パージを含む）解析モデル（Muneshwar et al. 2018）
https://pubs.aip.org/aip/jap/article-pdf/doi/10.1063/1.5044456/15216502/095302_1_online.pdf

R8: Hertz–Knudsen（分子運動論フラックス）の説明
https://en.wikipedia.org/wiki/Hertz%E2%80%93Knudsen_equation

R9: Avrami/JMAK（核生成・相変態の現象論式）
https://en.wikipedia.org/wiki/Avrami_equation

R10: 表面反応機構・被毒等の議論例（NREL）
https://docs.nrel.gov/docs/fy25osti/89378.pdf

R11: Eley–Ridealを含む速度式議論例（OSTI）
https://www.osti.gov/servlets/purl/1977248

R12: Stefan-Maxwell/移動現象系（教科書例）
https://ia800401.us.archive.org/8/items/B.KDutta/B.K%20Dutta_text.pdf

R13: 表面拡散・平滑化（Mullins系）に関連する論文例
https://arxiv.org/pdf/2210.15797
```

---

# 3. 詳細：各「理論モジュール」ごとの説明（式・項・メリデメ・用途・MS対応）

ここからは **モデルセットの部品（モジュール）**ごとに整理します。
（実装では `InputTransform / Driver / MassTransfer / RateCore / Modifiers / StateClosure / NetModel / SurfaceSolver / TimeIntegrator / PostProcess / Diagnostics` に対応）

---

## 3.1 TC結合（輸送–反応結合）＋進行度Rの1変数root（SurfaceSolverの核）

### 基本式（全MSの“接着剤”）

参照面→表面の物質移動（縮約）：
[
J_i = k_{m,i}(C_{\mathrm{ref},i}-C_{s,i})
]

支配反応が1本に縮約できる（まずCVD本命のMVP）とき：
[
J_i=\nu_i R
\Rightarrow
C_{s,i}=C_{\mathrm{ref},i}-\frac{\nu_i}{k_{m,i}}R
]

反応速度式 (r({C_{s,i}},T,z)) と整合する (R) を解く：
[
F(R)=R-r(C_s(R),T,z)=0
]

非負性を守る上限（物理拘束）：
[
0\le R \le R_{\max}
===================

\min_i \left(\frac{k_{m,i}C_{\mathrm{ref},i}}{\nu_i}\right)
]

### 何を近似しているか

* 参照面（z_ref）から表面までの“詳細境界層”を **km 1つ**に集約
* 反応（表面）と輸送（境界層）が同時に満たされるように Cs を決める

### メリット

* bracketing（bisection等）で **収束保証**が作りやすい
* Cs≥0 の物理拘束を守れる
* CVD/ALD/etch とも共通で使える（R1）

### デメリット/注意（重要）

* **負次数・強阻害・多チャネル**を入れると (F(R)) の単調性が崩れ、根が複数になる可能性
  → そのため実装では

1. **単調性チェック**
2. **区間分割rootフォールバック**
3. さらに必要なら **多変数（Cs直接）**へ昇格
   の段階戦略が必要（これが Validator とセット）

### 主に使うMS

MS-01〜MS-13（ほぼ全部）

---

## 3.2 MassTransfer（kmモデル）

### (A) Stagnant film（有効停滞膜）

[
k_m=\frac{D}{\delta_{\mathrm{eff}}}
]

* (D)：拡散係数（定数 or T依存 or 混合則）
* (\delta_{\mathrm{eff}})：有効境界層厚（校正・同化候補）

**メリット**

* 回転あり/なし両対応
* 実装が軽い
* z_refの違いを (\delta_{\mathrm{eff}}) に吸収しやすい

**デメリット**

* 流れ場差を完全には反映できない（kmが黒箱寄り）

**使うMS**：MS-01/02/04/06/07/08/09/11/12/13

---

### (B) Rotating disk correlation（回転ディスク相関）

代表スケール（係数は装置定義に依存）：
[
k_m \sim C_k D^{2/3}\omega^{1/2}\nu^{-1/6}
]
**重要：ω=0 のときは無効**
→ `guard.mode=error` か `fallback=stagnant_film` を必須化

**メリット**

* 回転single waferの輸送差を物理的に反映（R3）

**デメリット**

* Ckが装置固有になりやすい（同化対象にするのが現実的）

**使うMS**：MS-03

---

### (C) Low‑P bridge（Bosanquet/Knudsen混合）

[
\frac{1}{D_{\mathrm{eff}}}=
\frac{1}{D_m}+\frac{1}{D_K}
,\quad
k_m=\frac{D_{\mathrm{eff}}}{\delta_{\mathrm{eff}}}
]

**メリット**

* 低圧で “連続体拡散だけ” の破綻を抑える（R5）

**デメリット**

* (D_K) の推定や適用判定が必要（入力不足なら推定＋注記）

**使うMS**：MS-05/10（低圧ALDで特に効く）

---

### (D) Stefan flow correction（任意）

多成分で強消費時の輸送非線形：
[
N_A=-D\frac{dC_A}{dz}+y_A N_T
]
**メリット**：強消費でのズレを減らす
**デメリット**：適用範囲が狭く、複雑化（R12）

**使うMS**：MS-14（任意）

---

## 3.3 Surface kinetics core（RateCore：排他選択）

ここは **「同じ現象を二重計上しない」ために core は1つだけ**（排他）です。
（Power-lawとLangmuirを同時ONにしない、というあなたの懸念への構造的回答）

### (A) Power‑law（一般化パワー則、多次数）

[
r=k(T)\prod_i C_{s,i}^{n_i}
,\quad
k(T)=k_0\exp(-E_a/RT)
]
**項の意味**

* (n_i)：見かけ次数（正/負/分数も許容）
* Arrhenius：温度依存

**メリット**

* 最短で当てられる、同化の初期モデルに強い

**デメリット**

* 飽和/阻害/競合を次数に押し込めるとパラメータが条件で漂流

**使うMS**：MS-01（基礎）

---

### (B) Saturation/Inhibition（分母形：飽和・阻害）

[
r=
k(T),
\frac{\prod (K_i C_{s,i})^{\alpha_i}}
{\left(1+\sum_j K_j C_{s,j}\right)^{\beta}}
]
**ポイント**

* 高濃度で飽和（一次→零次）
* 阻害種が分母に入ると負の見かけ次数が自然に出る（R2）

**メリット**

* 次数遷移や負次数を物理に沿って説明できる
* 外挿が壊れにくい

**デメリット**

* Kやβが増え同定性が落ちやすい（診断が必須）

**使うMS**：MS-02（本命の次）

---

### (C) LHHW / competition（機構寄り分母）

[
r\propto
\frac{K_AC_A K_BC_B}{(1+\sum K C)^m}
]
**メリット**：競合吸着で次数変化を説明しやすい
**デメリット**：同定が難しい（簡約が必要）

**使うMS**：MS-13

---

### (D) ER-like（Eley–Rideal型縮約）

[
r \approx k,P_B,\theta_A
]
**メリット**：片方が吸着、片方が衝突という機構を縮約表現
**デメリット**：θの閉包（代数/ODE）設計が必要（R11）

**使うMS**：MS-13（派生）

---

### (E) sticking + 分子運動論フラックス（低圧ALD/ラジカル）

Hertz–Knudsen：
[
\Gamma=\alpha\frac{p}{\sqrt{2\pi m k_B T}}
,\quad
r=s(\theta)\Gamma
]
**メリット**

* 低圧で“衝突で決まる”挙動を自然に再現（R8）

**デメリット**

* Cs→p変換、sticking係数やθ依存の同定が必要

**使うMS**：MS-10

---

## 3.4 State（被覆率/被毒/核生成）と closure（none / 代数 / ODE / 定常解）

### (A) none（状態なし）

* CVDの初期モデル（MS-01/02/03/04/06）で標準

### (B) 代数閉包（準平衡）

例：Langmuir等温吸着（平衡）：
[
\theta_A=\frac{K_A C_{s,A}}{1+K_A C_{s,A}}
]

* ODEを回さずに飽和を表現できる
* “Power-lawの低濃度極限”も自然に含む

### (C) ODE（動力学：ALDの核）

[
\frac{d\theta}{dt}=A(C_s,T)(1-\theta)^m-B(C_s,T)\theta
]
**メリット**：phase依存・自己終端を表現（R7）
**デメリット**：stiff化しやすい→解法選択が必要

**使うMS**：MS-08/09/10

### (D) purge残留（Driver側で扱う：stateではなく入力時間変化）

[
C(t)=C_0 e^{-t/\tau}
]
**使うMS**：MS-09

### (E) poisoning（被毒）

[
\frac{d\theta_I}{dt}=k_{ads}C_I(1-\theta_I)-k_{des}\theta_I
,\quad
r=r_0(1-\theta_I)^m
]
**使うMS**：MS-12（R10）

### (F) incubation / JMAK（初期遅れ）

[
Y=1-e^{-Kt^n}
\quad\text{or}\quad
\frac{d\eta}{dt}=kC^n(1-\eta)
,\ \dot h=\eta r
]
**使うMS**：MS-11（R9）

### (G) steady_state closure（重要：steadyでも状態を使う）

* steady計算でODE stateを使いたい場合は
  [
  0=g(C_s,\theta,T)
  ]
  を解いて **定常解**として閉包する
* これを明示しないと「steadyなのにODEを回す」など矛盾が出る
  → Validator で `closure_mode` を必須にするのが安全

---

## 3.5 InputTransform（参照面入力補正：上流現象を吸収）

### 気相ロス（gas phase loss）

[
C_{eff}=C_{ref}\exp(-k_g t_{res})
]
**メリット**

* 気相反応/上流分解/壁反応でウェハ到達量が減るのを吸収し、外挿安定（R4）

**デメリット**

* (t_{res}) や (k_g) の推定が必要（CFD補助があると強い）

**使うMS**：MS-04

---

## 3.6 Pattern loading（S(x,y)）

[
C_{s,i}=C_{ref,i}-\frac{\nu_i}{k_{m,i}}R,S(x,y)
]
**メリット**

* パターン密度差によるムラを説明しやすい（R6）
* しかも未知数Rは増やさず root構造を維持

**デメリット**

* S(x,y) が必要（設計情報/推定）

**使うMS**：MS-06

---

## 3.7 NetModel（dep only / dep–etch–loss）

dep only：
[
\frac{dh}{dt}=\alpha r
]

dep–etch–loss：
[
\frac{dh}{dt}=\alpha r_{dep}-\alpha_e r_{etch}-r_{loss}
]
**メリット**：競合反応や符号反転に対応
**デメリット**：同定性が悪化しやすい（切り分け条件が必要）

**使うMS**：MS-07

---

## 3.8 Time integration（steady / transient / phases）

* **steady（CVD本命）**
  [
  h=h_0+\dot h,t_{proc}
  ]
* **transient（入力が時間変化）**
  [
  h(t+\Delta t)=h(t)+\dot h(t)\Delta t
  ]
* **phases（ALD）**：phaseごとに入力/driverを切替し、θとhを更新

---

## 3.9 PostProcess（任意：平滑化PDE）

簡略：
[
\partial_t h = R_{dep}-\kappa\nabla^2 h
]
**メリット**：形状緩和を表現
**デメリット**：PDEで重く、同定も難しい（R13）

**使うMS**：MS-15（任意）

---

# 4. Mermaid：用途ごとの「モデル選択」＋内部の「理論分岐」図

ユーザー要望どおり、2つ用意します：

1. **用途（現場の状況）→ MS選択**の分岐（意思決定ツリー）
2. **コード内部での方程式分岐**（Pipeline＋Core/State/Modifier/Validator）

---

## 4.1 用途（状況）からMSを選ぶ分岐（意思決定ツリー）

```mermaid
graph TD
  S[Start: Choose ModelSet] --> P{Process?}
  P -->|CVD continuous| CVD[CVD]
  P -->|ALD phases| ALD[ALD]

  %% CVD branch
  CVD --> C1{Need steady only?}
  C1 -->|yes| C2{Strong saturation/inhibition suspected?}
  C1 -->|no (transient)| C2t[Use MS-01/02 core + transient integrator]

  C2 -->|no| MS01[MS-01: TC + Power-law]
  C2 -->|yes| MS02[MS-02: TC + Sat/Inh (rational)]

  MS01 --> C3{Rotation (omega > 0)?}
  MS02 --> C3
  C3 -->|yes| MS03[MS-03: + Rotating-disk km]
  C3 -->|no| C4{Upstream gas-phase loss suspected?}

  MS03 --> C4
  C4 -->|yes| MS04[MS-04: + gas_phase_loss transform]
  C4 -->|no| C5{Low pressure / rarefaction?}

  MS04 --> C5
  C5 -->|yes| MS05[MS-05: + Bosanquet/Knudsen bridge]
  C5 -->|no| C6{Pattern loading dominant?}

  MS05 --> C6
  C6 -->|yes| MS06[MS-06: + S(x,y)]
  C6 -->|no| C7{Dep-etch-loss competition?}

  MS06 --> C7
  C7 -->|yes| MS07[MS-07: dep-etch-loss channels]
  C7 -->|no| C8{Need mechanistic explanation of order changes?}

  MS07 --> C8
  C8 -->|yes| MS13[MS-13: LHHW/ER core]
  C8 -->|no| C9{Extreme consumption -> Stefan correction?}
  C9 -->|yes| MS14[MS-14: Stefan flow correction]
  C9 -->|no| C10{Need morphology smoothing?}
  C10 -->|yes| MS15[MS-15: smoothing PDE]
  C10 -->|no| DoneCVD[Done: choose MS]

  %% ALD branch
  ALD --> A1{Need basic self-limiting?}
  A1 -->|yes| MS08[MS-08: phases + coverage]
  A1 -->|no| MS08

  MS08 --> A2{Purge dependence / residual suspected?}
  A2 -->|yes| MS09[MS-09: + purge_decay driver]
  A2 -->|no| A3{Low pressure / collision-limited?}

  MS09 --> A3
  A3 -->|yes| MS10[MS-10: sticking_flux core]
  A3 -->|no| A4{Incubation / initial delay?}

  MS10 --> A4
  A4 -->|yes| MS11[MS-11: incubation state]
  A4 -->|no| A5{Poisoning / drift?}

  MS11 --> A5
  A5 -->|yes| MS12[MS-12: poisoning state]
  A5 -->|no| DoneALD[Done: choose MS]
```

---

## 4.2 コード内部の方程式分岐（Validator + Core/State/Modifier を含む詳細版）

```mermaid
graph TD
  A[Start: load Hydra YAML] --> MS{model_set selected?}
  MS --> V[Preflight Validator: requires/excludes/time_modes]
  V -->|invalid| STOP[STOP with clear message]
  V -->|ok| D[Domain + ReferencePlane z_ref]
  D --> I[Load Inputs: Cref_i(x,y,t), U, T, scalars, optional S(x,y)]

  %% transforms
  I --> X{InputTransforms list}
  X -->|none| DR{Drivers list}
  X -->|gas_phase_loss| X1[Ceff = Cref * exp(-kg*t_res)]
  X -->|others| Xn[other transforms]
  X1 --> DR
  Xn --> DR

  %% drivers
  DR -->|none| KM{MassTransfer km}
  DR -->|purge_decay| DR1[In purge phase: C(t)=C0*exp(-t/tau)]
  DR -->|schedules| DR2[scalar/field schedules]
  DR1 --> KM
  DR2 --> KM

  %% transport
  KM -->|stagnant_film| KM1[km = D_eff / delta_eff]
  KM -->|rotating_disk| KM2[km ~ Ck*D^(2/3)*omega^(1/2)*nu^(-1/6)]
  KM -->|cfd_provided| KM3[km from CFD]

  KM2 --> OMG{omega == 0?}
  OMG -->|yes| G{guard.mode}
  G -->|error| STOP
  G -->|fallback| KM1
  OMG -->|no| KMo[ok]

  KM1 --> DE{Diffusivity model}
  KM3 --> DE
  KMo --> DE

  DE -->|constant D| DE1[D_eff = Dm]
  DE -->|Bosanquet| DE2[1/D_eff = 1/Dm + 1/DK]
  DE -->|Stefan corr| DE3[Stefan correction enabled]

  DE1 --> SOL[SurfaceSolve: TC coupling]
  DE2 --> SOL
  DE3 --> SOL

  %% surface solver
  SOL --> CORE{RateCore (exclusive)}
  CORE -->|powerlaw_terms| C1[r = k(T)*Π Cs^n]
  CORE -->|sat_inh_terms| C2[r = k(T)*Num/Den]
  CORE -->|LHHW| C3[r = numerator/(1+ΣKC)^m]
  CORE -->|ER_like| C4[r ~ k*P*theta]
  CORE -->|sticking_flux| C5[Gamma=alpha*p/sqrt(2*pi*m*kB*T); r=s(theta)*Gamma]

  %% state closure
  C1 --> ST{State closure}
  C2 --> ST
  C3 --> ST
  C4 --> ST
  C5 --> ST

  ST -->|none| MOD{RateModifiers list}
  ST -->|algebraic| STa[theta = f(Cs,T)]
  ST -->|dynamic_ode| STo[dtheta/dt = g(Cs,theta,T)]
  ST -->|steady_state| STs[solve 0 = g(Cs,theta,T)]
  STa --> MOD
  STo --> MOD
  STs --> MOD

  %% modifiers (multiplicative)
  MOD -->|none| NET[After modifiers]
  MOD -->|pattern_loading| M1[Cs uses S(x,y)]
  MOD -->|poisoning| M2[r *= (1-thetaI)^m]
  MOD -->|incubation| M3[r *= eta]
  MOD -->|others| Mn[other modifiers]
  M1 --> NET
  M2 --> NET
  M3 --> NET
  Mn --> NET

  %% TC root
  NET --> RDEF{TC root unknown}
  RDEF --> RR[Unknown: R; Cs_i = Ceff_i - nu_i*R/km_i (±S)]
  RR --> F[F(R)=R - r(Cs(R),state,T)]
  F --> RS{Root solver}

  %% ★ここを修正：ラベル内の [0,Rmax] を禁止（] が構文を壊す）
  RS -->|bisection (vector)| BISECT[Bracket 0..Rmax]
  RS -->|hybrid| HYB[Bisection + Newton guard]
  RS -->|multi-root fallback| FB[interval split + pick rule]
  BISECT --> OUT[Get R, Cs, r, diagnostics]
  HYB --> OUT
  FB --> OUT

  %% net model
  OUT --> NM{NetModel}
  NM -->|dep_only| H1[dh/dt = alpha * r]
  NM -->|multi_channel| H2[dh/dt = alpha*r_dep - alpha_e*r_etch - r_loss]

  %% time integration
  H1 --> TM{Time mode}
  H2 --> TM
  TM -->|steady| T1[h = h0 + dhdt * t_proc]
  TM -->|transient| T2[integrate over time]
  TM -->|phases| T3[loop phases: update drivers/state/h]
  T1 --> PP{PostProcess list}
  T2 --> PP
  T3 --> PP

  %% postprocess
  PP -->|none| MEAS[MeasurementAdapter: align sim map to measurement map]
  PP -->|smoothing PDE| P1[h_t = R - kappa*laplacian(h)]
  P1 --> MEAS

  %% measurement + metrics + report
  MEAS --> KPI[Compute KPIs + residuals]
  KPI --> REP[Report: index.html + plots + summary.json]
  REP --> END[End]
```

---

# 5. 補足：第三者が“有用性を理解しやすい”ための読み方（短いガイド）

* **まずMS-01**（TC + powerlaw）で「輸送–反応結合の枠」が動くことを確認
* うまく当たらない原因が

  * 飽和/阻害 → **MS-02**
  * 回転輸送 → **MS-03**
  * 上流分解 → **MS-04**
  * 低圧 → **MS-05**
  * パターン密度 → **MS-06**
  * dep–etch競合 → **MS-07**
  * ALD phase → **MS-08〜10**
  * 初期遅れ/被毒 → **MS-11/12**
  * 物理説明が必要 → **MS-13**
    というふうに **“原因別に追加”**していくのが最も成功率が高いです。

