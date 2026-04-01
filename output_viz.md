以下は、**今回の大幅改良（AIB‑ODE 1本化／role縮約 A/I/B／I・B≤1／B次数0/1固定／ODE主役／power‑law等は廃止）**に合わせて、
**物理的考察ができる“診断量（diagnostics）”**と、**結果把握のための可視化（plots）**、**出力内容・ファイル設計（outputs）**を、第三者（物理理論の専門家・半導体製造エンジニア）が迷わないように整理した検討案です。

---

# 1) 今回改良で「可視化・出力が担うべき役割」が変わった点

旧構成（MS多数、TC-root、power-law等）では、
「どの式を選んだか」「root収束したか」「見かけ次数は何か」などが主要診断でした。

今回の改良ではモデルが **AIB‑ODE統一コア**に絞られ、**“モデル選択”の中心は role割当と（少数の）整数次数の選択**になっています。したがって、可視化・出力が担うべき役割は次に移ります：

1. **Bが本当に必要か（AB/AIBが勝つのか）**を、誤解なく説明する
2. **Iが本当に必要か（AI/AIBが勝つのか）**を、誤解なく説明する
3. その判断が **過学習でない**こと（複雑さ罰則・上位候補の安定性・識別不能の警告）を示す
4. “当たった”だけでなく、**物理的にどのレジーム（輸送枯渇／サイト不足／反応律速）にいるか**を可視化する
5. ODE主役なので、**時間応答（CVD transient / ALD transient）**の理解に役立つ時系列出力が必要

---

# 2) 物理的考察に直結する「必須診断量」（AIB‑ODE専用）

AIB‑ODEの核を再掲すると（あなたの仕様に合わせた形）：

* 阻害（I）は準平衡のサイト占有で閉包：
  [
  \theta_*=\frac{1-\theta_A}{1+K_I C_{ref,I}},\quad
  f_I=\frac{1}{1+K_I C_{ref,I}}
  ]
* Aの表面濃度（境界層＋吸着/脱離で代数閉包）：
  [
  C_{s,A}=\frac{k_{m,A}C_{ref,A}+\Gamma_s k_{des}\theta_A}{k_{m,A}+\Gamma_s k_{ads}\theta_*^{m_{ads}}}
  ]
* Bは **m_B=0/1固定**。Bあり（m_B=1）のときの表面濃度：
  [
  C_{s,B}=\frac{k_{m,B}C_{ref,B}}{k_{m,B}+\Gamma_s,k_{rxn}\theta_A^{p_A}\theta_*^{p_*}/C_{B,scale}}
  ]
* 成膜イベント頻度：
  [
  r_{event}=k_{rxn}\theta_A^{p_A}\theta_*^{p_*}\left(\frac{C_{s,B}}{C_{B,scale}}\right)^{m_B}
  ]
* ODE：
  [
  \frac{d\theta_A}{dt}=k_{ads}C_{s,A}\theta_*^{m_{ads}}-k_{des}\theta_A-\nu_A r_{event}
  ]
* 膜厚：
  [
  \frac{dh}{dt}=\alpha_h\Gamma_s r_{event}
  ]

これを踏まえ、「物理考察に必要な診断量」を **“場（x,y[,t]）のマップとして出せる形”**で定義します。

---

## 2.1 必須フィールド（空間分布として出す）

### (A) 状態・濃度関連

* `theta_A(x,y[,t_end])`
* `theta_star(x,y[,t_end])`
* `CsA_over_CrefA = C_{s,A}/(C_{ref,A}+\epsilon)`
* `CsB_over_CrefB = C_{s,B}/(C_{ref,B}+\epsilon)`（Bあり候補のみ。Bなしは NaN で統一が扱いやすい）

**物理的意義**

* `Cs/Cref` が **境界層枯渇（輸送影響）**の最短の証拠
* `theta_star` が **サイト不足（阻害・飽和）**の最短の証拠

---

### (B) “Bが必要か”の物理診断（Bあり候補の必須）

* `phi_B(x,y)`：B側の輸送 vs 反応のレジーム指標（あなたが合意した定義でOK）
  [
  \phi_B=\frac{\Gamma_s k_{rxn}\theta_A^{p_A}\theta_*^{p_*}/C_{B,scale}}{k_{m,B}}
  ]

**目安**

* (\phi_B\ll1) ：(C_{s,B}\approx C_{ref,B})（Bは枯渇せず、B輸送は支配しにくい）
* (\phi_B\gg1) ：(C_{s,B}\ll C_{ref,B})（Bが枯渇し、B供給が効いている）

これが **AB/AIBが勝ったときに「本当にBが必要だった」**ことの説明に直結します。

---

### (C) “Iが必要か”の物理診断（Iあり候補の必須）

* `f_I(x,y) = 1/(1+K_I CrefI)`（空サイト低下率）

  * `f_I≈1`なら Iは実質効いていない（不要の疑い）
  * `f_I<<1`なら Iが強く効いている

---

### (D) 成膜速度と支配項

ODE主役のモデルでは、支配項を見ないと “当たった理由” が説明できません。

以下を **最終時刻（steadyなら定常近傍）**のマップとして保存するのを推奨します：

* `r_event(x,y)`
* `ads_term = k_ads*CsA*theta_star^m_ads`
* `des_term = k_des*theta_A`
* `rxn_sink = nu_A*r_event`
* `dhdt_nm_s = alpha_h*Gamma_s*r_event`（nm/s）

**有用性**

* どこが **吸着律速**か（ads_termが小さい）
* どこが **脱離支配**か（des_termが大きい）
* どこが **反応消費支配**か（rxn_sinkが大きい）
* どこが **サイト不足**か（theta_starが小さい）

を“見える化”できます。

---

## 2.2 数値解法の健全性診断（必須）

ODEを暗黙Euler＋二分法で解く以上、**ソルバ健全性を必ず出力**して、
「数値の破綻が物理に見えている」事故を防ぎます。

* `solver.n_steps`（CVD steadyなら t_proc/dt）
* `solver.total_bisect_iters` / `solver.mean_bisect_iters`
* `solver.bracket_fail_count`（根が挟めずフォールバックした回数）
* `solver.theta_clip_count`（0/1にクリップした回数）
* `nan_count` / `inf_count`（発生したら強warning）

これらを `metrics.json` にまとめ、reportに表示するのが良いです。

---

# 3) 可視化（Plots）：用途別の「必須セット」を固定する

“文字だけスライド禁止”の思想と同様に、**reportは必ず図が主役**であるべきです。
以下は **用途別に最低限必要な図セット**です。

---

## 3.1 共通（CVD/ALD、steady/transient すべて）

### (1) Thickness map（最重要）

* `h_nm` の2Dマップ（散布＋補間 or triangulation）
* ウェハ円を描画（半径、中心）

### (2) Residual map（測定がある場合）

* `h_sim - h_meas` の2D差分
* NU%だけでなく **差分の空間構造**を見る（リング、中心/外周、方位非対称）

### (3) Radial profile（中心→外周）

* `h_nm(r)` の平均/分位（mean, p10, p90）
* 測定も同じ処理で重ねる
  → これで“装置エンジニアがすぐ判断できる”形式になります。

### (4) Diagnostic maps（今回の改良で重要）

* `theta_A`
* `theta_star`
* `CsA_over_CrefA`
* （Bあり候補のみ）`CsB_over_CrefB` と `phi_B`
* （Iあり候補のみ）`f_I`

---

## 3.2 CVD steady（定常入力＋積算）

steadyは時間軸がないので、「レジームの説明」が重要になります。

推奨の追加図：

### (5) Term dominance map（支配項の領域分割）

例えば各点で

* `ads_term`
* `des_term`
* `rxn_sink`
  の最大項（または寄与比）を分類し、領域を色分けする。

これにより

* “外周はサイト不足”
* “中心はB枯渇”
* “中間は吸着律速”
  などを説明できます。

### (6) sanity plot：θの物理拘束

* θ_Aのヒストグラム（0〜1）
* クリップ率・フォールバック率も併記
  → パラメータが不自然だと端に張り付くので一目で分かります。

---

## 3.3 CVD transient / ALD transient（時間入力）

time seriesでは、「代表点の時系列」と「空間統計の時系列」の2系統が必須です。

### (1) 代表点の時系列（center / mid / edge）

各点で

* `theta_A(t)`
* `dh/dt(t)` or `h(t)`
* （Bありなら）`CsB/CrefB(t)` と `phi_B(t)`
* （Iありなら）`f_I(t)`

### (2) 空間平均・分位の時系列

* `mean(h(t))`, `p10`, `p90`
* `mean(theta_A(t))` など

### (3) “入力信号のプレビュー”

Fluentの `CrefA(t)`, `CrefB(t)`, `CrefI(t)` の時間波形（空間平均でもよい）

* ALDの場合は、A/Bパルスが見えていることが重要
  → ここが崩れるとALD解析自体が成立しません。

---

## 3.4 Opt（同化・モデル選択）専用の可視化

同化は「当たった」だけでなく、「どれが勝ったか」「識別できたか」を示す必要があります。

### (1) class_compare（A/AI/AB/AIB）

* 棒グラフ or 表で、各クラスの best score と Δ
* 罰則込みの score を表示（過学習防止の透明性）

### (2) ranking（topK）

* topK の `score` の落ち方（差が小さい＝識別不能の可能性）
* topK の `roles(A/I/B)` と `orders` を表で

### (3) role stability（非常に重要）

上位K候補で

* Aに選ばれたspeciesの頻度
* Iに選ばれたspeciesの頻度（None率も）
* Bに選ばれたspeciesの頻度（None率も）

→ “Bは必要だが、どれがBかは特定できない”が一発で分かります。

### (4) parameter distributions（Optuna試行の結果）

* `k_rxn`, `km_A`, `km_B`, `K_I` のヒストグラム（log軸推奨）
* 相関（散布）もあると理想
  ただし実装を複雑にしたくないなら、まずは1Dヒストだけで十分です。

---

# 4) 出力内容（Files / Schema）：迷子にならない固定設計

あなたの要望「yamlやディレクトリが増えて迷子を避けたい」に合わせ、**入口1つ＋run単位で完結**させます。

---

## 4.1 ディレクトリ規約（強制）

```
results/<project>/
  index.html                 # 常に入口。最新runと過去run一覧
  summary.json               # project全体の要約（最新bestなど）

  runs/<run_id>/
    config_resolved.yaml     # Hydra解決済み
    outputs/
      fields.npz             # 数値場（h, theta, ratios, phi_B…）
      metrics.json           # KPI/solver統計/物理統計
    tables/
      ranking.csv            # (optのみ) 候補ランキング
      class_compare.csv      # (optのみ) A/AI/AB/AIB比較
      topk_models.csv        # (optのみ) 上位Kのrole+orders+params+score
    plots/
      ...png                 # ルール固定
    report.html              # runの詳細レポート
```

---

## 4.2 `fields.npz` に入れる推奨キー（今回改良に最適化）

### forward（sim）最小セット

* `xy`（必須）
* `h_nm`
* `theta_A`
* `theta_star`
* `CsA_over_CrefA`
* `r_event`
* `dhdt_nm_s`

### Bあり候補なら追加

* `CsB_over_CrefB`
* `phi_B`

### Iあり候補なら追加

* `f_I`

### measurementありなら追加

* `residual_nm`
* `h_meas_nm`（sim座標上に整合したもの）

> Bなし/Iなしのときは、該当キーを出さないのではなく **NaNで出す**方が、後処理が楽です（plot側が条件分岐しなくて良い）。
> ただし保存サイズが気になるなら “キー省略”でもOK。その場合 report生成側で分岐が必要になります。

---

## 4.3 `metrics.json`（物理＋数値＋品質ゲート）

最低限、以下を固定キーで出すとレビューしやすいです。

* `kpi.nu_percent`
* `kpi.center_edge_delta_nm`
* `solver.bracket_fail_count`
* `solver.theta_clip_count`
* `solver.mean_bisect_iters`
* `physics.mean_CsA_over_CrefA`
* `physics.min_CsA_over_CrefA`
* `physics.mean_theta_star`
* `physics.mean_phi_B`（Bあり）
* `physics.mean_f_I`（Iあり）
* `quality.nan_count`, `quality.inf_count`

---

# 5) “今回の改良ならでは”の考察テンプレ（report構成案）

report.html を毎回同じ構成にすると、装置側レビューが速くなります。

## 5.1 forward run（sim）の章立て

1. **Run summary**：roles / orders / params / time_mode / t_proc / dt
2. **Inputs preview**：CrefA/CrefB/CrefI マップ or 時系列
3. **Thickness results**：h map + radial + stats
4. **State & transport diagnostics**：theta, Cs ratios, phi_B, f_I
5. **Term dominance**：ads/des/rxn の寄与
6. **Solver health**：bracket fail、clip率、nanなど

## 5.2 opt run（fit）の章立て（追加）

7. **Class compare**：A/AI/AB/AIB の best
8. **TopK candidates**：roles・orders・score・主要パラメータ
9. **Role stability**：A/I/Bの頻度（None率含む）
10. **Parameter distributions**：試行分布（最小はヒスト）

---

# 6) 実装を複雑にしないための“優先順位”（P0/P1/P2相当）

あなたの「複雑化させず、第三者が直せる」を守るため、可視化・出力も段階導入が安全です。

## P0（最短で現場に刺さる最低限）

* h map / residual map / radial profile
* theta_A / theta_star / CsA_ratio
* (Bあり) phi_B / CsB_ratio
* (Iあり) f_I
* ranking.csv / class_compare.csv（opt）
* index.html固定入口

## P1（物理説明能力を上げる）

* ads/des/rxn 支配項マップ
* 代表点の時系列（transient）
* role stability 図
* parameter hist 図

## P2（研究者向けの深掘り）

* topKアンサンブル予測：mean/stdの厚み不確かさマップ
* “識別不能”自動判定（score gapが小さい等）
* 追加のレジーム分類（簡易Da_A等）

---

# 7) 追加で入れるべき「品質ゲート」（誤った物理解釈を防ぐ）

可視化だけでなく、**自動で警告を出す**と現場導入が安定します。

* `theta_A` が 0/1 に張り付き領域が大きい → **“パラメータが過大/過小の疑い”**
* `CsA/CrefA` が極端に小さい領域が広い → **“輸送枯渇支配、kmかΓ_sの見直し”**
* AB/AIB が勝つが `phi_B<<1` ばかり → **“Bはスコア上必要に見えるが物理的寄与が薄い（過学習疑い）”**
* AI/AIB が勝つが `f_I≈1` ばかり → **“Iは形式的で実質不要（過学習疑い）”**
* bracket_fail_count が多い → **“dt大きすぎ or パラメータ異常。結果解釈前に数値健全性を確認”**

これらを report 上部に **警告ボックス**として出すのが有効です（実装は簡単）。

---
