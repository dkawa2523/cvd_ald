## 実行トラック（1→2→3）

### シンプル化で精度向上する改善パッケージ（今回）
- 比較指標を絶対差のみから `relative_delta_flux_minus_free` へ拡張。
  - 目的: 膜厚スケール差で判定が歪むのを抑える。
- `km_spread_ratio` は free-km固定値ではなく、flux-km実行時は `mean_km_A_flux_km` を優先して評価。
  - 目的: 「比較モードなのに比較情報を使わない」評価バイアスを除去。
- 代表値集計を単純平均から `area_weights_mm2` を使った面積重み付き平均へ統一。
  - 目的: 点群密度の偏りによる評価バイアス低減。
- `benchmark_wafer2d.py` の重複処理を共通化。
  - `masked mean` の一本化、追加physviz画像登録を共通関数化。
  - ケースoverride組立・ケース実行を関数化して分岐重複を削減。
  - 目的: ロジック重複を減らして保守時の不整合発生率を下げる。

### 削減・共通化対象（今回）
- 削減: `_masked_nanmean` / `_masked_nanmean_abs` の重複実装を統合。
- 共通化: input画像2枚のplot record手組みブロックを共通関数化。
- 共通化: ケース実行の override 組み立てと spec 実行を関数化。
- 共通化: `fit_optuna` の fidelity-level 反復評価を `_evaluate_fidelity_levels` に集約。
- 維持: 既存AIBトレンド判定は mandatory判定として温存（互換性優先）。

### 1) 仕様固定化（本ファイル更新）
- Requirement->File->Symbol->Test を固定。
- 変更予算を固定（P0: 9 files/+280LOC, P1: 11 files/+360LOC）。
- 非肥大化方針: `pipeline` は provider 呼び出しに限定し、複雑分岐を `transport_provider.py` に隔離。

### 2) P0-01/P0-02 実装
- `flux_sink` 任意入力を Fluent contract に追加。
- `TransportProvider` 導入（`fit_scalar` 実装）。
- 既存挙動一致確認（default と `km_source=fit_scalar` の一致テスト）。

### 3) P1導入 + docs同期
- `from_cfd_flux_sink` 実装（`km = flux/(cref+eps)` + `gamma_km` + guards）。
- `fit_optuna` で flux-km時の探索空間制約（`km_A/B`禁止、`gamma_km`必須）。
- `benchmark_wafer2d` に入力分布・km/tau 可視化と flux比較モードを追加。
- `docs/inputs_fluent.md`, `docs/transport_km.md`, `docs/EVAL_PROTOCOL.md`, `benchmark_cvd.md` を同期。

---

以下は、あなたの現在の方針（AIB‑ODE一本化／role縮約A・I・B（I/B≤1）／B次数0/1固定／ODE主役／旧MS・power‑law群は排除）を前提に、
**「Fluent濃度＋（可能なら）壁面フラックス情報を使って“滞在時間＝輸送更新”の影響を取り込む」**ための P0 / P1 実装ロードマップを、実装タスクに直結する粒度でまとめたものです。

⸻

全体方針（P0/P1の役割分担）

P0：壊さずに “導入可否を判断できる土台” を作る
	•	挙動は変えない（kmは従来どおりフィット可能）
	•	ただし後でP1を安全に入れるために、
	•	入力スキーマ（fluxを受け取れる）
	•	kmの供給を抽象化（TransportProvider）
	•	「kmの推定が暴れてる／輸送が支配してる」を診断できる可視化
を先に整備する
	•	BM‑01を回して、“flux‑km導入が効きそうか”を証拠ベースで判断できるようにする

P1：Flux（できれば sink flux）→ km(x,y,t)拘束を導入し、最適化を“軽く・物理的に”する
	•	Fluentから wall-normal sink flux（またはkm field）を受け取り、
k_m^{CFD}(x,y,t)=\frac{J_{sink}(x,y,t)}{C_{ref}(x,y,t)+\epsilon}
で km 分布を生成
	•	最適化の自由度を kmそのもの → スケール係数 \gamma_{km} に縮約
k_m^{used}=\gamma_{km}^{(cond)}\,k_m^{CFD}
	•	BM‑01で、AIB‑ODEが示す物理診断（\phi_B,f_I,C_s/C_{ref}）と、外挿性能が改善することを確認

⸻

P0 実装ロードマップ（詳細）

ゴール：現状のAIB‑ODEを壊さず、P1を入れるための“差し込み口”と“判断材料”を揃える。

⸻

P0‑01：TransportProvider（km供給）を抽象化し、AIB‑ODE側を“kmの出どころ”から独立させる

目的
	•	現在の km_A, km_B が “ただのスカラー” でも “場（x,y,t）” でも同じAPIで扱えるようにする
	•	P1でkmをCFD由来に切替える準備

実装内容（推奨インタフェース）
	•	TransportProvider（抽象）
	•	get_km(role: "A"|"B", t_index: int | float, condition_id: str) -> ndarray[n_pts]
	•	返すのは 点ごとの km（n_pts）
	•	実装1：FitScalarKmProvider
	•	既存挙動：km_A（scalar or per_condition scalar）を n_pts にブロードキャスト

影響範囲
	•	AIB‑ODEのコア（反応計算）には変更を入れず、kmの取得部分だけ差し替え

受け入れ基準（AC）
	•	既存のBM‑01（flux無し）で 数値結果が一致（許容差はfloat誤差程度）
	•	km_map として出力されるのは、スカラーのタイル状でOK（見える化のため）

⸻

P0‑02：Fluent入力スキーマに “flux” を“任意入力”として追加（まだ使わない）

目的
	•	P1で必要になる I/O を先に通しておく
	•	運用上「fluxがあるプロジェクトだけON」にできるようにする

入力仕様（推奨）
	•	flux_sink：shape [nt, n_pts, n_species]（wall-normalのスカラー）
	•	unit：(C unit) * (m/s) になるように揃える（例 mol/m²/s と mol/m³）
	•	符号規約：表面へ向かう向きを正にする（ここを統一しないと事故る）

実装内容
	•	FluentLoader が flux_sink キーを見つけたら読み込む（無ければNone）
	•	InputPreviewPlotter に flux_sink の time series summary を追加（任意）

AC
	•	flux無しNPZも今まで通り読める
	•	fluxありNPZを読み込んで、reportに “入力として読めている” ことを示せる

⸻

P0‑03：kmの“暴れ”と“輸送支配の兆候”を出力・可視化（P1導入判断の材料）

目的
	•	P1を入れる前に、現状の同化が
	•	kmに吸収されすぎていないか
	•	反応係数が安定しているか
	•	B/Iの必要性が物理診断で裏付けられるか
を判断できるようにする

追加すべき診断（P0で必須）
	•	km_A（per_condition）の推定値（fit後）を表とプロットで出す
	•	phi_B と f_I の分布（bestモデル）
	•	CsA/CrefA, CsB/CrefB（Bありの場合）の分布
	•	solver健全性（clip率、bracket_fail率）

P0で追加する “km関連の図”
	•	kmA_map.png：スカラーkmでもOK（全点同一）
	•	tauA_map.png：\tau=\Delta z / k_m の指標（\Delta z=z_ref を使う）
	•	z_refはsim configに必須メタ情報として保持（P0の時点で“表示だけ”）

AC
	•	BM‑01 run の report で「kmが条件間でどれくらい違うか」が一目で分かる
	•	“B/Iが勝ったのに診断が弱い”などの疑いを把握できる

⸻

P0‑04：BM‑01（既存版：free km）で「P1に進むべきか」判定できる基準を明文化

目的
	•	P1に進む判断が属人化しないようにする（現場導入で重要）

判定基準（P0のBM‑01結果から）

以下が成立するなら P1（flux‑km拘束）に進む価値が高い：
	•	kmが条件ごとに極端に振れる
	•	例：km_A(condA) と km_A(condB) が 10倍以上
	•	AIBが勝つのに \phi_B や f_I が “効いている形” を出せない
	•	Bありなのに edge で CsB/CrefB ≈ 1 ばかり
	•	test（Cond‑C）で外挿が崩れる
	•	A-onlyより改善が小さい、またはリング形状が再現できない

AC
	•	report末尾に “P1推奨判定（理由付き）” を自動で出せる（簡易でOK）

⸻

P0‑05：ドキュメント整備（第三者が迷わない）

追加ドキュメント（最低限）
	•	docs/inputs_fluent.md
	•	cref と flux_sink の意味・単位・符号規約
	•	docs/transport_km.md
	•	kmが本コードで担う物理（滞在時間・更新）
	•	P1導入後の k_m = gamma * (J_sink/Cref) の意味

AC
	•	“fluxが無いなら何も変えずに動く / fluxがあるならP1で活用できる” が明記されている

⸻

P1 実装ロードマップ（詳細）

ゴール：フラックス情報を「km拘束」として導入し、
最適化対象を増やさず、むしろ減らし、物理妥当性・外挿性・説明力を上げる。

⸻

P1‑01：CfdFluxSinkKmProvider の実装（km(x,y,t)を生成）

目的
	•	Fluent入力 flux_sink と cref から km分布を生成

実装仕様
	•	km生成：
k_m^{CFD}(x,y,t)=\frac{J_{sink}(x,y,t)}{C_{ref}(x,y,t)+\epsilon}
	•	数値安全策（最小限でよいが必須）
	•	eps_cref（例 1e‑12）
	•	km_clip_min, km_clip_max（極端値を抑制）
	•	flux_negative_policy: error|clip_to_zero|allow（基本は error 推奨）

YAML（sim側）

sim:
  transport:
    km_source: "from_cfd_flux_sink"
    from_cfd_flux_sink:
      flux_key: "flux_sink"
      eps_cref: 1.0e-12
      km_clip: [1.0e-8, 1.0e+4]
      flux_negative_policy: "error"

AC
	•	flux_sinkが与えられた条件で km_map が空間的に変化する
	•	flux_sink無しの条件では明確にエラー（or fallback）を出せる
※運用上は “ONにしたのに無い” は事故なので error推奨

⸻

P1‑02：kmの自由度を「km」から「gamma_km」に置換（最適化対象を削減）

目的
	•	kmが黒箱パラメータとして暴走するのを止める
	•	反応係数（k_ads/k_des/k_rxn/K_I）の識別を良くする

方式
	•	使用km：
k_m^{used}=\gamma_{km}^{(cond)}\,k_m^{CFD}
	•	\gamma_{km} は per_condition（装置条件差の吸収ノブ）
	•	Bありの場合のみ \gamma_{km,B} を導入（B無しなら不要）

YAML（opt側）

opt:
  parameters:
    per_condition:
      gamma_km_A: {type: float, low: 0.2, high: 5.0, log: true}
      gamma_km_B: {type: float, low: 0.2, high: 5.0, log: true, when: hasB}

互換性バリデータ（必須）
	•	km_source=from_cfd_flux_sink のとき
	•	km_A を最適化対象に入れていたら エラー
	•	gamma_km_A が無ければ エラー（またはdefault=1でWARN）

AC
	•	同化結果で kmが「CFDベース＋スケール」になり、条件間の不自然な暴れが減る
	•	反応係数が条件間で安定しやすくなる（特にmulti-condition）

⸻

P1‑03：物理診断・可視化の強化（km拘束の価値を見せる）

追加すべき必須図（P1）
	•	kmA_cfd_map（CFD由来）
	•	kmA_used_map（gamma適用後）
	•	tauA_map（\tau=\Delta z/k_m）
	•	Bありなら kmB_*, tauB_* も同様

追加すべき表（report）
	•	conditionごとの gamma_km_A, gamma_km_B
	•	gamma_km が極端値（>10, <0.1）なら警告

AC
	•	reportを見るだけで「輸送（滞在時間）が効いている」説明ができる
	•	gammaが極端なら “入力単位不一致/CFD設定不整合” を疑える

⸻

P1‑04：BM‑01 manifest を拡張し、judgeに “flux‑km版” の合格判定を追加

目的
	•	「入れたら良くなった」を自動で判定できるようにする

追加する run_plan（例）
	•	fit_main_free_km（従来）
	•	fit_main_flux_km（新）
	•	baseline_aonly_free_km
	•	baseline_aonly_flux_km

judge追加（例）
	•	flux‑km導入後の gate：
	•	gamma_km_A が [0.1, 10] に入っている（WARN/FAIL）
	•	testで A-onlyより改善（既存）

AC
	•	BM‑01で “free_km vs flux_km” の比較が1クリックで分かる（index.html）

⸻

P1‑05：運用リスク（単位・符号・ノイズ）に対するガードを実装

よくある事故
	•	fluxが kg/m²/s、Cが mol/m³ → kmが意味不明
	•	flux符号が逆 → kmが負になる
	•	Cref≈0で比が発散

必須の対策（最小）
	•	単位の自己申告：YAMLに units_hint を持たせ、reportに表示
（自動変換はしない。誤魔化すと危険）
	•	flux_negative_policy=error を基本にする
	•	eps_cref と km_clip を固定で使う
	•	“km分布のヒストグラム”を必ず出して異常検知

AC
	•	おかしな入力で「静かに当たる」ことがなく、必ず警告/停止できる

⸻

P0 / P1 の成果物まとめ（チェックリスト）

P0完了条件
	•	km供給がProvider化され、挙動が変わっていない
	•	Fluent入力が flux_sink を任意で読める
	•	BM‑01（free km）で、kmの暴れ・B/I診断・外挿の崩れが可視化される
	•	P1に進むべきかの判断材料がreportに出る

P1完了条件
	•	km_source=from_cfd_flux_sink で km(x,y,t) が生成できる
	•	最適化は gamma_km に縮約され、km自由度が減る
	•	km/tau の可視化が追加され、物理説明が強化される
	•	BM‑01で free_km vs flux_km の比較が自動ででき、judgeがPASS/FAIL/WARNを出す

⸻

最後に：P0→P1の移行判断（実務的な “Go/No‑Go”）

P0のBM‑01結果で、次のどれかが出たら P1は導入価値が高いです：
	•	kmが条件間で大きく振れて反応係数が安定しない
	•	AIBが勝つが \phi_B や f_I が“効いている形”にならない（過学習疑い）
	•	test（Cond‑C）の非単調ラジアル再現が崩れる

逆に、P0で既に
	•	kmが安定
	•	AIBの物理診断が明瞭
	•	testでも形状再現が良い
なら、P1は「必須ではない」可能性があります（ただし将来の外挿性向上の保険にはなる）。

⸻
