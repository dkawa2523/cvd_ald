了解です。ここまでの議論（MS-01〜MS-15、TC結合＋縮約表面運動論、多次数、CVD優先＋ALD対応、Hydra/YAML、レジストリ、出力迷子防止、CPU/GPUユーザー選択、互換性制約、メトロロジ整合、KPI、DOE、将来の同化/最適化）を踏まえ、**現場導入が成功しやすい順序**で **P0/P1/P2＋Decision** の実装タスクに落とし込みます。

> 方針：
>
> * **P0は「実測2Dマップと比較できる」ことが最優先**（研究者/装置エンジニアが価値を判断できる）。
> * そのため、数式モデルだけでなく **MeasurementAdapter＋KPI＋Report** をP0に入れます（これが無いと現場導入が止まります）。
> * 物理的整合性は **互換性メタデータ＋validator** で “不正な組合せを最初から止める” をP0に入れます（ご指摘への直接回答）。
> * P1でALD/状態/低圧/パターン等を広げ、P2で同化/最適化/UQ/運用連携を完成させます。

---

# 0) 成功しやすい実装順序（現場導入のための「ゲート」）

## 導入成功の最短ルート

1. **（P0前半）単発CVD steadyで膜厚2Dを出す**（TC+powerlawでOK）
2. **（P0中盤）実測2Dマップと“同じ座標系”で比較できる**（MeasurementAdapter＋KPI）
3. **（P0後半）DOEで10〜1000本回せる**（case集約出力＋index.html入口）
4. **（P0後半）モデル不正組合せを事前に止める**（互換性validator）
5. **（P1）ALD phases・被覆率・purge残留・sticking を追加**（現場ALDへ）
6. **（P1）低圧輸送ブリッジ・pattern loading・dep‑etch を追加**（外挿耐性UP）
7. **（P2）データ同化/最適化/JAX/追跡（ClearML）を追加**（運用自動化へ）

## 各マイルストーンの“合格条件（Gate）”

* **P0 Gate**：

  * YAML1つで単発runができ、膜厚2D＋診断＋KPI＋比較レポートが出る
  * DOEで100本（CPUでも可）を回して、case集約出力＋KPIランキングが出る
  * 不正構成（例：ω=0でrotating_disk、ODE stateをsteadyでdynamicなど）が preflight で止まる
* **P1 Gate**：

  * ALD（phases）でGPC・purge時間依存が再現できる
  * 低圧/パターン/複数チャネルを入れても破綻せず、互換性チェックが効く
* **P2 Gate**：

  * 実測2Dマップでパラメータ同化が回り、外挿テスト（条件変更）で過学習が見える化できる
  * ClearML連携が“core非依存”で動く

---

# 1) Decisionタスク（未決はここで止める：再作業を減らす）

現場導入で手戻りが大きい項目を **先にdecision化**します。P0を進める上で「必須でないもの」はP1以降へ回し、P0は止めない設計にします。

## Decision一覧（優先順）

### DEC-001（P0開始前に推奨）単位と膜厚換算の規約（αの定義）

* **論点**：(\dot h=\alpha r) の α を「材料密度/モル体積/組成依存」まで入れるか、P0は定数か
* **止める理由**：膜厚比較の基準がぶれると、同化/比較が崩壊
* **提案（暫定）**：P0は `alpha_const`、P1で `alpha_material`（密度/組成）を追加

### DEC-002（P0開始前に推奨）実測2Dマップの座標系・中心・回転・mask仕様

* **論点**：測定マップの原点、角度基準、edge exclusion、欠損点、補間の方針
* **止める理由**：ズレがモデルに吸収され、次数やkmが漂流する
* **提案（暫定）**：P0は `dx,dy,rot,mask` を必須入力にして、補間は最小（nearest/bilinear）

### DEC-003（P0後半で良い）温度Tはマップかスカラーか、将来熱モデルを要するか

* **論点**：Arrhenius依存が強いのでTの扱いが外挿耐性を左右
* **提案（暫定）**：P0は T入力必須（スカラーでも可）、P1でTマップ前提を強化

### DEC-004（P1で良い）z_refのmulti‑z対応の要否

* **提案（暫定）**：P0は fixed z_ref、P2で multi-z拡張

### DEC-005（P1で良い）pattern loading S(x,y) を入力として持てるか（データ入手性）

* **提案（暫定）**：Sは optional field。無ければ使わない。

---

# 2) P0 実装タスク（現場導入“初回成功”に必要な最小セット）

P0は「**CVD steady を本命**」「実測比較」「DOE運用」「互換性で事故防止」を同時に満たします。
（ALDや高度物理はP1以降）

以下、**現場導入が成功しやすい順序**で並べます。

---

## P0-001（implement）スキーマ確定：Domain/Time/ReferencePlane/Inputs/Outputs

* **目的**：第三者が迷わないI/O契約を固定
* **内容**

  * `DomainSpec`：`wafer_2d_polar`, `wafer_2d_xy`, `wafer_1d_radial`
  * `ReferencePlaneSpec`：`z_ref_mm`（固定・後で変更可）
  * `TimeSpec`：`steady/transient/phases`（P0ではsteadyを使うがスキーマは揃える）
  * `InputsSpec`：fields（Cref_i, U, T, optional S）、scalars（P, omega…）
  * `OutputSpec`：results入口固定（index.html）、case集約の器（zarr等）
* **受入条件**

  * YAMLをロードして型検証できる
* **依存**：なし

---

## P0-002（implement）Registry基盤：カテゴリ別プラグイン登録

* **目的**：MS-01〜15 を“差し替え”で実装できる骨格を作る
* **内容**

  * registries：`input_transform/driver/mass_transfer/diffusivity/rate_core/rate_modifier/state/net/surface_solver/root_solver/diagnostics/postprocess`
  * `name`→class factory
  * `metadata`（requires/excludes/time_modes/governing_class）を格納可能に
* **受入条件**

  * ダミープラグインで `name` 指定して生成できる
* **依存**：P0-001

---

## P0-003（implement）SurfaceSolver：TC結合＋進行度R 1変数root（bisection）

* **目的**：CVD steadyを最短で回す“核”
* **内容**

  * (C_{s,i}=C_{ref,i}-\nu_i R/k_{m,i})
  * (F(R)=R-r(C_s))
  * (R\in[0,R_{\max}]) のbisection（ベクトル化）
  * 診断：反復回数、収束フラグ、Rmaxヒット率
* **受入条件**

  * tiny gridで常に収束する（powerlawで）
* **依存**：P0-001, P0-002

---

## P0-004（implement）MassTransfer：stagnant_film + rotating_disk(guard)

* **目的**：回転あり/なし両対応、ω=0事故を防ぐ
* **内容**

  * stagnant_film：(k_m=D/\delta_\mathrm{eff})
  * rotating_disk：(\omega)依存相関＋`guard`（error/fallback）
* **受入条件**

  * ω=0で rotating_disk を選ぶと validator or guard が動く
* **依存**：P0-002, P0-003

---

## P0-005（implement）RateCore：powerlaw_terms + sat_inh_terms（項ON/OFF対応）

* **目的**：多次数・負次数・遷移を“式の選択”で扱えるようにする
* **内容**

  * powerlaw_terms：terms list（enabled, order）
  * sat_inh_terms：numerator_terms/denominator_terms（enabled）＋β
  * `apparent_orders()` を可能な範囲で実装（n_app診断に使う）
* **受入条件**

  * termsをON/OFFして結果が変わることを確認できる
* **依存**：P0-002, P0-003

---

## P0-006（implement）Diagnostics：Cs/Cref、n_app、Da proxy、regime numbers（Kn/Re/Sc）

* **目的**：物理学者に“妥当性”を説明できる状態にする（導入成功に直結）
* **内容**

  * (C_s/C_{ref}) マップ
  * (n_\mathrm{app}=\partial\ln r/\partial\ln C_s)（可能なもの）
  * 反応/輸送指標（Da proxy）
  * レジーム数：Kn/Re/Sc/Pe（入力が足りない場合は推定で注記）
* **受入条件**

  * reportで診断が必ず出る
* **依存**：P0-003〜005

---

## P0-007（implement）MeasurementAdapter：実測2Dマップとの座標整合＋mask

* **目的**：現場導入の“最大の壁”を最初に潰す
* **内容**

  * dx/dy/回転/スケール
  * edge exclusion mask、欠損処理
  * sim grid↔measurement grid の補間
* **受入条件**

  * 同じ入力を回した時、比較が再現可能（同じ設定で同じ誤差が出る）
* **依存**：DEC-002（座標仕様）※未決なら最小暫定で実装

---

## P0-008（implement）KPI/Metrics：NU%、center-edge、ring stats、規格外面積率

* **目的**：装置エンジニアが“使える”成果物にする
* **受入条件**

  * `summary.json` にKPIが出る
  * DOEでKPIランキングできる
* **依存**：P0-007

---

## P0-009（implement）Report：index.html固定入口＋図（膜厚/診断/KPI/比較）

* **目的**：ファイル迷子を防ぎ、レビュー可能にする
* **内容**

  * index.html（固定）
  * 膜厚2D、半径プロファイル、θ方向ばらつき
  * 実測との差分2D、誤差統計
  * 診断（Cs/Cref、n_app、solver stats、regime numbers）
* **受入条件**

  * “文字だけの出力”にならない（必ず図がある）
* **依存**：P0-006〜008

---

## P0-010（implement）DOE runner：10〜1000本、case集約出力（zarr/npz）

* **目的**：現場で探索できる（最適化の前段）
* **内容**

  * パラメータスイープ（grid/latin等はP1でよい、P0はgridでOK）
  * 出力は case 次元で集約（ディレクトリ乱立しない）
  * CPU並列（標準）、GPU/JAXは“ユーザー選択のオプション”として後で
* **受入条件**

  * 100ケースを回し、summaryにKPI一覧が出る
* **依存**：P0-009

---

## P0-011（implement）互換性Validator：Core/Modifier/State closure/Time の整合チェック

* **目的**：あなたの懸念（ON/OFFだけでは足りない）を構造で解決
* **内容**

  * plugin metadata（requires/excludes/time_modes/governing_class）
  * preflightで構成を検証し、ダメなら明確に止める
  * 例：

    * ω=0でrotating_diskは禁止（guardでfallbackも可）
    * `state_model.dynamic_ode` は `time.mode=transient/phases` のみ許可
      （steadyで使いたい場合は `closure_mode=steady_state` をP1で追加）
* **受入条件**

  * 代表的な不正構成が確実に弾かれる
* **依存**：P0-002

---

## P0-012（review）整合性スイープ（MS-01〜MS-05のプリセットYAMLを用意）

* **目的**：現場が“選べる”状態にする（MSカタログの最小）
* **内容**

  * MS-01〜MS-05（CVD基本/阻害/回転/気相ロス/低圧はP1でも可）
  * ただしP0は MS-01〜MS-04 まででも成立
* **受入条件**

  * どれを選んでもvalidatorで安全に回る
* **依存**：P0-001〜011

---

## P0-013（implement）テスト：smoke＋最小数値検証＋回帰防止

* **目的**：第三者追加で壊れないようにする
* **内容**

  * tiny gridで `sim_run` が落ちない
  * ω=0ガード、validatorの不正構成テスト
  * MeasurementAdapterの座標変換テスト
* **受入条件**

  * `pytest` が通る（or 最小verifyコマンドが通る）
* **依存**：P0全体

---

## P0-CP（checkpoint）P0完了：現場導入判定

* **停止条件**：P0 Gate を満たしたら止める（ここで一旦レビュー・現場投入）

---

# 3) P1 実装タスク（ALD/状態/低圧/パターン/多チャネルで“適用範囲を拡張”）

P1は、P0で価値を示した上で「ALDを当てる」「外挿耐性を上げる」「物理の説明力を上げる」を追加します。

---

## P1-001（implement）Time.phases + Driver拡張（phase内時間変化）

* **内容**

  * phasesループ（expose/purge…）
  * driverでphase内に指数減衰（purge_decay）
  * 入力プレビュー（適用前後差分）を必須出力
* **依存**：P0-001, P0-010

---

## P1-002（implement）StateModel：coverage（ODE）+ closure_mode（dynamic_ode / steady_state）

* **目的**：ALDの核（自己終端）＋steadyでも状態を使いたい要求に対応
* **内容**

  * dynamic_ode：(\dot\theta=A(C,T)(1-\theta)^m-B\theta)
  * steady_state：(0=g(C,\theta,T)) を解く（追加root/固定点）
* **依存**：P1-001、P0-011（validator更新）

---

## P1-003（implement）RateCore：sticking_flux（Hertz–Knudsen＋s(θ)）

* **内容**

  * (p=C_sRT)、(\Gamma=\alpha p/\sqrt{2\pi mk_BT})
  * (r=s(\theta)\Gamma)
* **依存**：P1-002

---

## P1-004（implement）StateModel：poisoning / incubation（＋composite state）

* **目的**：履歴（被毒/初期成長遅れ）で“合わないところだけ”を救う
* **内容**

  * poisoning：(\dot\theta_I=k_{ads}C_I(1-\theta_I)-k_{des}\theta_I)
  * incubation：(\dot\eta=kC^n(1-\eta))
  * composite：states list 合成（重複state禁止）
* **依存**：P1-002

---

## P1-005（implement）Diffusivity：Bosanquet（low‑P bridge）

* **内容**

  * (1/D_{eff}=1/D_m+1/D_K)
  * km側サブモジュールとして実装（階層化：二重ON/OFF事故を防ぐ）
* **依存**：P0-004

---

## P1-006（implement）Pattern loading S(x,y) 対応（TC式へのS挿入）

* **内容**

  * (C_s=C_{ref}-\nu R S/k_m)
  * Sが無い時は恒等（S=1）
* **依存**：DEC-005（データ入手性）

---

## P1-007（implement）NetModel：multi_channel（dep‑etch‑loss）＋rate辞書

* **目的**：PECVD等で符号反転・競合を表現
* **内容**

  * channels list（enabledで切替）
  * `rates["dep_rate"]` 等をNetModelが参照
* **依存**：P0-005（rate出力拡張）

---

## P1-008（review/implement）同定性・感度の一次診断（簡易FIM/相関）

* **目的**：同化前に“識別できない自由度”を炙り出す
* **内容**

  * パラメータ局所感度（有限差分でOK）
  * 相関/縮退検出をレポート
* **依存**：P0-010、P0-009

---

## P1-CP（checkpoint任意）ALD現場投入判定

* P0で導入済みの現場に、ALD/低圧/パターンを段階投入

---

# 4) P2 実装タスク（同化/最適化/UQ/運用連携で“自動化・高度化”）

P2は「当てる」から「自動で推定・最適化し、運用管理する」へ。

---

## P2-001（implement）deposim_opt：パラメータ同化（JAX）基盤

* **内容**

  * パラメータ変換（exp/softplus/sigmoid）
  * 誤差関数（2Dマップ＋KPI重み）
  * optimizer（L-BFGS/Adam）
* **依存**：P0-007, P0-009（比較/誤差定義）

---

## P2-002（implement）JAX engine（任意）：DOE vmap + JIT cache（ただしユーザー選択）

* **内容**

  * engine切替（numpy/jax）
  * JITキャッシュ活用
  * ただし “単発CPU/DOE GPU” の決め打ちはしない（ユーザー指定を尊重）
* **依存**：P0-010

---

## P2-003（implement）UQ：入力不確かさ（CFD誤差）を扱う

* **内容**

  * Cref誤差、T誤差、km誤差のモデル化
  * ensemble / ベイズ（段階導入）
* **依存**：P2-001

---

## P2-004（implement）運用連携：ClearML（core非依存の別パッケージ）

* **内容**

  * 条件・モデル・結果の紐付け
  * artifact登録（zarr/report）
* **依存**：P0-010, P0-009

---

## P2-005（decision/implement）multi‑z reference plane（必要なら）

* **内容**

  * `reference_plane.multi_plane`
  * z補正transform or km補正
* **依存**：DEC-004

---

## P2-006（optional physics）Stefan補正 / smoothing PDE / chamber seasoning

* **内容**

  * Stefan流補正（輸送非線形）
  * smoothing PDE（形状緩和）
  * chamber_state（seasoning drift）
* **依存**：現場必要性に応じて

---

# 5) “どの順で入れると成功しやすいか”の理由（短くまとめ）

* **P0で比較・KPI・レポートを入れる**：現場の評価軸が揃い、研究者も議論できる
* **P0でvalidatorを入れる**：ON/OFFが増えても“物理的にありえない組合せ”を排除できる
* **P1でALD・状態・低圧・パターン**：適用範囲を広げても、P0の比較枠で検証できる
* **P2で同化/最適化**：モデルが固まってからでないと過学習・漂流が起きやすい（現場不信に直結）

---

もし次のステップとして、あなたの既存Handoff Pack（tasks.json形式）にこの計画を **そのまま流し込めるJSON（task_id/type/stop_after/verification_commands/scope_limits付き）**で出力してほしい場合、同じ内容を “Codex自動実装向け” に整形した版も作れます。
