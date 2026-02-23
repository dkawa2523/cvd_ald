了解しました。ここでは **あなたが確定させた新方針（A/I/B role・I/B≤1・disjoint・unused許可・B次数0/1固定・ODE主役・power‑law等は廃止）**を、そのまま
	1.	YAML仕様（role割当・列挙・クラス別比較・出力項目）
	2.	ユーザーが迷わない最小設定テンプレ（CVD steady / CVD transient / ALD transient）
	3.	既存コード（旧MS-01..15/TC-root/RateCore中心の設計）を、今回方針へ改良する“具体的仕様（どこを消して何を入れるか）”

に落として提示します。VSCode + Codexでそのまま実装に入れる想定で、キー名・入出力契約・バリデーション規則・出力ファイルまで具体化します。

⸻

1) YAML仕様（Sim / Opt を分離、Hydraで合成）

あなたの方針どおり YAMLは sim と opt で分離します。
	•	configs/sim/*.yaml：単一条件 forward（CVD steady/transient、ALD transient）
	•	configs/opt/*.yaml：
	•	role列挙（A/AI/AB/AIBクラス比較含む）
	•	離散次数（最大3次）列挙
	•	連続パラメータ（ブラックボックス）最適化（Optuna）
	•	レポート出力

Opt側は Hydra の defaults で sim YAML を読み込んで合成する方式にします（=重複を最小化）。

⸻

1.1 Sim YAML（forward）仕様：sim: ブロック

必須の最小キー（迷わないため“これだけ”に絞る）

sim:
  process: cvd | ald                 # プロセス種別（表示/レポート用）
  time_mode: steady | transient       # steady=厚み積算, transient=時系列積分（ALDは基本transient）

  reference_plane:
    z_ref_mm: 1.0                     # Fluentの参照面高さ（固定だが変更可能）

  inputs:
    fluent:
      mode: steady | transient
      file: path/to/fluent_case.npz
      keys:
        cref: cref                     # (steady) [n_pts, n_species] or (transient) [n_t, n_pts, n_species]
        xy: xy                         # [n_pts, 2]  (mm or m どちらかを明記)
        time: time                     # (transient) [n_t]
      species: [s0, s1, s2]            # file内配列のspecies順（<=10）
    temperature:
      mode: scalar | field             # 最小はscalarでOK
      value_K: 600.0

  domain:
    kind: from_fluent_xy               # 最小：Fluentのxyをそのまま使う（格子生成を不要化）
    xy_unit: mm                        # fileのxy単位
    wafer_radius_mm: 150.0             # NU%などの指標計算で使用（任意）

  roles:
    mode: fixed
    A: s0
    I: null                            # null=無し
    B: null

  model:
    name: aib_ode                      # 今回の核（これだけ）
    orders:
      adsorption_site_order: 1         # m_ads ∈ {1,2}
      reaction_site_order_A: 1         # p_A ∈ {1,2}
      reaction_site_order_star: 0      # p_* ∈ {0,1,2}
      enforce_total_order_le: 3        # p_A + p_* + (B?1:0) <= 3 を強制
    params:
      # できるだけ “lumped” にして黒箱パラメータ増を抑える（後述）
      transport:
        km_A: {mode: constant, value: 0.02}    # [m/s] など（単位をCONTEXTに固定）
        km_B: {mode: constant, value: 0.02}    # Bが無いなら無視（validatorで不要扱い）
      kinetics:
        k_ads: 1.0                              # [1/(s*conc)]のlumpedでも可
        k_des: 0.1                              # [1/s]
        k_rxn: 0.01                             # [1/s]（B有りなら内部でB依存）
      inhibitor:
        K_I: 0.0                                # I無しなら0でOK（もしくは省略可）
      thickness:
        alpha_h: 1.0                            # [nm] per event（lumped）
      scaling:
        C_B_scale: 1.0                          # B有り時の無次元化スケール（推奨）

  time:
    # steadyなら t_proc_s が必須、transientなら input.time を使う
    t_proc_s: 30.0
    dt_s: 0.01
    solver:
      name: implicit_euler_bisect               # 1変数θの暗黙更新（安定優先）
      max_iter: 60
      theta_tol: 1.0e-10

  initial_conditions:
    theta_A:
      mode: scalar
      value: 0.0
    h_nm:
      mode: scalar
      value: 0.0

  measurement:                                  # 任意：与えたら比較・残差も出す（reportが充実）
    enabled: false
    file: path/to/meas.npz
    keys:
      h: h_nm                                   # [n_pts] or [nx,ny]（adapterで合わせる）
      xy: xy
    align:
      enable: true
      shift_mm: [0.0, 0.0]
      rotate_deg: 0.0
      mask_radius_mm: 150.0

  output:
    project: my_project
    run_name: cvd_steady_001
    root_dir: results
    store:
      format: npz                               # まずnpz（DOEで必要ならzarrへ）
    save_fields:
      - h_nm
      - theta_A
      - theta_star
      - CsA_over_CrefA
      - CsB_over_CrefB
      - phi_B
      - f_I
      - residual_nm
    report:
      enabled: true
      index_html: true

重要な“仕様ルール”（Validatorで強制する前提）
	•	I/B は null or 1 species（sim固定の時も同じ）
	•	A/I/B は disjoint（同じspeciesを複数roleに割り当てない）
	•	unused species OK（roleに入らないspeciesがあってよい）
	•	B次数は0/1固定で、Bの有無から自動決定
	•	roles.B == null → m_B=0（B無しモデル）
	•	roles.B != null → m_B=1（B有りモデル）
	•	3次制約：p_A + p_* + (B?1:0) \le 3

⸻

1.2 Opt YAML（列挙＋クラス比較＋Optunaフィット）仕様：opt: ブロック

Opt YAMLは Sim YAMLを defaults で読み込み、以下を追加します。

opt:
  task: fit_roles_and_params

  measurement:                     # optでは基本必須（合わせ込みの対象）
    file: path/to/meas.npz
    keys: {h: h_nm, xy: xy}
    align:
      enable: true
      shift_mm: [0.0, 0.0]
      rotate_deg: 0.0
      mask_radius_mm: 150.0

  role_enumeration:
    enabled: true
    species_source: from_sim_input           # sim.inputs.fluent.species を使う
    constraints:
      disjoint: true
      allow_unused: true
    roles:
      A:
        required: true
        candidates: auto                     # 最小はauto=全species
      I:
        required: false
        allow_none: true
        max_size: 1
        candidates: auto
      B:
        required: false
        allow_none: true
        max_size: 1
        candidates: auto

  order_enumeration:
    enabled: true
    # B次数は 0/1固定で role.B の有無から自動なのでここでは列挙しない
    candidates:
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 0}
      - {adsorption_site_order: 2, reaction_site_order_A: 1, reaction_site_order_star: 0}
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 1}
      - {adsorption_site_order: 1, reaction_site_order_A: 2, reaction_site_order_star: 0}
    enforce_total_order_le: 3

  class_compare:
    enabled: true
    classes: [A, AI, AB, AIB]                # sI/sB の None/非None で自動分類
    complexity_penalty:
      lambda_role: 0.1                       # 例：I/Bを使うほど罰則（過学習防止）
      # 罰則は “スコア差が小さいなら採用しない” を作るために必須

  parameter_fit:
    engine: optuna
    sampler: tpe
    seed: 123
    n_trials_per_candidate: 40               # 最小：20～50程度
    objective:
      loss: huber
      huber_delta_nm: 10.0
      # 任意：正則化（黒箱パラメータの暴走抑制）
      regularization:
        l2:
          enabled: true
          weight: 1.0e-4
          targets:
            - name: model.params.transport.km_A
              scale: log
            - name: model.params.transport.km_B
              scale: log
    search_space:
      # “まず当てたい最小セット”だけに絞る（黒箱パラメータ爆発を抑える）
      - name: model.params.kinetics.k_rxn
        type: loguniform
        low: 1.0e-6
        high: 1.0e+2
      - name: model.params.transport.km_A
        type: loguniform
        low: 1.0e-4
        high: 1.0e+1
      - name: model.params.transport.km_B
        type: loguniform
        low: 1.0e-4
        high: 1.0e+1
        condition: role_has_B                # B有り候補だけ有効
      - name: model.params.inhibitor.K_I
        type: loguniform
        low: 1.0e-6
        high: 1.0e+6
        condition: role_has_I                # I有り候補だけ有効

  selection:
    topk_overall: 20
    topk_per_class: 10

  output:
    project: my_project
    run_name: fit_cvd_steady_001
    root_dir: results
    save:
      ranking_csv: true
      class_compare_csv: true
      topk_models_npz: true
      resolved_configs: true
    report:
      enabled: true
      index_html: true

クラス別比較（A/AI/AB/AIB）の定義（コードの単純化）
	•	A：I=null, B=null
	•	AI：I!=null, B=null
	•	AB：I=null, B!=null
	•	AIB：I!=null, B!=null

※この分類は **role割当（Noneかどうか）から機械的に決まり、モデル式は同一（AIB‑ODE統一コア）**です。

⸻

2) 最小設定テンプレ（CVD steady / CVD transient / ALD transient）

ここでは「迷わない」ことを最優先に、6ファイルとして提示します：
	•	simテンプレ（3つ）
	•	optテンプレ（3つ：simテンプレを defaults で読み込み）

「列挙・クラス比較」を含めたいなら opt を使う。
forwardだけなら sim だけでOK。

⸻

2.1 configs/sim/cvd_steady_min.yaml

sim:
  process: cvd
  time_mode: steady

  reference_plane: {z_ref_mm: 1.0}

  inputs:
    fluent:
      mode: steady
      file: data/fluent_cvd_steady.npz
      keys: {cref: cref, xy: xy}
      species: [s0, s1, s2, s3]
    temperature: {mode: scalar, value_K: 650.0}

  domain:
    kind: from_fluent_xy
    xy_unit: mm
    wafer_radius_mm: 150.0

  roles:
    mode: fixed
    A: s0
    I: null
    B: null

  model:
    name: aib_ode
    orders:
      adsorption_site_order: 1
      reaction_site_order_A: 1
      reaction_site_order_star: 0
      enforce_total_order_le: 3
    params:
      transport:
        km_A: {mode: constant, value: 0.02}
        km_B: {mode: constant, value: 0.02}
      kinetics:
        k_ads: 1.0
        k_des: 0.1
        k_rxn: 0.01
      inhibitor:
        K_I: 0.0
      thickness:
        alpha_h: 1.0
      scaling:
        C_B_scale: 1.0

  time:
    t_proc_s: 30.0
    dt_s: 0.01
    solver: {name: implicit_euler_bisect, max_iter: 60, theta_tol: 1.0e-10}

  initial_conditions:
    theta_A: {mode: scalar, value: 0.0}
    h_nm: {mode: scalar, value: 0.0}

  measurement:
    enabled: false

  output:
    project: demo
    run_name: cvd_steady_min
    root_dir: results
    store: {format: npz}
    save_fields: [h_nm, theta_A, theta_star, CsA_over_CrefA, f_I, residual_nm]
    report: {enabled: true, index_html: true}


⸻

2.2 configs/sim/cvd_transient_min.yaml

sim:
  process: cvd
  time_mode: transient

  reference_plane: {z_ref_mm: 1.0}

  inputs:
    fluent:
      mode: transient
      file: data/fluent_cvd_transient.npz
      keys: {cref: cref, xy: xy, time: time}
      species: [s0, s1, s2, s3]
    temperature: {mode: scalar, value_K: 650.0}

  domain:
    kind: from_fluent_xy
    xy_unit: mm
    wafer_radius_mm: 150.0

  roles:
    mode: fixed
    A: s0
    I: null
    B: null

  model:
    name: aib_ode
    orders:
      adsorption_site_order: 1
      reaction_site_order_A: 1
      reaction_site_order_star: 0
      enforce_total_order_le: 3
    params:
      transport:
        km_A: {mode: constant, value: 0.02}
        km_B: {mode: constant, value: 0.02}
      kinetics:
        k_ads: 1.0
        k_des: 0.1
        k_rxn: 0.01
      inhibitor: {K_I: 0.0}
      thickness: {alpha_h: 1.0}
      scaling: {C_B_scale: 1.0}

  time:
    dt_s: 0.005                          # input.time が粗い時はsubstep用
    solver: {name: implicit_euler_bisect, max_iter: 60, theta_tol: 1.0e-10}

  initial_conditions:
    theta_A: {mode: scalar, value: 0.0}
    h_nm: {mode: scalar, value: 0.0}

  output:
    project: demo
    run_name: cvd_transient_min
    root_dir: results
    store: {format: npz}
    save_fields: [h_nm, theta_A, theta_star, CsA_over_CrefA, residual_nm]
    report: {enabled: true, index_html: true}


⸻

2.3 configs/sim/ald_transient_min.yaml

ALDは “phases専用ドライバ” を作らず、Fluentの時系列 Cref(t) をそのまま積分する最小形（あなたの方針に一致）。
A/Bパルスが時系列に現れる前提です。

sim:
  process: ald
  time_mode: transient

  reference_plane: {z_ref_mm: 1.0}

  inputs:
    fluent:
      mode: transient
      file: data/fluent_ald_transient.npz
      keys: {cref: cref, xy: xy, time: time}
      species: [s0, s1, s2, s3]          # A候補/B候補が混ざっている前提
    temperature: {mode: scalar, value_K: 550.0}

  domain:
    kind: from_fluent_xy
    xy_unit: mm
    wafer_radius_mm: 150.0

  roles:
    mode: fixed
    A: s0
    I: null
    B: s1                               # ALDは基本Bあり。fitでは列挙で変えられる

  model:
    name: aib_ode
    orders:
      adsorption_site_order: 1
      reaction_site_order_A: 1
      reaction_site_order_star: 0
      enforce_total_order_le: 3
    params:
      transport:
        km_A: {mode: constant, value: 0.02}
        km_B: {mode: constant, value: 0.02}
      kinetics:
        k_ads: 1.0
        k_des: 0.1
        k_rxn: 0.01
      inhibitor: {K_I: 0.0}
      thickness: {alpha_h: 1.0}
      scaling: {C_B_scale: 1.0}

  time:
    dt_s: 0.001
    solver: {name: implicit_euler_bisect, max_iter: 60, theta_tol: 1.0e-10}

  initial_conditions:
    theta_A: {mode: scalar, value: 0.0}
    h_nm: {mode: scalar, value: 0.0}

  output:
    project: demo
    run_name: ald_transient_min
    root_dir: results
    store: {format: npz}
    save_fields: [h_nm, theta_A, theta_star, CsA_over_CrefA, CsB_over_CrefB, phi_B, residual_nm]
    report: {enabled: true, index_html: true}


⸻

2.4 configs/opt/fit_cvd_steady_min.yaml

defaults:
  - /sim: cvd_steady_min
  - _self_

opt:
  task: fit_roles_and_params

  measurement:
    file: data/meas_cvd_steady.npz
    keys: {h: h_nm, xy: xy}
    align: {enable: true, shift_mm: [0.0, 0.0], rotate_deg: 0.0, mask_radius_mm: 150.0}

  role_enumeration:
    enabled: true
    species_source: from_sim_input
    constraints: {disjoint: true, allow_unused: true}
    roles:
      A: {required: true, candidates: auto}
      I: {required: false, allow_none: true, max_size: 1, candidates: auto}
      B: {required: false, allow_none: true, max_size: 1, candidates: auto}

  order_enumeration:
    enabled: true
    candidates:
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 0}
      - {adsorption_site_order: 2, reaction_site_order_A: 1, reaction_site_order_star: 0}
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 1}
    enforce_total_order_le: 3

  class_compare:
    enabled: true
    classes: [A, AI, AB, AIB]
    complexity_penalty: {lambda_role: 0.1}

  parameter_fit:
    engine: optuna
    sampler: tpe
    seed: 123
    n_trials_per_candidate: 40
    objective: {loss: huber, huber_delta_nm: 10.0}
    search_space:
      - {name: model.params.kinetics.k_rxn, type: loguniform, low: 1.0e-6, high: 1.0e+2}
      - {name: model.params.transport.km_A,  type: loguniform, low: 1.0e-4, high: 1.0e+1}
      - {name: model.params.transport.km_B,  type: loguniform, low: 1.0e-4, high: 1.0e+1, condition: role_has_B}
      - {name: model.params.inhibitor.K_I,   type: loguniform, low: 1.0e-6, high: 1.0e+6, condition: role_has_I}

  selection: {topk_overall: 20, topk_per_class: 10}

  output:
    project: demo
    run_name: fit_cvd_steady_min
    root_dir: results
    save: {ranking_csv: true, class_compare_csv: true, resolved_configs: true}
    report: {enabled: true, index_html: true}


⸻

2.5 configs/opt/fit_cvd_transient_min.yaml

defaults:
  - /sim: cvd_transient_min
  - _self_

opt:
  task: fit_roles_and_params

  measurement:
    file: data/meas_cvd_transient_final.npz   # 最小：最終膜厚だけでもOK
    keys: {h: h_nm, xy: xy}
    align: {enable: true, shift_mm: [0.0, 0.0], rotate_deg: 0.0, mask_radius_mm: 150.0}

  role_enumeration:
    enabled: true
    species_source: from_sim_input
    constraints: {disjoint: true, allow_unused: true}
    roles:
      A: {required: true, candidates: auto}
      I: {required: false, allow_none: true, max_size: 1, candidates: auto}
      B: {required: false, allow_none: true, max_size: 1, candidates: auto}

  order_enumeration:
    enabled: true
    candidates:
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 0}
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 1}
    enforce_total_order_le: 3

  class_compare:
    enabled: true
    classes: [A, AI, AB, AIB]
    complexity_penalty: {lambda_role: 0.1}

  parameter_fit:
    engine: optuna
    sampler: tpe
    seed: 123
    n_trials_per_candidate: 30
    objective: {loss: huber, huber_delta_nm: 10.0}
    search_space:
      - {name: model.params.kinetics.k_rxn, type: loguniform, low: 1.0e-6, high: 1.0e+2}
      - {name: model.params.transport.km_A,  type: loguniform, low: 1.0e-4, high: 1.0e+1}
      - {name: model.params.transport.km_B,  type: loguniform, low: 1.0e-4, high: 1.0e+1, condition: role_has_B}
      - {name: model.params.inhibitor.K_I,   type: loguniform, low: 1.0e-6, high: 1.0e+6, condition: role_has_I}

  selection: {topk_overall: 20, topk_per_class: 10}

  output:
    project: demo
    run_name: fit_cvd_transient_min
    root_dir: results
    save: {ranking_csv: true, class_compare_csv: true, resolved_configs: true}
    report: {enabled: true, index_html: true}


⸻

2.6 configs/opt/fit_ald_transient_min.yaml

defaults:
  - /sim: ald_transient_min
  - _self_

opt:
  task: fit_roles_and_params

  measurement:
    file: data/meas_ald_final.npz
    keys: {h: h_nm, xy: xy}
    align: {enable: true, shift_mm: [0.0, 0.0], rotate_deg: 0.0, mask_radius_mm: 150.0}

  role_enumeration:
    enabled: true
    species_source: from_sim_input
    constraints: {disjoint: true, allow_unused: true}
    roles:
      A: {required: true, candidates: auto}
      I: {required: false, allow_none: true, max_size: 1, candidates: auto}
      B: {required: false, allow_none: true, max_size: 1, candidates: auto}  # ALDでも “B不要” を判定できる

  order_enumeration:
    enabled: true
    candidates:
      - {adsorption_site_order: 1, reaction_site_order_A: 1, reaction_site_order_star: 0}
      - {adsorption_site_order: 2, reaction_site_order_A: 1, reaction_site_order_star: 0}
    enforce_total_order_le: 3

  class_compare:
    enabled: true
    classes: [A, AI, AB, AIB]
    complexity_penalty: {lambda_role: 0.1}

  parameter_fit:
    engine: optuna
    sampler: tpe
    seed: 123
    n_trials_per_candidate: 40
    objective: {loss: huber, huber_delta_nm: 10.0}
    search_space:
      - {name: model.params.kinetics.k_rxn, type: loguniform, low: 1.0e-6, high: 1.0e+2}
      - {name: model.params.transport.km_A,  type: loguniform, low: 1.0e-4, high: 1.0e+1}
      - {name: model.params.transport.km_B,  type: loguniform, low: 1.0e-4, high: 1.0e+1, condition: role_has_B}
      - {name: model.params.inhibitor.K_I,   type: loguniform, low: 1.0e-6, high: 1.0e+6, condition: role_has_I}

  selection: {topk_overall: 20, topk_per_class: 10}

  output:
    project: demo
    run_name: fit_ald_transient_min
    root_dir: results
    save: {ranking_csv: true, class_compare_csv: true, resolved_configs: true}
    report: {enabled: true, index_html: true}


⸻

3) 出力項目（必須・推奨）を“迷子にならない形”で固定

3.1 ランの出力ディレクトリ（固定）

results/<project>/
  index.html                      # 常に入口（最新runへのリンク＋集約サマリ）
  summary.json                    # ざっと見る用（最良モデル/最良クラス/主要KPI）

  runs/<run_id>/                  # 1実行=1run
    config_resolved.yaml
    outputs/
      fields.npz                  # h, theta, Cs_ratio, phi_B, f_I, residual...
      metrics.json
    tables/
      ranking.csv                 # (opt) 候補ランキング（上位＋主要診断）
      class_compare.csv           # (opt) A/AI/AB/AIB のベスト比較
      topk_assignments.csv        # (opt) 上位Kの(sA,sI,sB,orders,score)
    plots/
      thickness_map.png
      residual_map.png
      radial_profile.png
      thetaA_map.png
      CsA_ratio_map.png
      CsB_ratio_map.png           # B有り候補の診断
      phiB_map.png                # B必要性の物理診断
      fI_map.png                  # I必要性の物理診断
    report.html                   # そのrunの詳細

3.2 必須保存フィールド（Sim/Opt共通）
	•	h_nm（最終膜厚 2D/1D）
	•	theta_A, theta_star（最終）
	•	CsA_over_CrefA
	•	residual_nm（measurementがある場合）
	•	metrics.json（NU%、C/E差など）

3.3 B/I の必要性診断（今回の新仕様で“必須”）
	•	phi_B：
\phi_B=\frac{\Gamma_s k_{rxn}\theta_A^{p_A}\theta_*^{p_*}/C_{B,scale}}{k_{m,B}}
→ Bが枯渇/輸送で効いているか（物理的にBが必要な兆候）
	•	f_I：
f_I=\frac{1}{1+K_I C_{ref,I}}
→ Iが空サイトをどれだけ潰しているか（Iが効いている兆候）
	•	class_compare.csv：A/AI/AB/AIB のベストスコア比較（複雑さ罰則込み）

⸻

4) 既存コード（旧MS-01..15/TC-root/RateCore中心）をどう改良するか：具体仕様

あなたが貼ってくれた既存設計は「MS-01..15」「TC root」「RateCoreの排他選択」等が中心です。
しかし今回方針では、
	•	power-law/分母形などは廃止
	•	回転km等も廃止
	•	ODEが主役
	•	species爆発は role割当（A/I/B）へ縮約し、I/Bは≤1で列挙
	•	B次数0/1固定

なので、パイプラインそのものを置き換えます（“ON/OFFの組合せ”を残すと誤用が復活するため）。

⸻

4.1 削除/廃止（または実装しない）対象を明確化

以下は今回のターゲットに対し、混乱・二重計上・同定不能を招きやすいので P0から排除（コードから外す/実装しない）：
	•	RateCore：power-law / sat_inh / LHHW / ER-like（旧形式のもの）
	•	TC root solver (R)：F(R)=0 の進行度root（今回のAIB-ODEは Csを代数で閉じるので不要）
	•	rotating_disk km（回転はFluentの時系列/分布に含める前提）
	•	Stefan補正・低圧ブリッジ・smoothing PDE・incubation・poisoning 等の周辺拡張
→ 今回の「絞り込み」方針ではまず不要（必要になったら AIB-ODE枠に整合する形で別途追加）

※残すのは「共通スタック」：Validator/Diagnostics/MeasurementAdapter/KPI/Report。

⸻

4.2 追加する“新しい核”：AIB-ODE 統一コア（1式でA/AI/AB/AIBを表現）

モデルは1つだけ（aib_ode）。クラスは role割当で自動的に決まります。

数式仕様（実装の契約）
	•	I（阻害）は 代数閉包（ODE次元を増やさない）
\theta_*=\frac{1-\theta_A}{1+K_I C_{ref,I}}
	•	Aの表面濃度は境界層＋吸着/脱離で 解析的に閉じる
C_{s,A}=\frac{k_{m,A}C_{ref,A}+\Gamma_s k_{des}\theta_A}{k_{m,A}+\Gamma_s k_{ads}\theta_*^{m_{ads}}}
	•	成膜イベント
r_{event}=k_{rxn}(T)\theta_A^{p_A}\theta_*^{p_*}\left(\frac{C_{s,B}}{C_{B,scale}}\right)^{m_B}
ただし m_B は B有無から自動（0 or 1固定）
	•	B表面濃度（m_B=1の時のみ）も 解析的に閉じる
C_{s,B}=\frac{k_{m,B}C_{ref,B}}{k_{m,B}+\Gamma_s k_{rxn}\theta_A^{p_A}\theta_*^{p_*}/C_{B,scale}}
	•	ODE（主役）：
\frac{d\theta_A}{dt}=k_{ads}C_{s,A}\theta_*^{m_{ads}}-k_{des}\theta_A-\nu_A r_{event}
	•	膜厚：
\frac{dh}{dt}=\alpha_h\Gamma_s r_{event}

⸻

4.3 新しい Validator（今回の誤用を構造的に防ぐ）

旧Validator（MS排他など）を置き換え、今回必要な制約を明文化して機械的に止めます。

Validatorルール（必須）
	1.	roles.A != null（A必須）
	2.	roles.I in {null, species}（I≤1）
	3.	roles.B in {null, species}（B≤1）
	4.	disjoint：A/I/B のspeciesが重複したらエラー
	5.	orders：
	•	adsorption_site_order ∈ {1,2}
	•	reaction_site_order_A ∈ {1,2}
	•	reaction_site_order_star ∈ {0,1,2}
	•	reaction_site_order_A + reaction_site_order_star + (B?1:0) <= 3
	6.	B無しなら、output.save_fields に CsB_over_CrefB や phi_B があっても 空（NaN）で出すか、または自動で落とす（どちらかに統一）
	7.	Opt条件：role列挙時は species <= 10 をwarn（超えたら遅いことを明示）

⸻

4.4 既存「MS」概念の置換：ユーザー視点で迷わない命名へ

旧：MS-01..15 を選ぶ
新：「入力の時間性」と「B/Iの必要性」を opt で判定し、forwardは常に aib_ode
	•	forward（sim）は：
	•	CVD steady / CVD transient / ALD transient の3つ
	•	同化（opt）は：
	•	role列挙（A/AI/AB/AIB）× 離散次数 × 連続パラメータ最適化
	•	best class / best assignment を出す

⸻

5) 実装を始めるための “コード改良仕様” をファイル粒度で提示（Codex向け）

既存のアーキ（deposim_schema / deposim_sim / deposim_report / deposim_opt）は活かし、追加すべき中核だけを明確にします。

5.1 deposim_schema（設定スキーマ）

追加/更新するdataclass（例）：
	•	SimConfig
	•	InputsFluentSpec（steady/transient共通I/F）
	•	RoleFixedSpec
	•	AIBODEModelSpec
	•	AIBOrdersSpec
	•	ImplicitEulerSpec
	•	MeasurementSpec
	•	OutputSpec
	•	OptConfig
	•	RoleEnumerationSpec（I/B≤1・disjoint・unused許可）
	•	OrderEnumerationSpec（候補列挙）
	•	ParameterFitSpec（Optuna）
	•	ClassCompareSpec
	•	SelectionSpec

5.2 deposim_sim（forward計算）

追加する主要モジュール：
	•	deposim_sim/inputs/fluent_loader.py
	•	load_fluent_npz(mode)-> FluentData(cref, xy, time, species)
	•	deposim_sim/roles/assignment.py
	•	apply_roles(cref, species, roles_fixed)-> CrefA,CrefI,CrefB
	•	deposim_sim/models/aib_ode.py
	•	step_theta_implicit(theta, dt, params, CrefA,CrefI,CrefB,T)-> theta_next, diagnostics
	•	simulate(cfg_sim)-> SimOutputs(fields, metrics)
	•	deposim_sim/diagnostics/aib_metrics.py
	•	CsA_over_CrefA, CsB_over_CrefB, phi_B, f_I
	•	deposim_sim/measurement/adapter.py（既存CS-04を継続）
	•	deposim_sim/metrics/kpi.py（CS-05）
	•	deposim_sim/validator.py（今回の新Validator）

5.3 deposim_opt（列挙＋フィット）

追加する主要モジュール：
	•	deposim_opt/enumerate_roles.py
	•	910通り程度を全列挙（S≤10、I/B≤1、disjoint、unused許可）
	•	各候補に class_id を付与（A/AI/AB/AIB）
	•	deposim_opt/enumerate_orders.py
	•	orders_candidates を列挙し、3次制約でフィルタ
	•	deposim_opt/fit_optuna.py
	•	1候補（roles+orders）につき Optunaで連続パラメータ最適化
	•	objective = loss(meas, sim) + lambda_role*(#I + #B)
	•	deposim_opt/class_compare.py
	•	各classのbestを集約して class_compare.csv を作る
	•	deposim_opt/run_fit.py
	•	Hydraで sim+opt を合成して実行

5.4 deposim_report
	•	forwardだけでも index.html を作る
	•	optなら追加で
	•	ranking.csv の上位を表に
	•	class_compare の棒グラフ/表
	•	topKの割当一覧
	•	B/I診断マップ（phi_B, f_I）

以下に、**「最短で動く（＝forwardが回る→比較ができる→optが回る）」**ための実装手順を、クラス/関数の実装順として落とし込みます。
その後、VSCode Codex（gpt‑5.2‑codex）にそのまま貼れる implement.prompt 雛形（＋例）を提示します。

前提（ここは固定）：
	•	モデル核は aib_ode のみ（A/AI/AB/AIBは role割当で決まるクラス分類）
	•	I/B は None or 単一species（≤1）、Aはまず単一species（最短ルート）
	•	A/I/B は disjoint、unused species は許可
	•	B次数は 0/1 固定（B=None→0、B!=None→1）
	•	反応次数は “power‑law” ではなく **機構次数（整数）**で離散列挙（最大3次制約）
	•	ODEが主役。TC root（Rを解く）/ power‑law / sat_inh などは今回スコープ外（既存に残っても config からは触れない）

⸻

A) 最短で動く実装手順（推奨の実装順）

全体のMVP階段（最短ルート）
	1.	MVP0：データI/O（NPZ読み込み）＋Hydraでconfig読める
	2.	MVP1：AIB‑ODEコア（1点）＋暗黙Euler（θ1変数）で安定に回る
	3.	MVP2：ウェハ全点ベクトル化（n_pts）でCVD steady forwardが回る
	4.	MVP3：transient（時間列）でCVD/ALDが回る
	5.	MVP4：measurement読み込み＋座標整合（最小）＋残差/指標
	6.	MVP5：role列挙（A必須、I/B None-or-1、disjoint、unused許可）＋クラス分類
	7.	MVP6：order列挙（整数次数＋3次制約）
	8.	MVP7：Optunaで連続パラメータフィット（候補1つ）
	9.	MVP8：候補全列挙×フィット×ランキング×クラス比較（A/AI/AB/AIB）
	10.	MVP9：report（index.html固定入口＋主要図）

以下、どの順でクラス/関数を書くかを、具体のファイル/関数単位で提示します。

⸻

STEP 0：I/Oフォーマットを固定（ここが最重要で最短）

0-1. Fluent NPZ 仕様（固定）
	•	steady:
	•	cref: shape [n_pts, n_species]
	•	xy: shape [n_pts, 2]
	•	transient:
	•	cref: shape [n_t, n_pts, n_species]
	•	xy: shape [n_pts, 2]
	•	time: shape [n_t]（秒）
	•	species のラベルは YAML 側で与える（sim.inputs.fluent.species: [s0,..]）

0-2. Measurement NPZ（最小）
	•	h_nm: shape [n_pts]
	•	xy: shape [n_pts,2]

この仕様が固まると、以降の実装が一気に速くなります。

⸻

STEP 1：Configスキーマ（最低限）＋Hydraで読めるようにする

最初に “設定→実行” の配線だけ通すのが最短です。

実装対象
	•	configs/sim/*.yaml（前回提示した3テンプレをまず配置）
	•	configs/opt/*.yaml（3テンプレ、ただしoptは後で動かせばOK）
	•	Python側：
	•	load_config()（Hydra/OmegaConfを使うならここで集約）
	•	dataclass でもOK（既存コード流儀に合わせる）

最低限の受入条件
	•	python -c "import <pkg>" が通る
	•	python -m <pkg>.run_sim +sim=cvd_steady_min のような最小runが（まだ中身ダミーでも）起動する

⸻

STEP 2：Fluentローダ（NPZ）＋Domain（from_fluent_xy）

ここから forward の土台が完成します。

実装する関数（最優先）
	•	load_fluent_npz(path, mode, keys) -> FluentData
	•	FluentData.cref, .xy, .time|None, .species
	•	build_domain_from_xy(xy, xy_unit, wafer_radius_mm) -> Domain
	•	最小は “そのまま保持” でOK

受入条件
	•	shape を検証して明示エラー（分かりやすい例外）
	•	cref < 0 があったら clip to 0 + warning（CFDの数値ノイズ対策）

⸻

STEP 3：Role固定の適用（sim用）

まずは sim YAML の roles: fixed をそのまま使えるようにします。

実装する関数
	•	apply_fixed_roles(cref, species_list, roles_fixed) -> (CrefA, CrefI, CrefB)
	•	I/B=null なら 0ベクトル（shapeは [... , n_pts] を維持）

Validator（最小）もここで入れる
	•	A必須
	•	I/B は None または species名
	•	disjoint（A/I/B重複禁止）

⸻

STEP 4：AIB‑ODE「物理コア」（まずは1点→次にベクトル化）

ここが最短の核心です。

実装する “純関数” を先に書く（テストしやすい）

（入力はすべて numpy.ndarray を想定、n_ptsベクトル化前提）
	1.	theta_star(thetaA, K_I, CrefI) -> theta_star
\theta_*=\frac{1-\theta_A}{1+K_I C_{ref,I}}
	2.	CsA(CrefA, thetaA, theta_star, kmA, k_ads, k_des, Gamma_s, m_ads) -> CsA
C_{s,A}=\frac{k_{m,A}C_{ref,A}+\Gamma_s k_{des}\theta_A}{k_{m,A}+\Gamma_s k_{ads}\theta_*^{m_{ads}}}
	3.	CsB_if_needed(CrefB, thetaA, theta_star, kmB, k_rxn, Gamma_s, C_B_scale, pA, pStar, hasB) -> CsB
m_B=1のときだけ
C_{s,B}=\frac{k_{m,B}C_{ref,B}}{k_{m,B}+\Gamma_s k_{rxn}\theta_A^{p_A}\theta_*^{p_*}/C_{B,scale}}
B無しなら CsB=nan か 0（どちらかに統一）
	4.	r_event(thetaA, theta_star, CsB, k_rxn, C_B_scale, pA, pStar, hasB) -> r_event
r_{event}=k_{rxn}\theta_A^{p_A}\theta_*^{p_*}\left(\frac{C_{s,B}}{C_{B,scale}}\right)^{m_B}

	•	hasB=False → m_B=0 → B項は1

	5.	diagnostics(phi_B, f_I, CsA_over_CrefA, CsB_over_CrefB)

	•	\phi_B = \frac{\Gamma_s k_{rxn}\theta_A^{p_A}\theta_*^{p_*}/C_{B,scale}}{k_{m,B}}
	•	f_I = \frac{1}{1+K_I C_{ref,I}}
	•	比率は 0除算回避（Crefが0ならNaN等）

⸻

STEP 5：暗黙Euler（θ 1変数）更新（bisection）

ここは 「壊れない」最短解です。

まず実装する関数（最小）
	•	implicit_euler_theta_step(theta_n, dt, f_theta, bracket=(0,1), tol, max_iter) -> theta_next
	•	方程式：g(theta_next)=theta_next - theta_n - dt*f(theta_next)=0
	•	bracketは [0,1]

ブラケットできない時のフォールバック（実務で必要）
	•	g(0) と g(1) が同符号で根が挟めないとき：
	•	theta_next = clip(theta_n + dt*f(theta_n), 0, 1) にフォールバック
	•	diag.non_bracketed += 1 を必ず記録
	•	できれば dt を自動で半分にして再試行（MVP後でOK）

まず「落ちない」ことが最優先です。精密さは後から上げられます。

⸻

STEP 6：forwardシミュレーション（steady → transient の順）

6-1. CVD steady forward（最初に通す）
	•	入力：CrefA/I/B は時間一定
	•	ループ：n_steps = ceil(t_proc/dt) でθとhを更新

実装関数
	•	simulate_steady(cfg) -> Outputs

6-2. transient forward（次に通す）
	•	入力：time[t], Cref[t]
	•	取り扱い（最短）：
	•	区間 [t_i, t_{i+1}] で Cref を piecewise constant とする
	•	dt_s でサブステップして積分（dt_s <= min(diff(time)) になるまで）

実装関数
	•	simulate_transient(cfg) -> Outputs

6-3. ALD transient（同じ関数でOK）
	•	ALDは time series に A/Bパルスが入っている前提なので、transientと同じで動く
（phaseドライバは作らない方針）

⸻

STEP 7：Output保存（NPZ）＋最低限のプロット

最短で “使える” 状態にするには、
	•	fields.npz
	•	metrics.json
	•	plots/*.png
	•	report.html（簡易でOK）
	•	results/<project>/index.html（入口）

の固定構造を先に作るのが効果的です。

まず出すべき最小プロット（文字だけスライド防止ではなく、現場確認用）
	•	thickness map（h_nm）
	•	residual map（measurementがある時）
	•	radial profile（中心→外周）
	•	phi_B map（Bがある候補のみ）
	•	f_I map（Iがある候補のみ）

⸻

STEP 8：measurement adapter（最小）＋損失関数（Huber）

最短は “同じxy点を持つ前提” で始め、後で補間対応に拡張。

実装関数
	•	load_meas_npz(path)-> MeasData(h_nm, xy)
	•	align_meas_to_sim(sim_xy, meas_xy, meas_h, align_cfg)-> meas_on_sim_xy
	•	最短：xy一致ならそのまま、一致しないなら最近傍（後で線形補間）
	•	loss_huber(residual_nm, delta)-> scalar

⸻

STEP 9：role列挙（I/B≤1・disjoint・unused許可）＋クラス分類

列挙ロジック（そのままコードに）

for sA in species:
  for sI in [None] + [s for s in species if s != sA]:
    for sB in [None] + [s for s in species if s not in {sA, sI}]:
        yield CandidateRoles(A=sA, I=sI, B=sB, class_id=...)

クラスID
	•	A: I=None, B=None
	•	AI: I!=None, B=None
	•	AB: I=None, B!=None
	•	AIB: I!=None, B!=None

⸻

STEP 10：order列挙（整数次数）＋3次制約
	•	YAMLの候補リストを読み込む（短いリストでOK）
	•	pA + pStar + (hasB?1:0) <= 3 を満たすものだけ残す

⸻

STEP 11：Optunaフィット（候補1つ → 全候補）

最短で壊さない順：
	1.	単一候補（roles固定、orders固定）で Optuna を回し、損失が下がることを確認
	2.	次に roles を数個列挙
	3.	最後に full（最大910×orders候補）へ

実装上のポイント（条件付きパラメータ）
	•	km_B は hasB の時だけ search space に入れる
	•	K_I は hasI の時だけ入れる
	•	それ以外のパラメータも “role有無” で条件を付ける

スコア（クラス比較のための複雑さ罰則）
	•	score = loss + lambda_role*(hasI + hasB) + regularization

⸻

STEP 12：ランキング出力＋クラス比較（A/AI/AB/AIB）
	•	ranking.csv：候補ごとの最良スコア、roles、orders、主要パラメータ、診断統計
	•	class_compare.csv：各クラスのbestスコアと差分（Δ）

⸻

B) Codex に渡す implement.prompt 雛形（テンプレ）

以下を prompts/implement.prompt.txt などに保存して使ってください。
**「1タスク=1論点」**で回すと破綻しにくいです。

# implement.prompt (template)
You are gpt-5.2-codex (reasoning=high). Implement the following task in this repository.

## NON-NEGOTIABLE REQUIREMENTS (do not violate)
- Core model is ONLY `aib_ode` (A/AI/AB/AIB are classes determined by roles I/B being None or not).
- Fluent species explosion is handled ONLY by hard role assignment:
  - A: required (single species)
  - I: None or single species (max_size=1)
  - B: None or single species (max_size=1)
  - A/I/B must be disjoint; unused species are allowed.
- B order is fixed to 0/1: if role.B is None -> m_B=0 else m_B=1.
- No power-law, no rational saturation (sat_inh), no TC root solver (no progress variable R). Do NOT reintroduce old MS-01..15 logic into the new path.
- ODE is the main engine. Use implicit Euler + bisection (theta in [0,1]) for robustness.
- Keep the implementation simple, readable, and extensible. No large refactors.
- Do not add new dependencies unless explicitly allowed. If you think a dependency is needed, stop and write a TODO + explanation.

## TASK
[WRITE TASK TITLE HERE]

## GOAL
[What to implement, in one paragraph]

## SCOPE
- Allowed dirs/files:
  - [list directories you are allowed to modify]
- Forbidden:
  - mass refactor
  - unrelated cleanup
  - adding complex new features not mentioned in GOAL

## CONTEXT (physics + equations)
We model wafer thickness using a single ODE state theta_A per spatial point.
Definitions:
- theta_star = (1 - theta_A) / (1 + K_I * CrefI)
- CsA = (kmA*CrefA + Gamma_s*k_des*theta_A) / (kmA + Gamma_s*k_ads*theta_star^m_ads)
- If hasB (role.B != None and m_B=1):
  CsB = (kmB*CrefB) / (kmB + Gamma_s*k_rxn*theta_A^pA*theta_star^pStar / C_B_scale)
  r_event = k_rxn*theta_A^pA*theta_star^pStar*(CsB/C_B_scale)
- Else (hasB=False, m_B=0):
  r_event = k_rxn*theta_A^pA*theta_star^pStar
ODE:
- dtheta_A/dt = k_ads*CsA*theta_star^m_ads - k_des*theta_A - nu_A*r_event
Thickness:
- dh/dt = alpha_h*Gamma_s*r_event

Constraints:
- theta_A must stay within [0,1]. Use implicit Euler step with bisection; fallback to clamped explicit step if bracketing fails (record diagnostics).
- Concentrations must be non-negative (clip Cref to >=0 with warning).

## YAML CONTRACT
Sim configs live under `configs/sim/` and include:
- sim.process: cvd|ald
- sim.time_mode: steady|transient
- sim.inputs.fluent: steady/transient NPZ with keys (cref, xy, time)
- sim.roles: fixed {A: s0, I: null, B: null}
- sim.model.name: aib_ode
- sim.model.orders: m_ads, pA, pStar with total order constraint
- sim.output: results/<project>/runs/<run_id>/...

Opt configs live under `configs/opt/` and include:
- role_enumeration (A required, I/B None or 1, disjoint, allow_unused)
- order_enumeration (finite candidates + total order constraint)
- class_compare (A/AI/AB/AIB best comparison + complexity penalty)
- parameter_fit using Optuna (no gradients)

## ACCEPTANCE CRITERIA
- [List concrete checks: files created, functions present, forward run works, etc.]

## VERIFICATION COMMANDS (run locally)
- [Put exact commands to run: smoke run, minimal tests]
- Ensure imports work (src layout pitfalls).

## OUTPUTS / ARTIFACTS
- [e.g., fields.npz, metrics.json, ranking.csv, class_compare.csv, report.html]

## IMPLEMENTATION NOTES
- Prefer pure functions for physics core.
- Add type hints where reasonable.
- Add minimal unit tests (shape checks, validator checks) if test framework exists.

Now implement the task. If anything is unclear in the repository (missing structure), inspect the repo and follow existing conventions. Do not invent new architecture beyond what is required for this task.

TIme Periodic
⸻

C) implement.prompt の “具体例”（最初のCodexタスクにおすすめ）

最短で前進するには、Codexに最初にやらせるタスクはこれが良いです：

Task-01：AIB‑ODE forward（CVD steady）を固定rolesで動かす
（optはまだ入れない。まず forward が回ることが最優先）

以下はそのまま貼れる例です。

# implement.prompt (example: Task-01)
You are gpt-5.2-codex (reasoning=high). Implement the following task in this repository.

## NON-NEGOTIABLE REQUIREMENTS (do not violate)
- Core model is ONLY `aib_ode` (A/AI/AB/AIB are classes determined by roles I/B being None or not).
- Hard role assignment only: A required single species; I/B are None or single species; disjoint; unused allowed.
- B order fixed 0/1 from role.B.
- No power-law, no sat_inh, no TC root solver R.
- ODE main engine: implicit Euler + bisection in [0,1].

## TASK
Implement forward simulation for `aib_ode` with fixed roles for CVD steady (no optimization yet).

## GOAL
Given a sim YAML config (configs/sim/cvd_steady_min.yaml), load Fluent NPZ (cref, xy), apply fixed roles (A/I/B), run steady-time integration using AIB-ODE (theta_A + thickness h), and write outputs under results/<project>/runs/<run_id>/ including fields.npz, metrics.json, and a minimal report.html + at least one plot (thickness_map.png).

## SCOPE
- Allowed dirs/files:
  - src/** (or existing package dirs)
  - configs/sim/**
  - scripts/** or an existing run entrypoint
- Forbidden:
  - deleting large existing modules
  - implementing optimization / Optuna in this task

## CONTEXT (physics + equations)
(Use the AIB-ODE equations from the template above.)

## YAML CONTRACT
Use `configs/sim/cvd_steady_min.yaml` contract:
- sim.inputs.fluent.mode=steady
- NPZ contains `cref` [n_pts, n_species], `xy` [n_pts,2]
- sim.roles.fixed: A is a species string in sim.inputs.fluent.species; I/B may be null.
- sim.time.t_proc_s and sim.time.dt_s define integration steps.

## ACCEPTANCE CRITERIA
- A command exists to run the sim:
  - Example: `python -m <pkg>.run_sim +sim=cvd_steady_min` OR `python scripts/run_sim.py +sim=cvd_steady_min`
- Running it produces:
  - results/demo/runs/<run_id>/config_resolved.yaml
  - results/demo/runs/<run_id>/outputs/fields.npz containing at least: h_nm, theta_A, theta_star, CsA_over_CrefA
  - results/demo/runs/<run_id>/plots/thickness_map.png
  - results/demo/index.html that links to the run report
- Validator checks: A required; I/B None or valid species; disjoint.
- No new dependencies are added.

## VERIFICATION COMMANDS
- `python -c "import <pkg>"`
- `python -m <pkg>.run_sim +sim=cvd_steady_min`

## OUTPUTS / ARTIFACTS
- fields.npz
- metrics.json (at least NU% or simple stats)
- thickness_map.png
- report.html and index.html

Now implement the task. Inspect the repository first and follow its conventions for package layout, Hydra config loading, and output directories.


⸻

D) おすすめ：Codexに投げるタスク分割（この順で最短・破綻しにくい）
	1.	Task‑01：AIB‑ODE forward（CVD steady, fixed roles）＋出力
	2.	Task‑02：transient対応（CVD/ALD共通）
	3.	Task‑03：measurement読み込み＋残差＋Huber loss（forward側に診断追加）
	4.	Task‑04：role列挙（A必須、I/B None-or-1、disjoint）＋クラス分類（A/AI/AB/AIB）
	5.	Task‑05：order列挙（整数次数＋3次制約）
	6.	Task‑06：Optunaフィット（候補1つ）
	7.	Task‑07：全候補×フィット×ranking.csv＋class_compare.csv＋report拡充

⸻

必要なら次の返信で、あなたの既存repoの **実行入口（python -m なのか、scripts/なのか、srcレイアウトか）**に合わせて、上の Task‑01 の VERIFICATION COMMANDS を “あなたのrepoで確実に通る形” に調整した版も出せます（repo情報が分かれば一発で合わせられます）。