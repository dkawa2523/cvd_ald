# CVD/ALD反応役割同化

本リポジトリは、匿名化されたFluent化学種場を解釈可能な反応役割へ写像し、反応役割に基づく表面モデルを実測膜分布へ当てはめる。数値当てはめだけでは化学的解釈を裏づけられない場合も明示する。

主経路は次のとおりである。

```text
Fluentの生の化学種
-> A / B / I 割当て候補
-> CVDまたはALD反応役割モデル
-> 実測膜厚または成膜速度への当てはめ
-> 条件間転移と空間分布の検証
-> 根拠不足を添えた採用・要検討・棄却
```

`s0`、`adn_2`、`n2` などの生の名称は入力表記である。データが役割を識別するまでは、既知の前駆体、共反応物、阻害種として扱わない。

## 現在の対象範囲

- 定常CVD方程式の網羅比較
  - 逐次AIB準定常応答
  - 並列AおよびA+B準定常応答
  - 二吸着種Langmuir–Hinshelwood応答
  - 阻害なし、有限損失なしなどの厳密縮約
- 動的プロセスモデル
  - 連続CVD AIB表面被覆率
  - Mars–van Krevelen酸化還元リザーバー
  - 観測時刻におけるMvK状態、経路速度、表面濃度、フラックス履歴
  - ALDの貯蔵、変換、阻害状態
- 輸送閉包
  - 与えた壁面濃度
  - 当てはめたスカラーまたは場の物質移動係数
  - CFD輸送容量フラックスから変換した物質移動係数
- 検証
  - 条件均衡を保った当てはめ
  - 一条件除外モデル選択
  - 同一実行内で再当てはめしない固定条件予測
  - 角度・半径ブロック診断
  - 役割、縮約、方程式系の安定性
  - 対象用途の準備状況と、それを確立するために必要な追加測定

## 主なコマンド

実装モデル一覧を表示する。

```powershell
uv run python scripts/analyze_cvd_multicond_case.py --list-models
```

現在の定常CVD評価を実行する。

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --response-model surface_compare `
  --reaction-input bulk_concentration `
  --models all `
  --loss mse `
  --sampler pattern `
  --bootstrap-samples 100 `
  --spatial-response radial_quartic `
  --seed 123 `
  --output results/current_cvd_separated
```

局所反応入力と、任意の選択後空間応答を明示的に選ぶ。

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --reaction-input bulk_concentration `
  --spatial-response radial_quartic `
  --wafer-temperature-k 773.15 `
  --output results/cvd_separated_response
```

全条件に文書化された化学種別列があれば、`surface_concentration` と `transport_capacity_flux` も利用できる。化学候補を列挙する前に入力位置を固定する。空間応答はその後に当てはめ、別に報告するため、選択された役割や反応式を変更できない。

各実行は、方程式系ごとの収束、化学種から役割への割当て、割当て安定性、入力相関、当てはめ役割感度、反応状態・経路分率、反応段階図、代替モデル予測差、運動論パラメータ感度、任意補正前後の空間残差を示す簡潔な図も出力する。

ウェハー全体の損失関数と表面サンプラーを明示的に選ぶ。

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --models all `
  --loss wafer_normalized_mse `
  --sampler pattern `
  --output results/cvd_wnmse
```

一つの固定した反応役割方程式で、損失関数とサンプラーを比較する。

```powershell
uv sync --extra optuna
uv run python scripts/benchmark_surface_optimization.py `
  --candidate-id "<model_id from role_ranking.csv>" `
  --trials 4096 `
  --repetitions 3 `
  --workers 8 `
  --output results/surface_optimization_benchmark_4096
```

比較試験は、`benchmark_report.md`、全実行・条件分割のCSV、損失関数とサンプラーを比較する二つのヒートマップを出力する。順位は学習条件間の転移で決め、固定テスト条件は選択後の監査にだけ使う。長時間実行では部分CSVをチェックポイントとして保存する。同じコマンドへ `--resume` を加えると、完了済み組合せを飛ばして再開する。`mse` は線形成膜速度単位の二乗誤差である。正の運動論形状パラメータは宣言範囲が複数桁に及ぶため、対数座標で標本化する。

Bash環境からリポジトリ検証を実行する。

```bash
./scripts/commands.sh smoke cvd
./scripts/commands.sh smoke ald
./scripts/commands.sh verify
```

運用用の当てはめ設定はOptuna TPEを使う。`fit` の前に任意の最適化実装方式を導入する。

```bash
uv sync --extra optuna
bash scripts/preflight.sh
# or: python -m pip install -e '.[optuna]'
./scripts/commands.sh fit cvd
./scripts/commands.sh fit ald
```

`random`、`tpe`、`cmaes`、差分進化（`de`）、粒子群（`pso`）、Lévy飛行（`levy`）、CMA-MAE（`cma_mae`）は同じサンプラー境界を使う。4種類のOptunaHub手法は任意依存で、レビュー済みの登録表の版へ固定している。実装方式がなければ導入手順を示して失敗し、指定サンプラーを暗黙に別手法へ置き換えない。CMA-MAEは多様な応答領域を探索するため、最小損失関数探索へ使う前に比較試験で評価する。

生成したシミュレーター入力は `runs/generated_inputs/`、実行出力は `results/` に置く。どちらも出典 データ保管場所ではない。

## 文書

- [技術報告書](docs/TECHNICAL_REPORT.md): モデルの基礎、数値手法、ソフトウェア設計、現行データの結論を統合した科学報告。
- [理論と方程式](docs/THEORY.md): 導出、近似、単位、モデル比較、参考文献。
- [評価ワークフロー](docs/EVALUATION_WORKFLOW.md): 入力に応じた実行・判定フロー、指標、数値処理。
- [可視化ガイド](docs/VISUALIZATION_GUIDE.md): 定常CVDの全図について、軸定義、出典成果物、解釈規則を示す例。
- [architecture](docs/ARCHITECTURE.md): パッケージ境界、登録表、責務、拡張規則。
- [現行データ評価](docs/CURRENT_DATA_EVALUATION.md): 与えた5条件CVDの再現可能な結果。
- [Fluent入力](docs/inputs_fluent.md) と [輸送方針](docs/transport_km.md): 場の意味と輸送仮定。
- [既知の課題](docs/GAPS.md): 現在のコード限界と、それを解消するために必要な測定。

各データ集合の解析で `data_requirements.csv` を出力する。ウェハー面内分布補正、匿名化学種の役割割当て、素反応速度推定の根拠が不足する場合、このファイルは各用途を評価可能にする測定と実験変動を示す。与えた5条件データは一つの実例であり、要件は化学種名ではなく根拠状態から導く。
