# 検証および評価手順

本手順は、コード変更時と新しい科学評価時に必要な確認項目を定める。一般検証の入口は
`scripts/commands.sh` とする。定常反応役割の網羅評価には、訓練・評価分割と乱数シードが
明示される専用コマンドを用いる。

## 通常のコード検証

Bash環境から次を実行する。

```bash
./scripts/commands.sh smoke cvd
./scripts/commands.sh smoke ald
./scripts/commands.sh test
./scripts/commands.sh verify
```

読込みに副作用がなく、最小設定のシミュレーションが完了し、数値試験が成功し、実行結果に
解決済み設定、場、指標、図、有効な `output.v1` 成果物目録が含まれる場合に合格とする。
モデル固有の変更では、最も近い単体試験も実行する。

| 変更箇所 | 対象試験 |
| --- | --- |
| 定常方程式または選択 | `deposim_opt.test_surface_kinetics`、`deposim_opt.test_cvd_multicond_analysis` |
| 目的関数と不確かさの尺度 | `deposim_opt.test_objective` |
| 動的AIB | `deposim_sim.test_aib_ode`、`deposim_sim.test_pipeline_aib` |
| Mars–van Krevelen | `deposim_sim.test_mvk_state`、`deposim_sim.test_process_models` |
| ALD状態 | `deposim_sim.test_ald_role_state` |
| 輸送位置または係数 | `deposim_sim.test_transport_provider` |
| 設定の意味 | `deposim_schema.test_sim_config_v2`、`tests/test_sim_config_compose.py` |

## 現行の定常データ評価

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

処理が完了しただけでは合格としない。生成された証拠を次の順で確認する。

1. 全入力行と座標が品質検査を通過している。除外がある場合はその内容が列挙されている。
2. 固定ホールドアウトがフィッティング、正規化、モデル選択に使われていない。
3. 適用可能な全方程式族、厳密縮約、重複しない役割割当てが含まれている。
4. 条件再フィット誤差で選択し、訓練条件だけから求めた基準モデルを明示している。
5. 平均誤差、中心化面内誤差、相関、バイアス、分布幅再現率を別々に報告している。
6. 外側条件分割により、選択手順全体の安定性を示している。
7. 外挿、縮約の証拠、係数不確かさ、モデル構造による予測幅を報告している。
8. `data_requirements.csv` に、未解決用途を評価可能にする測定と制御変数が示されている。
9. 方程式族、条件コントラスト、反応経路、代替モデル予測、役割重要度・安定性、
   パラメータ感度、ホールドアウト面内分布、半径シェル、モデル構造、任意の空間応答の図が
   生成され、元CSVと照合されている。
10. 最終状態が `adopt`、`review`、`reject` のいずれかであり、採用用途が検証済みの
    証拠範囲を超えていない。

## 動的モデルの評価

動的AIB、MvK、ALDは、まず状態範囲、物質移動閉包、時間刻み収束、生成・消滅項の符号を
確認する。科学的なモデル比較には時間分解観測も必要である。定常成膜速度CSVは極限応答を
確認できるが、リザーバーまたは蓄積の動力学を順位付けできない。

動的機構を主張する場合、次を必須とする。

- 初期状態とその調製方法を明示する。
- 入力時刻、濃度位置、温度、圧力が既知である。
- 全モデルへ同じ観測演算子を適用する。
- 完全な切替、ドーズ/パージ、またはサイクル系列を少なくとも一つ留保する。
- フィットした状態時定数を時間サンプリング間隔で分解できる。
- A還元、B再生、転化、輸送フラックスを互換単位で報告する。

## 文書および成果物の検証

リリース前に次を確認する。

- `git diff --check` に空白エラーがない。
- 現行Markdown文書の相対リンクが解決する。
- `THEORY.md` の式が実行コードの応答と一致する。
- 入力ハッシュまたは選択規則が変わった場合、`CURRENT_DATA_EVALUATION.md` を更新する。
- レポート用図が宣言した実行結果から複製され、目視確認されている。
- 作業用生成物は `results/` に置き、選択した報告用図だけを `docs/assets/` に保存する。
