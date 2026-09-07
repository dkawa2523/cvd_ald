# 文書案内

以下の文書は、現在の実装を説明するためのものである。設計判断の経緯は
`docs/adr/` に残しているが、現行のモデル仕様および評価手順より優先されるものでは
ない。

| 文書 | 責務 |
| --- | --- |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | 科学的根拠、実装、現時点の運用判断をまとめた総合技術報告 |
| [THEORY.md](THEORY.md) | 支配方程式、縮約、仮定、単位、参考文献 |
| [EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md) | 入力に応じた実行、最適化、検証、判定の流れ |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | 定常CVDで生成する全図の例、軸・色の定義、元データ、解釈方法 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | パッケージ境界、モデル登録、拡張点、出力の所有責務 |
| [CURRENT_DATA_EVALUATION.md](CURRENT_DATA_EVALUATION.md) | `data/` にある定常CVD 5条件の再現可能な評価 |
| [CONTEXT.md](CONTEXT.md) | 分野用語と主張できる範囲 |
| [inputs_fluent.md](inputs_fluent.md) | Fluent場の意味、必要単位、入力から評価できる事項 |
| [transport_km.md](transport_km.md) | 参照面から壁面までの輸送閉包と適用限界 |
| [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) | 日常的な評価コマンドと合否確認 |
| [GAPS.md](GAPS.md) | ソフトウェア上の限界と、それを解消するために必要な測定 |
| [REQUIREMENTS.md](REQUIREMENTS.md) | 反応役割同化に関する製品要件 |

科学的なレビューでは、まず `TECHNICAL_REPORT.md` を読み、モデルの詳細を
`THEORY.md`、実行と判定の意味を `EVALUATION_WORKFLOW.md`、今回の正確な数値を
`CURRENT_DATA_EVALUATION.md` で確認する。図の軸、表現、元データ、図から許される
結論は `VISUALIZATION_GUIDE.md` にまとめた。
[生成レポート](../results/current_cvd_separated/report.md)は個別実行の成果物であり、
一般仕様を定める文書ではない。

## リポジトリ限定のCodex Skills

保守対象の手順は、このリポジトリの `.agents/skills/` にだけ置く。各Skillの責務は
次のように分離している。

| Skill | 用途 |
| --- | --- |
| `evaluate-steady-cvd-roles` | 定常方程式を網羅評価し、最適化、役割、機構、ウェハーマップ、空間応答、図を解釈する |
| `evaluate-transient-ald-roles` | 過渡ALDの蓄積・転化役割を当てはめ、レシピ間転用性と適合診断を評価する |
| `run-reaction-role-state-models` | 指定パラメータのCVD AIB、CVD MvK、ALD状態モデルを実行し、状態、フラックス、単位、図を点検する |
| `extend-reaction-role-models` | 現行構成の中で、方程式、輸送、フィッティング部品、証拠成果物、科学図を変更する |

詳細な実行手順と図の読み方は各Skillの参照資料に分けている。ひとつの手順を読み込む
だけで、無関係なプロセスモードまで文脈に入らない構成である。

## 仕様の優先順位

モデルの動作については実行可能なコードを正とする。`THEORY.md` は実装済みの式を
読みやすい形で記述し、`EVALUATION_WORKFLOW.md` は選択・検証の意味、
`ARCHITECTURE.md` はファイル責務を定める。`results/*/report.md` は一回の実行結果で
あり、一般的なモデル仕様として扱わない。

モデルの意味、公開設定、採用規則を変更した場合は、同じ変更の中で該当する現行文書も
更新する。ADRは判断履歴として保持するが、現行文書と矛盾する箇所は廃止済みまたは
置換済みとして解釈する。
