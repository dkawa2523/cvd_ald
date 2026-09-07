# ADR 0011: 出力・可視化仕様V1（`output.v1`）

- 日付: 2026-02-23
- 状態: 採択
- 範囲: AIB 実行系出力仕様、報告書 リンク、可視化メタデータ

## 背景

AIB移行後、出力配置が実行処理（`sim`、`doe`、`benchmark`、`fit`）ごとに分かれ、報告書 リンクの一部が固定記述されていた。追跡性が下がり、レビューと自動処理が壊れやすくなっていた。

## 判断

1. すべての主実行処理に厳密な `outputs/manifest.json` を導入し、次を含める。
   - `schema_version = "output.v1"`
   - `run_id`、`mode`、`created_at_utc`
   - `artifacts[]` 記録（`id`、`path`、`kind`、`required`）
   - `plots[]` 記録（`plot_id`、`path`、`source_key`、`cmap`、`discrete`）
2. `summary.json` は領域指標を保持し、`manifest_path` と成果物目録から導いた `artifact_paths` で成果物を参照する。
3. 報告書の成果物リンクは成果物目録 記録から生成し、実行処理別の固定記述一覧を持たない。
4. 可視化形式は静的PNGとHTMLを維持する。
5. `from_fluent_xy` の主マップ表示には三角形分割を用い、散布図は代替処理に限る。

## 破壊的変更と互換方針

- 任意形式の `summary.artifact_paths` を読む利用側に対する破壊的変更となる。
- 互換性は明示的な成果物目録対応だけで提供し、同じ出力を別名で長期重複させない。

## 失敗時の扱い

- 必須成果物目録 キーの欠落は警告ではなく誤差とする。
- 不正な成果物目録 内容は検証課題と試験を失敗させる。
