# Fluent入力仕様

## 基本原則

反応モデルを選ぶ前に、各入力場の物理的意味を記録する。生の化学種名は配列ラベルであり、
反応役割ではない。濃度位置、フラックスの意味、座標単位、時間基準を明示する。

## 多条件定常CSV

標準の定常CVD解析では、次のファイルを条件ごとに組み合わせる。

```text
data/condition_<id>.csv
data/validation_<id>.csv
```

条件ファイルの必須列は次のとおりである。

| 列 | 意味 | 単位 |
| --- | --- | --- |
| `x`、`y`、`z` | サンプリング座標 | データ所有者が明示する |
| `concentration_<species>` | 指定したFluent位置の濃度 | kmol m\(^{-3}\) |

検証ファイルの必須列は次のとおりである。

| 列 | 意味 | 単位 |
| --- | --- | --- |
| `x`、`y`、`z` | 条件ファイルと対応する測定座標 | 条件座標と同じ |
| `dr_nm_per_sec` | 測定成膜速度 | nm s\(^{-1}\) |

条件ファイルには次の任意列を追加できる。

| 列 | 位置と意味 | 単位 |
| --- | --- | --- |
| `surface_concentration_<species>` | 同じ面内座標へ対応付けた反応表面直近濃度 | kmol m\(^{-3}\) |
| `transport_capacity_flux_<species>` | フィット対象反応から独立した境界条件で計算した、ウェハー向きの非負供給フラックス | kmol m\(^{-2}\) s\(^{-1}\) |
| `realized_reactive_flux_<species>` | 連成計算で得た実反応壁面フラックス。閉包比較専用 | kmol m\(^{-2}\) s\(^{-1}\) |
| `molef_<species>` | 整合確認用モル分率 | 1 |
| `density` | 混合気体密度 | kg m\(^{-3}\) |

検証ファイルへ点ごとの標準不確かさ `sigma_nm_per_sec` を追加できる。フィッティングに
選択した任意の化学種列は、全化学種・全条件に存在しなければならない。

標準ファイル名の代わりに `--conditions-file` を指定できる。これは条件IDと条件・検証
ファイルを対応付けたJSON 成果物目録であり、相対パスは成果物目録の位置から解決する。

## シミュレーション用NPZ

一般シミュレーターでは、YAMLでNPZキーを対応付ける。代表的なFluent設定を示す。

```yaml
inputs:
  fluent:
    mode: transient
    file: data/fluent_cvd_transient.npz
    keys:
      cref: cref
      xy: xy
      time: time
      flux_sink: flux_sink
    species: [s0, s1, s2]
domain:
  kind: from_fluent_xy
  xy_unit: mm
reference_plane:
  z_ref_mm: 1.0
```

配列形状は次のとおりである。

| 量 | 定常 | 過渡 |
| --- | --- | --- |
| `cref` | `[species, point]` または読込み部で定義した等価配置 | `[time, species, point]` または読込み部で定義した等価配置 |
| `xy` | `[point, 2]` | `[point, 2]` |
| `time` | なし | 単調増加する秒単位の `[time]` |
| `flux_sink` | 化学種／点の場 | 時刻／化学種／点の場 |

正確な配列方向は読込み部と解決済み設定を正とする。動的状態モデルでは、各時間区間に対応する
濃度フレームが必要である。

### MvK履歴観測

MvKシミュレーションは、Fluent時刻ごとの状態と経路量を保存する。NPZ測定へ酸化状態履歴を
加え、最終膜厚と同時に評価できる。

```yaml
measurement:
  enabled: true
  file: data/mvk_measurement.npz
  keys:
    xy: xy
    h: h_nm
    sigma: h_sigma_nm
    time: time_s
    oxidized_fraction_history: oxidized_fraction
    oxidized_fraction_history_sigma: oxidized_fraction_sigma
```

`time_s` はFluent時間配列と一致させる。履歴の形状は `[time, *space]` とし、対応する
`_sigma` キーを必須とする。履歴観測により多観測目的関数を使う場合、膜厚不確かさも必要
である。同じ規約で、`configs/sim/cvd_mvk_transient_min.yaml` に列挙されたMvK速度、
表面濃度、表面フラックス履歴を扱える。

状態と膜厚は各入力時刻に保存する。区間終端の速度・表面濃度・フラックスには、直前区間へ
適用した区分一定のFluentフレームを使い、初期値には最初のフレームを使う。

## 濃度位置ごとの入力能力

| 入力 | 意味 | 対応する定常輸送モード |
| --- | --- | --- |
| `bulk_concentration` | 参照位置またはバルク抽出位置の濃度 | `bulk_as_surface` 近似 |
| `surface_concentration` | 反応壁面直近で与えた濃度 | `direct_surface` |
| `transport_capacity_flux` | 定義済みの反応非依存境界条件におけるウェハー向き供給フラックス | 定常網羅評価の `direct_flux`、またはシミュレーション処理系で \(k_m\) を求める入力 |
| `realized_reactive_flux` | 連成CFDから得た実反応壁面フラックス | 比較または閉包観測専用 |

同じ反応モデルの \(k_m\) 推定に実反応フラックスを使ってはならない。反応応答を自分自身の
輸送境界として再利用することになるためである。

定常解析では、化学モデルを列挙する前に入力表現を一つだけ選ぶ。

```bash
--reaction-input bulk_concentration
--reaction-input surface_concentration
--reaction-input transport_capacity_flux
```

第1指定は `concentration_<species>` を表面入力の代用とし、第2指定は
`surface_concentration_<species>`、第3指定は
`transport_capacity_flux_<species>` を使う。これらを競合する反応機構として自動順位付け
しない。いずれの場合も、役割式へは無次元局所ドライバー
\(u_j=X_j/X_{j,\mathrm{ref}}\) を渡す。したがってフラックス駆動で得た係数群を、濃度に
対する吸着定数として報告してはならない。

定常ワークフローはウェハー温度一定を仮定する。`--wafer-temperature-k` は既知のスカラー
温度を記録するだけで、半径方向温度補正を生成またはフィットしない。

## 位置合わせと品質検査

フィッティング前に、定常変換部は次を確認する。

1. 条件ファイルと検証ファイルの行数。
2. 座標、濃度、速度が有限値であること。
3. 小数点以下6桁へ正規化した座標の一致と、正規化前の最大差。
4. 重複座標。
5. 正の参照濃度。
6. 利用可能な場合、モル分率和および濃度とモル分率の整合。
7. 一意値の数と最小正増分。
8. 条件内範囲、条件間対数幅、ランク、化学種間相関。
9. ホールドアウトが同定範囲外にある割合。

これらへの合格は、配列が数値的に利用可能であることを意味する。実験条件が反応役割を
区別できることまでは意味しない。

## 現在の `data/` で利用できる入力

現在の5条件データには `bulk_concentration` と定常成膜速度だけがある。時間配列、
座標単位、壁面濃度、輸送容量フラックス、温度系列、圧力、測定不確かさ、反復マップはない。
したがって解析は `bulk_as_surface` を使い、独立に検証された壁面変換や絶対フラックスを
計算できない。

定量結果は [CURRENT_DATA_EVALUATION.md](CURRENT_DATA_EVALUATION.md)、輸送式は
[transport_km.md](transport_km.md) を参照する。
