# docs

図をここに置きます。README から次の名前で参照しています:

| ファイル | 内容 |
|---|---|
| `architecture.svg` | 1パス2ヘッド（audio → encoder →〔CTC=文字起こし / turnヘッド=確率〕） |
| `internals.svg` | encoder内部とプーリング（50層SAN-M・特殊枠4フレーム・末尾8フレーム） |
| `runtime-flow.svg` | 実行時（VADポーズ → 判定 → 確定/継続 → 安全弁） |
| `training-pipeline.svg` | 学習（録音 → ポーズ地点で切る → 自動ラベル → ヘッド学習） |

置いたら README の該当箇所の `<!-- TODO: ... -->` を
`![...](docs/xxx.svg)` に差し替えてください。

## 図に使う実測値（ONNXから抽出したもの）

- 入力 `x`: (N, T, 560) = 80次元fbank × LFR(窓7/シフト6)、CMVN済み
- プロンプト埋め込み `embed.weight`: [16, 560]（language / text_norm で選択）
- 先頭に挿入される特殊枠 4フレーム: `[LID] [SER] [AED] [ITN]`
- encoder: `encoders0` 1層 + `encoders` 49層 = **50層** SAN-M
  - 各層: self_attn(`linear_q_k_v` 512→1536) + FSMN畳み込み(kernel 11) + FFN(512→2048→512)
  - hidden = 512
- 最終 `tp_norm` の出力 = **`/encoder/tp_norm/Add_1_output_0`** (N, T, 512) ← 露出させたテンソル
- CTC: `ctc_lo` 512 → **25055** 語彙
- 総パラメータ **234M**（int8で228MB / fp32で895MB）
- turnヘッド: 2048 → 1（+bias）= **30KB**
- プーリング内訳（各512次元を連結して2048）:
  1. 特殊枠4フレームの平均
  2. 実フレーム全体の平均
  3. 末尾8フレームの平均（LFR後1フレーム=60ms なので約0.5秒）
  4. 最終1フレーム
- 実行時の定数: VADポーズ 400ms / 最小発話 500ms / 閾値 0.70 / 安全弁 5秒 / 判定対象は直近8秒
