# docs

図をここに置きます。README から次の名前で参照しています:

| ファイル | 内容 |
|---|---|
| `architecture.png` | **配置済み**。全体図(1パス2ヘッド + encoder内部 + プーリング + 実行時フロー) |
| `training-pipeline.svg` | (未作成) 学習（録音 → ポーズ地点で切る → 自動ラベル → ヘッド学習） |

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

## architecture.png の既知の誤り（差し替え時に修正）

1. 「(4) last **17th** frame」→ 正しくは **last frame (index −1)**。コードは `h[-1]`。
2. 「(3) ... useful for utterance-end **suppression**」→ 抑制ではなく、
   **語尾の抑揚(下降/平坦)を捉える** の意。`captures the final intonation contour` 等が適切。

※ プーリングの連結順はコード上 `[全体平均, 末尾8, 最終, 特殊枠]`。図の番号は説明用の並び。
