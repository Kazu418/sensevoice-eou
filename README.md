# sensevoice-turn

**SenseVoice の 1 回の推論から「文字起こし」と「話し終わったか（end-of-turn）」を同時に得る。**

音声アシスタントで一番いらつくのは、**言い終わっていないのに切られる**ことです。
VAD の無音タイマーだけでは「考えている間の沈黙」と「言い終わった沈黙」を区別できません。

このリポジトリは、**すでに動かしている ASR の encoder をそのまま使って**ターン判定します。
追加のモデルを常駐させないので、Raspberry Pi のような小さな機械でも実質ゼロコストです。

```
音声 ──► SenseVoice encoder ──┬─► CTC 射影    ─► 文字起こし（従来どおり）
        （1 回だけ前進）        └─► turn ヘッド ─► 「話し終わったか」の確率
                                    ← 追加は 2048→1 の線形 1 層（30KB）
```

同じ文字起こしでも、**語尾の抑揚だけで判定が反転します**:

```
[END ] p=1.000  「電気消したいんだけど。」   ← 語尾を下げた（言い切り）→ 確定してよい
[WAIT] p=0.007  「電気消したいんだけど。」   ← 語尾は平坦（まだ続く）→ 待つべき
```

## なぜ別モデルではなくこれなのか

| | このリポジトリ | Pipecat Smart Turn v3 | LiveKit Turn Detector |
|---|---|---|---|
| 追加モデル | **なし**（ASR と共有） | 8MB / 8M params | 66MB / 135M params |
| 追加レイテンシ | **ほぼ 0**（同じ前進を再利用） | 別モデルの推論が必要 | 別モデルの推論が必要 |
| 判定に使う情報 | 意味 **＋** 抑揚 | 抑揚（音のみ） | 意味 ＋ 抑揚 |
| 文字起こし | **同じパスで一緒に出る** | 別途 ASR が必要 | 別途 ASR（多くはクラウド） |

汎用モデルは多言語・不特定話者向けに作られているため、特定の用途では取りこぼします。
このリポジトリは**自分の声・自分のコマンドで学習し直せる**ことを前提にしています
（手元の実測では、汎用モデルで 12.5% しか区別できなかった抑揚ペアが 88% になりました）。

## 仕組み

SenseVoice-Small の配布 ONNX は encoder と CTC 射影が 1 グラフに融合していて、
出力は `logits` だけです。ターン判定に使いたいフレーム埋め込みは内部に隠れています。

そこで**再 export せず、既存の ONNX に出力を 1 本足す**「手術」をします
（CTC 射影の入力テンソルをグラフ出力に追加するだけ。fp32 / int8 の両方で動きます）。

あとはその埋め込みを 1 ベクトルに畳んで、小さな分類器を載せるだけです。
語尾の抑揚を見る必要があるので、全体平均だけでなく**末尾を厚く**取ります:

```
[全体平均, 末尾 8 フレーム平均, 最終フレーム, 特殊枠平均] = 512 × 4 = 2048 次元
```

特殊枠は SenseVoice が言語 ID / 感情 / 音響イベント 用に使うクエリで、
プロソディの情報を持っているため一緒に入れています。

## セットアップ

```bash
git clone https://github.com/<you>/sensevoice-turn && cd sensevoice-turn
pip install -r requirements.txt

# SenseVoice の ONNX を取得（sherpa-onnx 配布版）
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# encoder 出力を生やす（一度だけ・数秒）
python -m sensevoice_turn.expose \
  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  model.int8.encout.onnx
```

## 使う

```bash
python -m sensevoice_turn.infer sample.wav \
  --model model.int8.encout.onnx \
  --tokens sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt
```

```python
from sensevoice_turn import SemanticTurn

st = SemanticTurn("model.int8.encout.onnx", "models/turn_head.npz", "tokens.txt")
r = st(audio)                    # float32, 16kHz, mono
r["probability"]   # 話し終わった確率
r["is_complete"]   # 閾値を超えたか
r["text"]          # 同じ推論で得た文字起こし（再度 ASR を回す必要はない）
```

VAD が無音を検出したタイミングで呼ぶ想定です:

```python
if vad.pause_after_speech(400):          # 400ms の無音
    r = st(audio_so_far)
    if r["is_complete"]:
        handle(r["text"])                # 確定（文字起こしは取得済み）
    else:
        keep_listening()                 # まだ考えている
```

同梱の `models/turn_head.npz` は**日本語・単一話者**で学習したものです。
そのままでも動きますが、**自分の声で学習し直すと大きく良くなります**（次項）。

## 自分の声で学習する

GPU は要りません。**CPU だけで数分**で終わります（Raspberry Pi でも可）。

### 1. 録音を集める

```bash
python -m sensevoice_turn.collect --out data/recordings
# → http://localhost:8100
```

ブラウザで押している間だけ録音し、カテゴリを付けて保存します。
**マイクは HTTPS か localhost でしか使えません**（スマホから使うなら Tailscale / ngrok 等）。

いちばん大事なのは **ペア録音** です:

> 同じ文を「**⤵ 語尾を下げて言い切る**」と「**→ 語尾を平坦に（まだ続ける気持ち）**」の
> 2 通りで録る。**文字列が同じ**なので、モデルは意味では区別できず、
> **音（抑揚）を聞くしかなくなります。**

これが決定的でした。ペア無しで学習したモデルは抑揚ペアを 16 組中 2 組しか区別できず、
ペアを入れたら 32 組中 28 組になりました。

目安は **ペア 40〜80 組**、他のカテゴリは各 30 件程度。

コツ:

- 「〜したい」「〜かな」で終わる文は**ペアに使わない**。言い切りの形なので、
  継続として自然に演じられず、**嘘の抑揚**を教えてしまいます（実際に外したら精度が上がりました）。
- 「〜なんだけど」「〜して」「体言止め」は、言い切りにも継続にも自然になるので向いています。

### 2. データセットを作る

```bash
python -m sensevoice_turn.build --rec data/recordings --out data/dataset
```

ここが地味に効きます。**推論時と同じ形にする**ためです。

素朴にやると「学習＝発話全体 / 実行時＝ポーズまでの途中音声」で入力分布がずれます。
そこで録音を VAD に流し、**判定が走るポーズ地点ごとに切って** 1 サンプルにします。
ラベルは自動で付きます:

```
そのポーズの後にまだ発話が残っている → noturn（ここで確定してはいけない）
後に発話が無い（最後のポーズ）        → その録音のラベル
```

これで「音量を…（ポーズ）…30% にして」の**中間ポーズ**が、
人工的な切り貼り無しに現実的な負例になります。

### 3. 学習する

```bash
python -m sensevoice_turn.train --data data/dataset --model model.int8.encout.onnx
```

encoder は凍結したまま、その上の線形 1 層だけを学習します。
学習後に**閾値スイープ**（早切れと待ちすぎのトレードオフ表）が出て、推奨値が `models/turn_head.npz` に埋め込まれます。

```
閾値 |  早切れ(切りすぎ) | 待ちすぎ
0.50 |        5.1%       |    5.8%
0.70 |        2.0%       |    5.8%   ← ここまで無料で改善（推奨）
0.80 |        1.0%       |    8.7%
0.95 |        0.0%       |   15.9%
```

**早切れ**（まだ話しているのに切る）と**待ちすぎ**（言い終わったのに待つ）は、
体感の重さが違います。切られると言い直しになるので、早切れを重く見るのが普通です。

## 実測（Raspberry Pi 5 / int8 / 2 スレッド）

| | 汎用モデル（比較用） | 自分の声で学習後 |
|---|---|---|
| 抑揚ペアの判別 | 2 / 16 組 | **28 / 32 組** |
| 早切れ | 33.3% | **2.0%** |
| 待ちすぎ | 0% | 5.8% |
| 正解率 | 84.9% | **94.6%** |
| レイテンシ | — | 中央値 326ms / p90 559ms |

検証データは**学習に一切使っていない実録音**（考え込み・フィラー・ペア）で、
ペアは同じ文が train と val の両方に出ないよう**文単位で分割**しています。

## 制限

- **単一話者・日本語**で学習した重みを同梱しています。他の言語・話者では学習し直してください
  （SenseVoice 自体は中/英/日/韓/粤に対応しているので、素材さえあれば同じ手順で学習できます）。
- ターン判定は**ポーズのたびに encoder を前進**させます。発話が長いとコストが伸びるため、
  既定では**直近 8 秒**だけを見ます（`max_tail_sec`）。
- 学習に使う録音は**自分で用意する必要があります**。同梱していません（声は個人情報です）。

## ライセンスと謝辞

- このリポジトリのコード: MIT
- **SenseVoice / SenseVoiceSmall**: [FunAudioLLM](https://github.com/FunAudioLLM/SenseVoice) — モデル本体のライセンスは
  [FunASR](https://github.com/modelscope/FunASR?tab=readme-ov-file#license) に従います。モデルは各自で取得してください。
- ONNX 配布と export スクリプト: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（Xiaomi Corp., Fangjun Kuang 氏）
- VAD: [silero-vad](https://github.com/snakers4/silero-vad)

「ASR と終端検出を同時に学習する」という発想自体は新しくありません
（Google のオンデバイス ASR は以前から end-of-query を統合しています）。
このリポジトリの狙いは、それを**ローカルの公開 ASR で、追加モデルなしに、
自分のデータで**できるようにすることです。
