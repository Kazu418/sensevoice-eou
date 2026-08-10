# sensevoice-turn

**Get the transcript *and* "did they finish talking?" out of a single SenseVoice forward pass.**

The most annoying failure in a voice assistant is being **cut off mid-sentence**. A silence
timer can't tell the difference between *thinking* silence and *finished* silence — that
distinction lives in the words and in the intonation, not in the length of the gap.

This repo answers it using the **ASR encoder you are already running**. No second model is
loaded, so the turn decision is essentially free — small enough to run on a Raspberry Pi.

<!-- TODO: docs/architecture.svg — one pass, two heads -->

```
audio ──► SenseVoice encoder ──┬─► CTC projection ─► transcript  (as before)
        (one forward pass)     └─► turn head      ─► P(end of turn)
                                   └ a single 2048→1 linear layer (30 KB)
```

Same words, opposite decision — the model is listening to the **prosody**:

```
[END ] p=1.000  "電気消したいんだけど。"   (falling tone — done)  → commit
[WAIT] p=0.007  "電気消したいんだけど。"   (flat tone — more coming) → keep listening
```

## Why not just add a turn-detection model?

|                     | this repo                     | Pipecat Smart Turn v3 | LiveKit Turn Detector |
| ------------------- | ----------------------------- | --------------------- | --------------------- |
| Extra model         | **none** (shares the ASR)     | 8 MB / 8 M params     | 66 MB / 135 M params  |
| Extra latency       | **~0** (reuses the same pass) | separate inference    | separate inference    |
| Signals used        | semantics **+** prosody       | prosody (audio only)  | semantics + prosody   |
| Transcript          | **falls out of the same pass** | needs a separate ASR  | needs a separate ASR (usually cloud) |

Off-the-shelf detectors are trained to be speaker- and language-agnostic, which is exactly
why they miss *your* speech patterns. This repo is built around **retraining on your own
voice**: on our hardware a general model separated only 12.5% of minimal prosody pairs,
while a head trained on ~80 pairs of the target speaker reached 88%.

## How it works

The distributed SenseVoice-Small ONNX fuses the encoder and the CTC projection into one
graph whose only output is `logits`. The 512-dim frame embeddings we want are stuck inside.

Rather than re-exporting the model, we perform **graph surgery**: add the CTC projection's
input tensor as a second graph output. It works on both the fp32 and int8 checkpoints.

<!-- TODO: docs/internals.svg — encoder internals and pooling -->

What is actually inside (measured from the ONNX):

```
x (N, T, 560)              80-dim fbank, LFR window 7 / shift 6, CMVN applied
  + language / text_norm   selected from embed.weight [16, 560] (prompt embeddings)
        │
        │  4 special query frames are prepended:  [LID] [SER] [AED] [ITN]
        ▼
  encoders0 : 1 layer   ┐
  encoders  : 49 layers ┘  50 SAN-M blocks, hidden 512, FFN 2048, FSMN kernel 11
        ▼
  tp_norm  (final LayerNorm)
        ▼
  /encoder/tp_norm/Add_1_output_0   ◄── the tensor we expose, (N, T, 512)
        ├───────────────────────────┐
        ▼                           ▼
  ctc_lo (512 → 25055)          turn head
        ▼                           ▼
  logits → greedy decode        pool → 2048 → 1 → sigmoid
```

234 M parameters total; the turn head adds 30 KB.

**Pooling matters.** End-of-turn depends on how the utterance *ends*, and a plain mean over
all frames washes that out. We keep the tail thick:

```
hidden (T, 512)
  ├ [0:4]  special frames  ─► mean ─┐ 512   LID/SER/AED queries — carry paralinguistics
  └ [4:]   speech frames            │
       ├ all frames        ─► mean ─┤ 512   what was said
       ├ last 8 frames     ─► mean ─┤ 512   the final ~0.5 s — the intonation contour
       └ last frame        ─────────┘ 512   the very end
                                concat = 2048
```

One post-LFR frame is 60 ms, so eight frames ≈ 0.5 s.

## Install

```bash
git clone https://github.com/<you>/sensevoice-turn && cd sensevoice-turn
pip install -r requirements.txt

# Get the SenseVoice ONNX (sherpa-onnx distribution)
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# Expose the encoder output (once, takes seconds)
python -m sensevoice_turn.expose \
  sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  model.int8.encout.onnx
```

## Use

```bash
python -m sensevoice_turn.infer sample.wav \
  --model model.int8.encout.onnx \
  --tokens sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt
```

```python
from sensevoice_turn import SemanticTurn

st = SemanticTurn("model.int8.encout.onnx", "models/turn_head.npz", "tokens.txt")
r = st(audio)          # float32, 16 kHz, mono
r["probability"]       # P(end of turn)
r["is_complete"]       # above the calibrated threshold?
r["text"]              # transcript from the same pass — don't run ASR twice
```

Call it when your VAD reports a pause:

<!-- TODO: docs/runtime-flow.svg — VAD pause → decide → commit / keep listening -->

```python
if vad.pause_after_speech(400):        # 400 ms of silence
    r = st(audio_so_far)
    if r["is_complete"]:
        handle(r["text"])              # commit; transcript already in hand
    else:
        keep_listening()               # they're still thinking
```

Runtime loop as deployed:

```
mic → 32 ms frames → silero VAD
                        │  speech ≥ 500 ms, then 400 ms of silence = pause
                        ▼
              one encoder forward pass          ← the only place it runs
                   ├─ transcript
                   └─ p
                        ▼
              p ≥ 0.70 ?  ── yes ─► commit (reuse the transcript)
                        │
                        no ─► keep listening, decide again at the next pause
                        ▼
              safety net: force-commit after 5 s of continuous silence
```

The bundled `models/turn_head.npz` was trained on **Japanese, single speaker**. It works as
a starting point, but training on your own voice is where the accuracy comes from.

## Train on your own voice

No GPU. **CPU only, a few minutes** — it runs on a Raspberry Pi.

### 1. Collect recordings

```bash
python -m sensevoice_turn.collect --out data/recordings
# open http://localhost:8100
```

Hold the button to record, release to stop, tap to save. **Microphone access requires HTTPS
or localhost** — to record from a phone, expose it via Tailscale, ngrok, or similar.

The category that matters most is **paired recording**:

> Say the same sentence twice: once **⤵ with a falling, finished tone**, once **→ flat, as if
> a condition is coming next**. The transcript is identical, so the model cannot cheat on
> wording — it is forced to listen to the intonation.

This was the decisive ingredient. Trained without pairs, the head separated 2 of 16 pairs.
With pairs, 28 of 32.

Aim for **40–80 pairs**, plus ~30 clips of each other category.

Two things learned the hard way:

- **Don't use sentences that can't naturally continue.** In Japanese, "〜したい" / "〜かな"
  (desiderative) are already complete; forcing a "flat" take produces *fake* prosody and
  teaches the model a lie. Dropping them measurably improved accuracy.
- Forms that work well in both readings: "〜なんだけど" (concessive), bare "〜して"
  (imperative that can be chained), and noun-final phrases.

### 2. Build the dataset

```bash
python -m sensevoice_turn.build --rec data/recordings --out data/dataset
```

This step quietly matters: it makes training data **shaped like inference input**.

The naive approach trains on whole utterances while inference sees a *partial* utterance
ending at a pause — a train/serve skew. Instead we replay each recording through the VAD and
cut a sample at **every pause where the detector would actually fire**. Labels come for free:

```
speech still remains after this pause  → noturn   (must not commit here)
nothing after it (final pause)         → the recording's own label
```

So the mid-utterance pause in "turn the volume … (pause) … up to 30%" becomes a realistic
negative, with no synthetic splicing.

<!-- TODO: docs/training-pipeline.svg — recordings → pause-point cuts → auto labels → head -->

### 3. Train

```bash
python -m sensevoice_turn.train --data data/dataset --model model.int8.encout.onnx
```

The encoder stays frozen; only the linear head is fitted. Afterwards a **threshold sweep**
prints the trade-off and the recommended value is baked into `models/turn_head.npz`:

```
thresh |  cut off early | waited too long
0.50   |      5.1%      |      5.8%
0.70   |      2.0%      |      5.8%    ← free improvement, recommended
0.80   |      1.0%      |      8.7%
0.95   |      0.0%      |     15.9%
```

The two errors are not equally bad. Being **cut off early** forces the user to repeat
themselves; **waiting too long** just feels sluggish. Tune accordingly.

### 4. Check it end to end

```bash
python -m sensevoice_turn.eval --rec data/recordings --data data/dataset \
  --model model.int8.encout.onnx --val-only
```

Clip accuracy is not the number you care about. This replays whole recordings through the
real VAD loop and reports **per-utterance** outcomes — including whether the model ever
committed *while the speaker was still going*, which is the failure users actually notice.

## Measured results

Hardware: **Raspberry Pi 5 (4 cores, 16 GB)**, `model.int8.encout.onnx`, onnxruntime CPU.

### End-to-end, per utterance

Clip-level accuracy flatters a turn detector, because at runtime an utterance passes
**several** pauses and the **first** "commit" ends it — being right later doesn't help. So
this is measured by replaying whole recordings through the real VAD loop, on **held-out**
recordings only (82 utterances; their sentences never appear in training).

| | rate |
| --- | --- |
| **Cut off early** (still talking, committed anyway) | **0.0%**  (0/82) |
| False commit (should have waited, committed by the end) | 2.4%  (2/82) |
| Missed (finished, but never committed → waits for the safety net) | 4.9%  (4/82) |
| **Per-utterance success** | **92.7%**  (76/82) |

The failure that actually hurts — getting cut off mid-sentence — did not happen once.

For reference, the same held-out clips scored per-fragment: accuracy 94.6%, cut-off 2.0%,
waited 5.8%; and a general-purpose baseline separated only 2/16 minimal prosody pairs versus
**28/32** here.

### Latency (median over 40 clips, median clip 2.0 s)

| threads | median | p90 | max | RTF |
| --- | --- | --- | --- | --- |
| 1 | 231 ms | 739 ms | 871 ms | 0.11 |
| **2** | **163 ms** | 476 ms | 588 ms | **0.078** |
| 4 | 191 ms | 492 ms | 617 ms | 0.090 |

Two threads is the sweet spot on a Pi 5 — four threads contend with the rest of the system
and get slower. Peak RSS ≈ 1.1 GB with the model loaded; the int8 ONNX is 228 MB on disk and
the turn head is 9 KB.

Because the head rides along with an ASR pass you were going to run anyway, the *marginal*
cost of the turn decision is the pooling and a dot product — microseconds.

## Limitations

- The bundled head is **one speaker, Japanese**. Retrain for other speakers or languages
  (SenseVoice itself covers zh/en/ja/ko/yue, so the same recipe applies).
- The encoder runs at every pause, so cost grows with utterance length. By default only the
  **last 8 seconds** are considered (`max_tail_sec`).
- Training recordings are **not included** — voice data is personal. Record your own.

## License and credits

- Code in this repository: MIT
- **SenseVoice / SenseVoiceSmall**: [FunAudioLLM](https://github.com/FunAudioLLM/SenseVoice) —
  the model itself is governed by the
  [FunASR license](https://github.com/modelscope/FunASR?tab=readme-ov-file#license). Download it yourself.
- ONNX distribution and export script: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Xiaomi Corp., Fangjun Kuang)
- VAD: [silero-vad](https://github.com/snakers4/silero-vad)

Jointly modelling ASR and endpointing is not a new idea — Google shipped end-of-query
prediction inside on-device ASR years ago. What's awkward today is that open voice stacks
put the ASR behind a cloud API, so they *can't* reach into the encoder and have to bolt on a
second model. This repo does the joint version with a **local, open ASR, no extra model, and
your own data**.
