# docs

`architecture.png` — the figure used at the top of the README. Everything in it was measured
from the SenseVoice-Small ONNX graph rather than taken from a paper, so it should stay in
sync with the code. If you regenerate it, keep these facts straight:

| item | value |
| --- | --- |
| encoder input `x` | (N, T, 560) — 80-dim fbank, LFR window 7 / shift 6, CMVN applied |
| prompt embeddings | `embed.weight` [16, 560], selected by `language` / `text_norm` |
| special frames | 4, **prepended**: `[LID] [SER] [AED] [ITN]` |
| encoder | `encoders0` 1 layer + `encoders` 49 layers = **50 SAN-M blocks** |
| per block | self_attn (`linear_q_k_v` 512→1536) + FSMN depthwise conv (kernel 11) + FFN 512→2048→512 |
| hidden size | 512 |
| exposed tensor | `/encoder/tp_norm/Add_1_output_0`, (N, T, 512) — output of the final LayerNorm |
| CTC | `ctc_lo` 512 → 25055 (non-autoregressive; greedy decode, no LM decoder) |
| totals | 234 M params — int8 ≈ 228 MB, fp32 ≈ 895 MB; turn head 2048×1 + bias ≈ 30 KB |
| pooling (concat order in code) | all-frame mean, last-8-frame mean, last frame, special-frame mean |
| frame rate after LFR | 1 frame = 60 ms (10 ms × shift 6), so 8 frames ≈ 0.5 s |
| runtime constants | VAD pause 400 ms, min speech 500 ms, threshold 0.70, safety valve 5 s, last 8 s considered |

Known label fixes pending in `architecture.png`:

1. "(4) last **17th** frame" → **last frame** (index −1); the code uses `h[-1]`.
2. "(3) … utterance-end **suppression**" → it captures the **final intonation contour**
   (falling vs flat), which is what separates a finished utterance from a continuing one.
