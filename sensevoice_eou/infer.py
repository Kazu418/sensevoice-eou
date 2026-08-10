"""1パス2ヘッド推論 — 文字起こしとターン判定を encoder の1回の前進から同時に取る。

    audio ──► SenseVoice encoder ──┬─► CTC 射影      ─► 文字起こし
                                   └─► turn ヘッド   ─► 「話し終わったか」の確率

ターン判定のために別モデルを走らせないので、追加コストはほぼゼロ
(ヘッドは 2048→1 の線形1層、29KB)。

    from sensevoice_eou import SemanticTurn
    st = SemanticTurn("model.int8.encout.onnx", "models/turn_head.npz")
    r = st(audio_float32_16k)
    r["probability"], r["text"]
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .frontend import load_cmvn_from_onnx, wav_to_features

SR = 16000
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")


def pool(hidden: np.ndarray, n_special: int = 4, tail_frames: int = 8) -> np.ndarray:
    """語尾を厚く見る埋め込み(学習時と必ず同じにすること)。

    ターン判定は文の意味だけでなく**語尾の抑揚**で決まる。同一テキストで
    「〜だけど⤵(言い切り)」と「〜だけど→(まだ続く)」を区別するには全体平均だけでは
    足りないので、末尾フレームと特殊枠(LID/SER/AED 用のクエリ。プロソディを持つ)も足す。

      [全体平均, 末尾8フレーム平均, 最終フレーム, 特殊枠平均] = 512*4 = 2048次元
    """
    sp = hidden[:n_special] if hidden.shape[0] > n_special else hidden
    h = hidden[n_special:] if hidden.shape[0] > n_special else hidden
    tail = h[-tail_frames:] if h.shape[0] >= tail_frames else h
    return np.concatenate([h.mean(0), tail.mean(0), h[-1], sp.mean(0)]).astype(np.float32)


class SemanticTurn:
    def __init__(self, model: str, head: str | None = None, tokens: str | None = None,
                 threshold: float | None = None, threads: int = 2,
                 max_tail_sec: float = 8.0, language: str = "ja"):
        self.max_tail_sec = max_tail_sec
        self.cmvn = load_cmvn_from_onnx(model)
        self.lang_id = self.cmvn["langs"].get(f"lang_{language}", self.cmvn["langs"]["lang_auto"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        if len(self.sess.get_outputs()) < 2:
            raise SystemExit(
                "この ONNX には encoder 出力がありません。先に "
                "`python -m sensevoice_eou.expose <model.onnx> <out.onnx>` を実行してください")

        self.W = self.b = None
        self.n_special, self.tail_frames = 4, 8
        self.threshold = 0.70 if threshold is None else threshold
        if head:
            d = np.load(head)
            self.W, self.b = d["W"].astype(np.float32), float(d["b"])
            self.n_special = int(d.get("n_special", 4))
            self.tail_frames = int(d.get("tail_frames", 8))
            if threshold is None and "threshold" in d:
                self.threshold = float(d["threshold"])

        self.id2tok: dict[int, str] = {}
        if tokens and Path(tokens).exists():
            for line in Path(tokens).read_text(encoding="utf-8").splitlines():
                p = line.split(" ")
                if len(p) == 2:
                    self.id2tok[int(p[1])] = p[0]

    def _decode(self, logits: np.ndarray) -> str:
        if not self.id2tok:
            return ""
        ids = logits[0].argmax(-1)
        out, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                out.append(int(i))
            prev = i
        txt = "".join(self.id2tok.get(i, "") for i in out)
        return _SPECIAL_RE.sub("", txt).replace("▁", " ").strip()

    def embed(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """encoder を1回前進させ (logits, 埋め込み) を返す。"""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.max_tail_sec > 0:
            keep = int(self.max_tail_sec * SR)
            if audio.size > keep:
                audio = audio[-keep:]          # 発話が伸びてもコストを一定に保つ
        feat = wav_to_features(audio, SR, self.cmvn)
        x = feat[None].astype(np.float32)
        feed = {"x": x, "x_length": np.array([x.shape[1]], np.int32),
                "language": np.array([self.lang_id], np.int32),
                "text_norm": np.array([self.cmvn["with_itn"]], np.int32)}
        logits, hidden = self.sess.run(None, feed)[:2]
        return logits, pool(hidden[0], self.n_special, self.tail_frames)

    def __call__(self, audio: np.ndarray) -> dict:
        t0 = time.time()
        logits, v = self.embed(audio)
        prob = None
        if self.W is not None:
            prob = float(1.0 / (1.0 + np.exp(-(float(v @ self.W) + self.b))))
        return {"probability": prob, "is_complete": (prob is not None and prob >= self.threshold),
                "text": self._decode(logits), "latency_ms": (time.time() - t0) * 1000.0}

    # 旧 pipecat/smart-turn 互換の別名
    predict_endpoint = __call__


def _cli() -> None:
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser(description="音声1本のターン判定+文字起こし")
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--model", default=os.getenv("SENSEVOICE_ENCOUT", "model.int8.encout.onnx"))
    ap.add_argument("--head", default="models/turn_head.npz")
    ap.add_argument("--tokens", default=os.getenv("SENSEVOICE_TOKENS", "tokens.txt"))
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    st = SemanticTurn(a.model, a.head, a.tokens, a.threshold)
    for p in a.wav:
        w, sr = sf.read(p)
        if w.ndim > 1:
            w = w[:, 0]
        r = st(w.astype(np.float32))
        mark = "END " if r["is_complete"] else "WAIT"
        print(f"[{mark}] p={r['probability']:.3f} {r['latency_ms']:.0f}ms  {r['text']}  ({p})")


if __name__ == "__main__":
    _cli()
