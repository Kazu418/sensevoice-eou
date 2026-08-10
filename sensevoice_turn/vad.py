#!/usr/bin/env python3
"""
ストリーミング VAD（発話区間検出）ラッパ — silero-vad (ONNX)

ウェイクワード検出後の発話キャプチャ中に、16kHz PCM をチャンク単位で流し込み、
「発話が始まったか」「発話後にポーズ（無音）が来たか」を追跡する。
このポーズ地点で Smart Turn v3 にターン終了判定をかける（固定2秒待ちはしない）。
"""
from __future__ import annotations

import os

import numpy as np
import torch
from silero_vad import load_silero_vad

_SR = 16000
_WINDOW = 512  # silero-vad の 16kHz 窓サイズ（32ms）


class StreamingVAD:
    def __init__(self, speech_threshold: float | None = None):
        self.model = load_silero_vad(onnx=True)
        self.speech_threshold = float(
            speech_threshold if speech_threshold is not None
            else os.getenv("VAD_SPEECH_THRESHOLD", "0.5")
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self.has_speech = False
        self.silence_samples = 0
        self.speech_samples = 0

    def process(self, pcm: np.ndarray) -> dict:
        """16kHz float32 PCM を流し込み、最新状態を返す。"""
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.float32).reshape(-1)])
        last_prob = 0.0
        while len(self._buf) >= _WINDOW:
            win = self._buf[:_WINDOW]
            self._buf = self._buf[_WINDOW:]
            last_prob = float(self.model(torch.from_numpy(win.copy()), _SR).item())
            if last_prob >= self.speech_threshold:
                self.has_speech = True
                self.speech_samples += _WINDOW
                self.silence_samples = 0
            else:
                self.silence_samples += _WINDOW
        return {
            "prob": last_prob,
            "has_speech": self.has_speech,
            "silence_ms": self.silence_samples / _SR * 1000.0,
            "speech_ms": self.speech_samples / _SR * 1000.0,
        }

    def pause_after_speech(self, min_silence_ms: float = 300.0) -> bool:
        """発話が一度あり、その後 min_silence_ms 以上の無音が続いていれば True。"""
        return self.has_speech and (self.silence_samples / _SR * 1000.0) >= min_silence_ms


if __name__ == "__main__":
    import sys, wave, glob

    def load(p):
        with wave.open(p, "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    rec = sys.argv[1] if len(sys.argv) > 1 else "/home/pi/TalkGPT-fast/backend/wakeword/recordings"
    vad = StreamingVAD()
    # 無音
    vad.reset(); r = vad.process(np.zeros(_SR * 2, dtype=np.float32))
    print(f"silence 2s: has_speech={r['has_speech']} silence_ms={r['silence_ms']:.0f}")
    # 実録音（末尾に発話あり）を 80ms チャンクで流す
    for f in sorted(glob.glob(f"{rec}/true_positive/*.wav"))[:3]:
        if os.path.getsize(f) < 100000:
            continue
        a = load(f); vad.reset()
        step = 1280
        for i in range(0, len(a), step):
            r = vad.process(a[i:i + step])
        print(f"{os.path.basename(f)[:19]}: has_speech={r['has_speech']} "
              f"speech_ms={r['speech_ms']:.0f} trailing_silence_ms={r['silence_ms']:.0f}")
