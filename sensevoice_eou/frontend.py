"""SenseVoice の特徴量frontend(fbank80 → LFR(7,6) → CMVN)を再現する。

sherpa-onnx が内部で行う前処理を外部で再現し、露出encoder付きONNX
(sensevoice_encout.onnx)に直接 x を与えられるようにする。学習の埋め込み抽出でも
実行時のターン判定でも同じこの frontend を使う。

CMVN(neg_mean/inv_stddev)・LFR窓・言語IDは元モデルのONNXメタデータから取得。
normalize_samples=0 なので波形は int16 レンジ([-32768,32767])で fbank に渡す。
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi


def load_cmvn_from_onnx(onnx_path: str):
    import onnxruntime as ort
    meta = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    neg_mean = np.array([float(x) for x in meta["neg_mean"].split(",")], dtype=np.float32)
    inv_stddev = np.array([float(x) for x in meta["inv_stddev"].split(",")], dtype=np.float32)
    lfr_m = int(meta["lfr_window_size"])
    lfr_n = int(meta["lfr_window_shift"])
    langs = {k: int(v) for k, v in meta.items() if k.startswith("lang_")}
    return {"neg_mean": neg_mean, "inv_stddev": inv_stddev, "lfr_m": lfr_m, "lfr_n": lfr_n,
            "with_itn": int(meta["with_itn"]), "without_itn": int(meta["without_itn"]), "langs": langs}


def _apply_lfr(feat: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
    """FunASR と同一の LFR(low frame rate) スタック。"""
    T = feat.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left = np.tile(feat[0], ((lfr_m - 1) // 2, 1))
    feat = np.vstack([left, feat])
    Tp = feat.shape[0]
    out = []
    for i in range(T_lfr):
        if lfr_m <= Tp - i * lfr_n:
            out.append(feat[i * lfr_n:i * lfr_n + lfr_m].reshape(1, -1))
        else:
            frame = feat[i * lfr_n:].reshape(-1)
            pad = lfr_m - (Tp - i * lfr_n)
            frame = np.hstack([frame] + [feat[-1]] * pad)
            out.append(frame.reshape(1, -1))
    return np.vstack(out).astype(np.float32)


def wav_to_features(wav: np.ndarray, sr: int, cmvn: dict) -> np.ndarray:
    """16k mono float波形([-1,1]) → (T_lfr, 560) の SenseVoice 入力特徴。"""
    assert sr == 16000, f"expected 16k, got {sr}"
    samples = torch.tensor(wav, dtype=torch.float32).reshape(1, -1) * 32768.0  # normalize_samples=0
    feat = kaldi.fbank(samples, num_mel_bins=80, frame_length=25.0, frame_shift=10.0,
                       dither=0.0, energy_floor=0.0, sample_frequency=16000,
                       window_type="hamming", snip_edges=True).numpy()
    feat = _apply_lfr(feat, cmvn["lfr_m"], cmvn["lfr_n"])
    feat = (feat + cmvn["neg_mean"]) * cmvn["inv_stddev"]
    return feat.astype(np.float32)
