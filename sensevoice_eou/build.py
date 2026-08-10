"""録音から学習データを作る — **実行時と同じ形**（VADポーズで切った断片）にする。

素朴にやると「学習=発話全体 / 実行時=ポーズまでの途中音声」で入力分布がズレる
(train/serve skew)。ここでは録音をVADに流し、**判定が走るポーズ地点ごとに**
音声を切って1サンプルにする。ラベルは自動で決まる:

    そのポーズの後にまだ発話が残っている → noturn (ここで確定してはいけない)
    後に発話が無い(最後のポーズ)        → その録音のラベル

これで「音量を…(ポーズ)…30%にして」の中間ポーズが、人工的な継ぎ接ぎ無しに
現実的な negative になる。

    python -m sensevoice_eou.build --rec data/recordings --out data/dataset
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FRAME = int(SR * 0.032)
PAUSE_MS = 400.0          # 実行時と同じ「判定をかける無音長」
MIN_SPEECH_MS = 500.0     # これ未満の発話では判定しない
SPEECH_AFTER_MS = 200.0   # これ以上後に残っていれば「まだ続く」

# 録音カテゴリ → 最後まで聞いたときの正解ラベル
TERMINAL = {"turn": "turn", "pair_fall": "turn", "pair_flat": "noturn",
            "thinking": "noturn", "filler": "noturn", "dangling": "noturn",
            "fillerbank": "noturn"}
# 文単位で train/val を分けるカテゴリ(同じ文が両側に出るとリークになる)
SPLIT_BY_TEXT = {"pair_fall", "pair_flat", "thinking", "filler", "dangling"}


def load_wav(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() == ".wav":
            w, sr = sf.read(str(path))
            if w.ndim > 1:
                w = w[:, 0]
            if sr == SR:
                return w.astype(np.float32)
        tmp = Path("/tmp") / f"_sv_{path.stem}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                        "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
        w, _ = sf.read(str(tmp))
        tmp.unlink(missing_ok=True)
        return (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
    except Exception:
        return None


def pause_points(wav: np.ndarray, vad) -> list[int]:
    vad.reset()
    pts, pending = [], False
    for i in range(0, (len(wav) // FRAME) * FRAME, FRAME):
        r = vad.process(wav[i:i + FRAME])
        if r["silence_ms"] == 0:
            pending = False
        if not pending and r["speech_ms"] >= MIN_SPEECH_MS and vad.pause_after_speech(PAUSE_MS):
            pending = True
            pts.append(i + FRAME)
    return pts


def speech_after(wav: np.ndarray, pos: int, vad) -> float:
    tail = wav[pos:]
    if tail.size < FRAME:
        return 0.0
    vad.reset()
    best = 0.0
    for j in range(0, (len(tail) // FRAME) * FRAME, FRAME):
        best = max(best, vad.process(tail[j:j + FRAME])["speech_ms"])
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="録音 → VADポーズ断片データセット")
    ap.add_argument("--rec", default="data/recordings", help="collect.py の出力先")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--val-ratio", type=float, default=0.30, help="検証に回す文の割合")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    from .vad import StreamingVAD
    vad = StreamingVAD()

    REC, OUT = Path(a.rec), Path(a.out)
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    rows, counter = [], {}

    recs = []
    man = REC / "manifest.jsonl"
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            p = REC / r["file"]
            if p.exists() and r.get("label") in TERMINAL:
                recs.append((p, r["label"], r.get("text", "")))
    else:   # manifest が無ければディレクトリ名をラベルとして拾う
        for lab in TERMINAL:
            for p in sorted((REC / lab).glob("*.wav")):
                recs.append((p, lab, ""))
    if not recs:
        raise SystemExit(f"{REC} に録音が見つかりません。先に collect を実行してください")

    texts = {}
    for _p, lab, t in recs:
        if lab in SPLIT_BY_TEXT and t:
            texts.setdefault("pair" if lab.startswith("pair") else lab, set()).add(t)
    val_texts = set()
    for _k, ts in texts.items():
        ts = sorted(ts)
        random.shuffle(ts)
        val_texts |= set(ts[:max(1, int(len(ts) * a.val_ratio))])

    def emit(w, label, split, source, meta):
        d = OUT / split / label
        d.mkdir(parents=True, exist_ok=True)
        counter[label] = counter.get(label, 0) + 1
        name = f"{source}_{counter[label]:05d}.wav"
        sf.write(str(d / name), w, SR)
        rows.append({"path": f"{split}/{label}/{name}", "label": label,
                     "split": split, "source": source, **meta})

    for p, lab, text in recs:
        w = load_wav(p)
        if w is None or len(w) < SR * 0.3:
            continue
        split = "val" if (lab in SPLIT_BY_TEXT and text in val_texts) else "train"
        terminal = TERMINAL[lab]
        pts = pause_points(w, vad)
        if not pts:
            emit(w, terminal, split, lab, {"text": text, "origin": p.name})
            continue
        for pt in pts:
            remain = speech_after(w, pt, vad)
            label = "noturn" if remain >= SPEECH_AFTER_MS else terminal
            emit(w[:pt], label, split, lab, {"text": text, "origin": p.name,
                                             "cut_ms": round(pt / SR * 1000)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    agg = {}
    for r in rows:
        agg[(r["split"], r["label"])] = agg.get((r["split"], r["label"]), 0) + 1
    print(f"録音 {len(recs)}件 → サンプル {len(rows)}件")
    for k in sorted(agg):
        print(f"  {k[0]:5} {k[1]:7}: {agg[k]}")
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
