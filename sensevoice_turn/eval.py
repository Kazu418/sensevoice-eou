"""エンドツーエンド評価 — 録音を実行時と同じループに流して測る。

断片ごとの分類精度は**実力を過大評価します**。実行時は1発話がポーズを複数回通過し、
**最初に「確定」と判定した時点で打ち切られる**ため、後のポーズで正解できても手遅れです。
つまり 1発話につき 1回でも誤爆したら失敗。ここではその連鎖込みで測ります。

    python -m sensevoice_turn.eval --rec data/recordings --data data/dataset \
        --model model.int8.encout.onnx --head models/turn_head.npz

指標:
  早切れ(cut-off)   … まだ発話が残っているのに確定した(致命的。言い直しになる)
  誤確定(false commit) … 「待つべき」録音を最後までに確定した
  取りこぼし(missed) … 言い切ったのに最後まで確定しなかった(安全弁待ち=もたつく)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16000
FRAME = int(SR * 0.032)
PAUSE_MS = 400.0
MIN_SPEECH_MS = 500.0
SPEECH_AFTER_MS = 200.0


def replay(wav: np.ndarray, st, vad, threshold: float) -> dict:
    """実行時と同じ手順で流し、最初に確定した地点を返す。"""
    vad.reset()
    pending = False
    events = []
    for i in range(0, (len(wav) // FRAME) * FRAME, FRAME):
        r = vad.process(wav[i:i + FRAME])
        if r["silence_ms"] == 0:
            pending = False
        if not pending and r["speech_ms"] >= MIN_SPEECH_MS and vad.pause_after_speech(PAUSE_MS):
            pending = True
            pos = i + FRAME
            p = st(wav[:pos])["probability"]
            events.append((pos, p))
            if p >= threshold:
                # 確定した。この後にまだ発話が残っていたら「早切れ」
                vad.reset()
                tail, remain = wav[pos:], 0.0
                for j in range(0, (len(tail) // FRAME) * FRAME, FRAME):
                    remain = max(remain, vad.process(tail[j:j + FRAME])["speech_ms"])
                return {"committed": True, "at": pos, "prob": p,
                        "cut_off": remain >= SPEECH_AFTER_MS, "remain_ms": remain,
                        "events": events}
    # 音声が尽きた = 実機ではこの後の無音で判定される
    p = st(wav)["probability"]
    events.append((len(wav), p))
    return {"committed": p >= threshold, "at": len(wav), "prob": p,
            "cut_off": False, "remain_ms": 0.0, "events": events}


def main() -> None:
    ap = argparse.ArgumentParser(description="実行時ループ込みのエンドツーエンド評価")
    ap.add_argument("--rec", default="data/recordings")
    ap.add_argument("--data", default="data/dataset", help="val分割を読むため(あれば)")
    ap.add_argument("--model", default="model.int8.encout.onnx")
    ap.add_argument("--head", default="models/turn_head.npz")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--val-only", action="store_true", help="検証分割の録音だけを対象にする")
    a = ap.parse_args()

    import soundfile as sf
    from .infer import SemanticTurn
    from .vad import StreamingVAD
    from .build import TERMINAL, load_wav

    st = SemanticTurn(a.model, a.head)
    th = a.threshold if a.threshold is not None else st.threshold
    vad = StreamingVAD()

    val_origins = None
    dm = Path(a.data) / "manifest.jsonl"
    if a.val_only and dm.exists():
        val_origins = {json.loads(l).get("origin") for l in dm.read_text(encoding="utf-8").splitlines()
                       if l.strip() and json.loads(l).get("split") == "val"}

    REC = Path(a.rec)
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
                if val_origins is None or p.name in val_origins:
                    recs.append((p, r["label"], r.get("text", "")))
    if not recs:
        raise SystemExit("対象の録音がありません")

    agg = {}
    fails = []
    for p, lab, text in recs:
        w = load_wav(p)
        if w is None or len(w) < SR * 0.3:
            continue
        res = replay(w.astype(np.float32), st, vad, th)
        want_commit = TERMINAL[lab] == "turn"
        k = agg.setdefault(lab, {"n": 0, "cut_off": 0, "false_commit": 0, "missed": 0, "ok": 0})
        k["n"] += 1
        if res["cut_off"]:
            k["cut_off"] += 1
            fails.append(("早切れ", lab, text, res))
        elif want_commit and not res["committed"]:
            k["missed"] += 1
            fails.append(("取りこぼし", lab, text, res))
        elif (not want_commit) and res["committed"]:
            k["false_commit"] += 1
            fails.append(("誤確定", lab, text, res))
        else:
            k["ok"] += 1

    print(f"\n=== エンドツーエンド評価 (閾値 {th:.2f}, {'val分割のみ' if val_origins else '全録音'}) ===")
    print(f"{'カテゴリ':16} {'件数':>5} {'成功':>5} {'早切れ':>6} {'誤確定':>6} {'取りこぼし':>8}")
    tot = {"n": 0, "ok": 0, "cut_off": 0, "false_commit": 0, "missed": 0}
    for lab in sorted(agg):
        k = agg[lab]
        for m in tot:
            tot[m] += k[m]
        print(f"{lab:16} {k['n']:5} {k['ok']:5} {k['cut_off']:6} {k['false_commit']:6} {k['missed']:8}")
    print(f"{'合計':16} {tot['n']:5} {tot['ok']:5} {tot['cut_off']:6} {tot['false_commit']:6} {tot['missed']:8}")
    if tot["n"]:
        print(f"\n発話単位の成功率: {tot['ok']/tot['n']*100:.1f}%")
        print(f"  早切れ(致命的): {tot['cut_off']/tot['n']*100:.1f}%   "
              f"誤確定: {tot['false_commit']/tot['n']*100:.1f}%   "
              f"取りこぼし: {tot['missed']/tot['n']*100:.1f}%")
    if fails:
        print("\n--- 失敗例(最大10件) ---")
        for kind, lab, text, res in fails[:10]:
            ps = ", ".join(f"{p:.2f}" for _pos, p in res["events"])
            print(f"  [{kind}] {lab} 「{text[:24]}」 p=[{ps}]")


if __name__ == "__main__":
    main()
