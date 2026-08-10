"""ターンヘッドを学習する（ローカルCPUで完結。GPU不要）。

encoder は凍結したまま埋め込みを取り出し、その上のロジスティック回帰だけを学習する。
学習対象は 2048→1 の線形1層なので、数百サンプルなら CPU で数分で終わる
(Raspberry Pi でも動く)。GPU も Modal も要らない。

    python -m sensevoice_eou.train --data data/dataset --model model.int8.encout.onnx

出力:
  models/turn_head.npz     … 学習済みヘッド(推論はこれだけあればよい, 約30KB)
  埋め込みキャッシュ         … 2回目以降の学習が速くなる
併せて閾値スイープ(早切れ/待ちすぎのトレードオフ表)も表示する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def extract(data: Path, model: str, tokens: str | None, cache: Path, force: bool):
    """全サンプルの埋め込みを取り出す(キャッシュあり)。"""
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=True)
        print(f"埋め込みキャッシュを使用: {cache}")
        return d["Xtr"], d["ytr"], d["Xva"], d["yva"], json.loads(str(d["meta_va"]))

    import soundfile as sf
    from .infer import SemanticTurn
    st = SemanticTurn(model, head=None, tokens=tokens)
    rows = [json.loads(l) for l in (data / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    Xtr, ytr, Xva, yva, meta_va = [], [], [], [], []
    for i, r in enumerate(rows):
        try:
            w, _sr = sf.read(str(data / r["path"]))
            if w.ndim > 1:
                w = w[:, 0]
            _logits, v = st.embed(w.astype(np.float32))
        except Exception as e:
            print("skip", r["path"], e)
            continue
        y = 1 if r["label"] == "turn" else 0
        if r["split"] == "train":
            Xtr.append(v); ytr.append(y)
        else:
            Xva.append(v); yva.append(y)
            meta_va.append({"source": r.get("source", ""), "text": r.get("text", "")})
        if i % 50 == 0:
            print(f"  埋め込み {i}/{len(rows)}")
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xva, yva = np.array(Xva), np.array(yva)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
             meta_va=np.array(json.dumps(meta_va, ensure_ascii=False)))
    return Xtr, ytr, Xva, yva, meta_va


def main() -> None:
    ap = argparse.ArgumentParser(description="ターンヘッドの学習(CPUのみ)")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--model", default="model.int8.encout.onnx")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--out", default="models/turn_head.npz")
    ap.add_argument("--cache", default="data/embeddings.npz")
    ap.add_argument("--force", action="store_true", help="埋め込みを取り直す")
    ap.add_argument("--threshold", type=float, default=None, help="既定はスイープの推奨値")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr, Xva, yva, meta_va = extract(Path(a.data), a.model, a.tokens, Path(a.cache), a.force)
    print(f"train={Xtr.shape} val={Xva.shape}")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xva))[:, 1]

    print("\n閾値 |  早切れ(切りすぎ) | 待ちすぎ")
    print("-" * 40)
    best, table = None, []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
        pred = (p >= th).astype(int)
        early = float((pred[yva == 0] == 1).mean()) if (yva == 0).any() else 0.0
        wait = float((pred[yva == 1] == 0).mean()) if (yva == 1).any() else 0.0
        table.append((th, early, wait))
        print(f"{th:.2f} |      {early*100:5.1f}%       |  {wait*100:5.1f}%")
    # 待ちすぎを増やさずに早切れが最小になる点(=カーブの膝)を推奨
    base_wait = min(w for _t, _e, w in table)
    cands = [(t, e, w) for t, e, w in table if w <= base_wait + 0.005]
    best = min(cands, key=lambda x: x[1])[0] if cands else 0.7
    th = a.threshold if a.threshold is not None else best
    print(f"\n推奨閾値: {best:.2f}  (採用: {th:.2f})")

    # ペア指標: 同一テキストで語尾だけ違う組を音で区別できたか
    pair = {}
    for i, m in enumerate(meta_va):
        if m.get("source", "").startswith("pair") and m.get("text"):
            pair.setdefault(m["text"], {})[m["source"]] = p[i]
    both = [(t, v) for t, v in pair.items() if len(v) == 2]
    if both:
        ok = sum(1 for _t, v in both if v.get("pair_fall", 0) >= th and v.get("pair_flat", 1) < th)
        print(f"ペア判別(同一文・語尾違い): {ok}/{len(both)} 組")

    W = (clf.coef_[0] / sc.scale_).astype(np.float32)
    b = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, W=W, b=np.float32(b), n_special=np.int32(4),
             tail_frames=np.int32(8), threshold=np.float32(th))
    print(f"保存: {a.out}  (推論はこのファイルだけでよい)")


if __name__ == "__main__":
    main()
