"""SenseVoice の ONNX に encoder 隠れ状態の出力を生やす。

SenseVoice-Small の配布 ONNX は encoder と CTC 射影が 1 グラフに融合していて、
出力は `logits` (N, T, vocab) だけ。ターン判定のような下流タスクに使いたい
**512次元のフレーム埋め込み**は内部テンソルのまま外に出てこない。

ここでは再 export せず、既存の ONNX に出力を 1 本足すだけの「手術」を行う。
CTC 射影(MatMul)の入力テンソル `/encoder/tp_norm/Add_1_output_0` をグラフ出力に
追加すると、1 回の推論で

    logits      … 文字起こし(従来どおり)
    encoder_out … (N, T, 512) フレーム埋め込み(ターン判定などに使う)

の両方が取れる。fp32 版と int8 版のどちらにも同じ名前のテンソルが存在する
(int8 は量子化の手前にある)ので、同じ手順で処理できる。

使い方:
    python -m sensevoice_eou.expose model.int8.onnx model.int8.encout.onnx
"""
from __future__ import annotations

import argparse
import sys

# CTC 射影の直前 = encoder 最終 LayerNorm の出力。fp32/int8 とも同名。
HIDDEN_TENSOR = "/encoder/tp_norm/Add_1_output_0"


def find_hidden_tensor(graph, logits_name: str = "logits") -> str | None:
    """logits から辿って CTC 射影の入力(=隠れ状態)を特定する。

    既定の名前が見つからないモデル(将来の版など)でも動くよう、グラフを遡って探す。
    """
    producer = {o: n for n in graph.node for o in n.output}
    seen = set()
    cur = logits_name
    for _ in range(12):
        node = producer.get(cur)
        if node is None:
            break
        # MatMul / MatMulInteger の第1入力がフレーム埋め込み
        if node.op_type in ("MatMul", "MatMulInteger"):
            cand = node.input[0]
            # int8 は DynamicQuantizeLinear を挟むのでさらに 1 段戻る
            q = producer.get(cand)
            if q is not None and q.op_type == "DynamicQuantizeLinear":
                return q.input[0]
            return cand
        nxt = [i for i in node.input if i in producer and i not in seen]
        if not nxt:
            break
        seen.add(cur)
        cur = nxt[0]
    return None


def expose(src: str, dst: str, tensor: str | None = None) -> str:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(src)
    graph = model.graph
    existing = {o.name for o in graph.output}

    name = tensor or HIDDEN_TENSOR
    all_outputs = {o for n in graph.node for o in n.output}
    if name not in all_outputs:
        found = find_hidden_tensor(graph)
        if not found:
            raise SystemExit(f"隠れ状態テンソルが見つかりません(指定: {name})")
        print(f"既定名が無いため自動検出しました: {found}")
        name = found

    if name in existing:
        print(f"既に出力済み: {name}")
    else:
        graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    onnx.save(model, dst)
    print(f"保存: {dst}")
    print(f"outputs: {[o.name for o in graph.output]}")
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="SenseVoice ONNX に encoder 出力を追加する")
    ap.add_argument("src", help="元の model.onnx / model.int8.onnx")
    ap.add_argument("dst", help="出力先(encoder出力付き)")
    ap.add_argument("--tensor", default=None, help="隠れ状態テンソル名(既定は自動)")
    args = ap.parse_args()
    expose(args.src, args.dst, args.tensor)


if __name__ == "__main__":
    sys.exit(main())
