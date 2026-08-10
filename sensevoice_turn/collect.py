"""学習データ収集サーバ — ブラウザ(スマホ可)で録音してラベル付き音声を貯める。

    python -m sensevoice_turn.collect --out data/recordings
    → http://localhost:8100 を開く

カテゴリ:
  pair      … **最重要**。同じ文を「⤵下降=言い切り」「→平坦=まだ続く」の2通りで録る。
              文字列が同じなのでモデルは音(抑揚)で判断せざるを得なくなる。
  turn      … 普通に言い切ったコマンド
  thinking  … 途中で考え込む(「…なんだっけ」)
  filler    … 末尾がフィラー/接続詞(「あと…」)
  dangling  … 何も足さず宙ぶらりんに止める(「画面の明るさを」)
  fillerbank… 言いよどみ単体(コマンドに継ぎ足して負例を増やす素材)

注意: マイクは HTTPS か localhost でしか使えない(ブラウザの制約)。スマホから
使うなら Tailscale / ngrok などで HTTPS を通すこと。
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

LABELS = ("turn", "thinking", "filler", "dangling", "fillerbank", "pair_fall", "pair_flat")

# ペア用の文。**言い切りと継続の両方が自然に成立する文だけ**を選ぶこと。
# 「〜したい」「〜かな」のような願望表現は、平坦(継続)として自然に演じられず
# 偽のイントネーションを教えてしまうので入れない(実際に外して精度が上がった)。
PAIRS = [
    ("ジャズを流してほしいんだけど", "静かめのやつで"),
    ("音量を上げたいんだけど", "半分くらいまでで"),
    ("明日の天気が知りたいんだけど", "特に夕方以降を"),
    ("タイマーをセットしたいんだけど", "10分くらいで"),
    ("電気を消したいんだけど", "リビングだけ"),
    ("音楽を止めてほしいんだけど", "5分後に"),
    ("予定を確認したいんだけど", "今週の分だけ"),
    ("アラームを止めたいんだけど", "スヌーズは残して"),
    ("部屋を暗くしたいんだけど", "真っ暗じゃなくて"),
    ("音楽を止めて", "電気も消して"),
    ("電気をつけて", "少し暗めにして"),
    ("音量を下げて", "半分くらいに"),
    ("タイマーをセットして", "5分で"),
    ("画面を暗くして", "そのあと音楽流して"),
    ("次の曲に進んで", "そのあと少し音量上げて"),
    ("3分のタイマー", "をセットして"),
    ("次の曲", "にしてほしい"),
    ("リビングの電気", "を消して"),
    ("明日の天気", "を教えて"),
    ("今日の予定", "を確認したい"),
    ("音量", "を30%にして"),
    ("電気を消してほしい", "寝室のほうだけ"),
    ("音楽を流してほしい", "小さめの音で"),
    ("天気を教えて", "明日の朝の"),
]

FILLER_BANK = [
    "えーと", "えっと", "えーっと", "あのー", "そのー", "うーん", "んー", "ええと", "あー",
    "うーんと", "なんか", "こう", "まあ", "なんだっけ", "なんだっけな", "なんて言うんだっけ",
    "何だったかな", "どれだっけ", "なんていうか", "ほら", "ほらあれ", "何て言うの",
    "あれなんだっけ", "名前なんだっけ", "じゃなくて", "あ、ちがう", "ちがうちがう",
    "あ、そうじゃなくて", "いや", "じゃなかった", "あ、間違えた", "あと", "あとは", "それと",
    "でも", "だから", "っていうか", "ついでに", "それから", "ちょっと待って", "どうしようかな",
    "そうだなあ", "えーっとね", "んーとね",
]

DANGLING = [
    "画面の明るさを", "音量を", "タイマーを", "アラームを", "電気を", "明日の天気を",
    "今日の予定を", "リビングの電気を", "次の曲に", "3分のタイマーを", "10分後に",
    "音楽をかけてほしくて", "音楽を止めて、", "電気をつけて、", "音量を下げて、",
]

HTML = r"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>turn data collector</title><style>
:root{color-scheme:light dark}*{box-sizing:border-box}
body{font-family:-apple-system,"Hiragino Kaku Gothic ProN",sans-serif;max-width:720px;margin:0 auto;padding:20px;line-height:1.6}
.cat{border:2px solid #ccc;border-radius:12px;padding:12px 14px;cursor:pointer;margin:8px 0}
.cat.sel{border-color:#2d7;background:rgba(40,220,120,.12)}
.cat small{color:#888;display:block}
button{font-size:17px;padding:12px 18px;border-radius:12px;border:none;cursor:pointer}
#rec{background:#e33;color:#fff;min-width:190px;touch-action:none;user-select:none;-webkit-user-select:none}
#rec.on{background:#900;animation:p 1s infinite}@keyframes p{50%{opacity:.55}}
#box{border:2px dashed #3a9;border-radius:12px;padding:16px;margin:14px 0;display:none}
#ptext{font-size:24px;font-weight:700}
.stats{color:#888;white-space:pre-line;margin-top:16px;font-size:14px}
#status{min-height:22px;color:#2d7;font-weight:600}
</style></head><body>
<h1>🎙 turn data collector</h1>
<div id="cats"></div>
<div id="box"><div style="color:#888;font-size:13px">📖 これを読んでください</div>
<div id="ptext">—</div>
<button id="next" style="background:#eee;color:#222;font-size:15px;padding:8px 14px;margin-top:10px">↻ 別の例</button></div>
<div class="row"><button id="rec">● 押している間だけ録音</button> <span id="status"></span></div>
<audio id="play" controls hidden style="width:100%"></audio>
<div class="stats" id="stats"></div>
<script>
const CATS=[["pair","⑥ ペア録音（同じ文を語尾だけ変えて2回）★最重要","同一文なのでモデルは音で判断するしかなくなる"],
["turn","① 言い切り","普通にコマンドを言い切る"],
["thinking","② 途中で考え込み","「…なんだっけ」など"],
["filler","③ 末尾フィラー","「あと…」「でも…」"],
["dangling","⑤ 宙ぶらりん止め","何も足さずそのまま止める"],
["fillerbank","④ フィラー銀行","言いよどみ単体"]];
let label=null,P={},pIdx=0,pairPhase=0;
const catsEl=document.getElementById('cats'),box=document.getElementById('box'),ptext=document.getElementById('ptext');
const recBtn=document.getElementById('rec'),statusEl=document.getElementById('status'),play=document.getElementById('play');
CATS.forEach(([k,t,d])=>{const e=document.createElement('div');e.className='cat';e.innerHTML=`<b>${t}</b><small>${d}</small>`;
 e.onclick=()=>{document.querySelectorAll('.cat').forEach(c=>c.classList.remove('sel'));e.classList.add('sel');
 label=k;pIdx=Math.floor(Math.random()*999);pairPhase=0;show();prepare();};catsEl.appendChild(e);});
function show(){const L=label==='pair'?P.pairs:P[label]||[];if(!L||!L.length){box.style.display='none';return;}
 box.style.display='block';const it=L[pIdx%L.length];
 if(label==='pair'){const fall=pairPhase===0;
  ptext.innerHTML=`「${it.text}${fall?'⤵':'→'}」<div style="font-size:16px;margin-top:8px;color:${fall?'#3a9':'#e66'}">`+
  (fall?'① <b>語尾を下げて言い切る</b> → 確定してほしい':'② <b>語尾を下げず平坦に</b>（続ける気持ちで）→ 待ってほしい')+'</div>'+
  (!fall&&it.cont?`<div style="color:#e66;font-size:14px;margin-top:6px">心の中で続ける:「…${it.cont}」<br><span style="color:#888">続きは言わずに止める</span></div>`:'')+
  `<div style="color:#888;font-size:13px;margin-top:6px">同じ文を2回。今は ${fall?'1回目(下降)':'2回目(平坦)'}</div>`;
 } else ptext.textContent=typeof it==='string'?it:it.text;}
document.getElementById('next').onclick=()=>{pIdx++;pairPhase=0;show();};
function pickMime(){for(const c of['audio/webm;codecs=opus','audio/webm','audio/mp4','audio/aac'])
 {try{if(window.MediaRecorder&&MediaRecorder.isTypeSupported(c))return c}catch(e){}}return''}
let stream=null,mr=null,chunks=[],blob=null,hold=false,ready=false,early=false,t0=0;
async function prepare(){if(stream)return;try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:false}});
 if(!statusEl.textContent)statusEl.textContent='🎤 準備OK'}catch(e){statusEl.textContent='マイク不可: '+e.message}}
function release(){if(stream){try{stream.getTracks().forEach(t=>t.stop())}catch(e){}stream=null}setTimeout(prepare,150)}
async function start(){await prepare();if(!stream)return false;chunks=[];blob=null;play.hidden=true;
 const m=pickMime();mr=m?new MediaRecorder(stream,{mimeType:m}):new MediaRecorder(stream);
 mr.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
 mr.onstart=()=>{statusEl.textContent='🔴 どうぞ'};
 mr.onstop=()=>{release();const held=Date.now()-t0;blob=new Blob(chunks,{type:(mr&&mr.mimeType)||'audio/webm'});
  recBtn.classList.remove('on');
  if(!blob.size||held<400){blob=null;ready=false;recBtn.textContent='● 押している間だけ録音';
   statusEl.textContent=`⚠️ 録音できず(${held}ms)。長めに押してください`;return}
  play.src=URL.createObjectURL(blob);play.hidden=false;ready=true;
  recBtn.textContent='💾 タップで保存';statusEl.textContent='確認して、もう一度タップで保存'};
 t0=Date.now();mr.start();recBtn.textContent='● 録音中…（離すと停止）';recBtn.classList.add('on');return true}
recBtn.addEventListener('pointerdown',async e=>{e.preventDefault();if(ready)return;early=false;hold=await start();
 if(hold&&early)setTimeout(()=>{hold=false;if(mr&&mr.state==='recording')mr.stop()},500)});
const rel=e=>{if(!hold){early=true;return}e.preventDefault();hold=false;if(mr&&mr.state==='recording')mr.stop()};
['pointerup','pointercancel','pointerleave'].forEach(t=>recBtn.addEventListener(t,rel));
recBtn.addEventListener('contextmenu',e=>e.preventDefault());
recBtn.addEventListener('click',async e=>{e.preventDefault();if(!ready)return;ready=false;await save();
 recBtn.textContent='● 押している間だけ録音'});
async function save(){if(!blob||!label)return;statusEl.textContent='保存中…';
 const L=label==='pair'?(pairPhase===0?'pair_fall':'pair_flat'):label;
 const list=label==='pair'?P.pairs:P[label]||[];const it=list[pIdx%list.length];
 const txt=typeof it==='string'?it:(it?it.text:'');
 const fd=new FormData();fd.append('audio',blob,'r.webm');fd.append('label',L);fd.append('text',txt||'');
 try{const d=await(await fetch('/save',{method:'POST',body:fd})).json();
  if(d.ok){statusEl.textContent=`✓ 保存 (${d.sec}秒) ${d.total}件`;blob=null;play.hidden=true;
   if(label==='pair'){if(pairPhase===0)pairPhase=1;else{pairPhase=0;pIdx++}}else pIdx++;show();stats()}
  else statusEl.textContent='✗ '+(d.message||d.error)}catch(e){statusEl.textContent='失敗: '+e.message}}
async function stats(){try{const d=await(await fetch('/stats')).json();
 document.getElementById('stats').textContent='収集済み:\n'+Object.entries(d.counts).map(([k,v])=>`  ${k}: ${v}`).join('\n')+`\n  合計: ${d.total}`}catch(e){}}
fetch('/prompts').then(r=>r.json()).then(d=>{P=d;stats()});
</script></body></html>"""


def build_app(outdir: Path):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse

    app = FastAPI()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.jsonl"

    @app.get("/")
    async def index():
        return HTMLResponse(HTML)

    @app.get("/prompts")
    async def prompts():
        pairs = [{"text": t, "cont": c} for t, c in PAIRS]
        random.shuffle(pairs)
        return {
            "pairs": pairs,
            "turn": ["音量を30%にして", "3分のタイマーをセットして", "ジャズを再生して",
                     "今何時", "明日の天気を教えて", "電気を消して"],
            "thinking": [f"「{t}…」まで言って『{f}』で止める"
                         for (t, _c), f in zip(PAIRS, random.sample(FILLER_BANK, len(PAIRS)))],
            "filler": [f"最後に『{w}…』で終わって続きそうな感じで止める" for w in FILLER_BANK[:20]],
            "dangling": [f"「{d}」まで言って何も足さず止める" for d in DANGLING],
            "fillerbank": FILLER_BANK,
        }

    @app.get("/stats")
    async def stats():
        counts = {l: len(list((outdir / l).glob("*.wav"))) for l in LABELS
                  if (outdir / l).exists()}
        return {"counts": counts, "total": sum(counts.values())}

    @app.post("/save")
    async def save(audio: UploadFile = File(...), label: str = Form(...), text: str = Form("")):
        if label not in LABELS:
            return {"ok": False, "error": "bad_label"}
        data = await audio.read()
        if len(data) < 512:                     # 空データを「保存済み」と数えない
            return {"ok": False, "error": "empty", "message": f"録音が空です({len(data)}B)"}
        d = outdir / label
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        raw, wav = d / f"{ts}.src", d / f"{ts}.wav"
        raw.write_bytes(data)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw),
                        "-ac", "1", "-ar", "16000", str(wav)], check=False)
        raw.unlink(missing_ok=True)
        if not wav.exists() or wav.stat().st_size < 12800:      # 0.4秒未満は事故
            wav.unlink(missing_ok=True)
            return {"ok": False, "error": "too_short",
                    "message": "録音が短すぎます。ボタンを長めに押してください"}
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"file": f"{label}/{wav.name}", "label": label, "text": text,
                                "timestamp": datetime.now().isoformat(timespec="seconds")},
                               ensure_ascii=False) + "\n")
        total = sum(len(list((outdir / l).glob("*.wav"))) for l in LABELS if (outdir / l).exists())
        return {"ok": True, "sec": round(wav.stat().st_size / 32000, 1), "total": total}

    return app


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="ターン学習データ収集サーバ")
    ap.add_argument("--out", default="data/recordings")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    print(f"録音は {a.out} に保存されます → http://localhost:{a.port}")
    print("※ スマホから使う場合はHTTPS必須(Tailscale/ngrok等)")
    uvicorn.run(build_app(Path(a.out)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
