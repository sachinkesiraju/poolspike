"""replay_out/index.html: the game video + a clickable shot list (jump to a
shot, live highlight of the current one). Rebuilt by fast_video.py; serve
replay_out/ over HTTP with Range support so seeking works."""

import json
import os
import subprocess

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>8-Ball Example Run</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#12161c;--panel:#1c222b;--text:#e6e6e6;--muted:#9aa3ad;--accent:#ffd25a}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{padding:16px 20px;border-bottom:1px solid #2a313b}
h1{margin:0;font-size:20px} header p{margin:4px 0 0;color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:16px;padding:16px 20px}
video{width:100%;border-radius:8px;background:#000}
ol{list-style:none;margin:0;padding:0;max-height:calc(100vh - 120px);overflow:auto}
li{padding:10px 12px;margin-bottom:6px;background:var(--panel);border-radius:8px;cursor:pointer;border-left:4px solid transparent}
li:hover{background:#242c37} li.on{border-left-color:var(--accent);background:#262f3b}
.meta{color:var(--muted);font-size:13px;margin-top:3px}
@media (max-width:900px){main{grid-template-columns:1fr} ol{max-height:none}}
</style></head><body>
<header><h1>8-Ball example run</h1><p id="sub"></p>__BACK__</header>
<main><section><video id="v" src="game.mp4" controls autoplay muted></video></section>
<ol id="list"></ol></main>
<script>
const D=__DATA__;
document.getElementById('sub').textContent=`Seed ${D.seed} · ${D.shots.length} shots · winner: Player ${D.winner}. Click a shot to jump.`;
const v=document.getElementById('v'),L=document.getElementById('list');
D.shots.forEach(s=>{const li=document.createElement('li');
 li.innerHTML=`<b>Shot ${String(s.n).padStart(2,'0')} · Player ${s.p}</b><div class="meta"></div>`;
 li.querySelector('.meta').textContent=s.txt; li.onclick=()=>{v.currentTime=s.t+0.01;v.play();}; L.appendChild(li);});
v.addEventListener('timeupdate',()=>{let k=0;D.shots.forEach((s,i)=>{if(v.currentTime>=s.t)k=i;});
 [...L.children].forEach((li,i)=>li.classList.toggle('on',i===k));});
</script></body></html>"""


def build(out_dir: str, back: bool = False) -> str:
    items = json.load(open(os.path.join(out_dir, "video_items.json")))
    log = json.load(open(os.path.join(out_dir, "replay_log.json")))
    shots, t = [], 0.0
    for it in items:
        seg = os.path.join(out_dir, "segments", f"s{it['shot_no']:03d}.mp4")
        shots.append({"n": it["shot_no"], "p": it["player"], "t": round(t, 2),
                      "txt": it["narration"]})
        t += float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", seg]).decode())
    data = {"winner": log["winner"], "seed": log["seed"], "shots": shots}
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as f:
        page = PAGE.replace("__DATA__", json.dumps(data)).replace(
            "__BACK__", '<p><a href="../index.html" style="color:#ffd25a">'
            '← all games</a></p>' if back else "")
        f.write(page)
    return path


if __name__ == "__main__":
    print(build(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "replay_out")))
