"""Play several games, render each, audit each, and build a gallery page.

Usage: python gallery.py 3 21 42 77 [--draft] [--no-audit]
Output: gallery/<seed>/ (game.mp4, index.html, logs) + gallery/index.html.
Serve gallery/ with a Range-capable HTTP server to watch.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(HERE, "gallery")


def audit_flags(seed: int) -> list[str]:
    out = os.path.join(GAL, str(seed))
    a = subprocess.run([sys.executable, os.path.join(HERE, "audit_policy.py"),
                        os.path.join(out, "replay_log.json")],
                       env={**os.environ, "POOL_OUT": out},
                       capture_output=True, text=True)
    return [l for l in a.stdout.splitlines()
            if any(k in l for k in ("MISSED_LANE", "DECLINED", "THIN_TAKE"))]


def run(seed: int, draft: bool, audit: bool) -> dict:
    out = os.path.join(GAL, str(seed))
    os.makedirs(out, exist_ok=True)
    env = {**os.environ, "POOL_OUT": out}
    py = sys.executable
    t = time.time()
    args = [py, os.path.join(HERE, "replay.py"), str(seed)] + (["--draft"] if draft else [])
    play = subprocess.run(args, env=env, capture_output=True, text=True)
    t_play = time.time() - t
    t = time.time()
    subprocess.run([py, os.path.join(HERE, "fast_video.py")], env=env,
                   capture_output=True, text=True)
    subprocess.run([py, os.path.join(HERE, "soundtrack.py"), out], env=env,
                   capture_output=True, text=True)    # table sounds (needs node)
    t_vid = time.time() - t
    flags = None  # audited after every game is watchable
    log = json.load(open(os.path.join(out, "replay_log.json")))
    ok = "every logged shot re-simulates exactly" in play.stdout
    return {"seed": seed, "winner": log["winner"], "shots": log["shots"],
            "play_s": round(t_play), "video_s": round(t_vid),
            "replay_ok": ok, "flags": flags}


def page(rows: list[dict]) -> str:
    cards = "".join(
        f'<a class="card" href="{r["seed"]}/index.html"><b>Game {r["seed"]}</b>'
        f'<span>winner: Player {r["winner"]} · {r["shots"]} shots</span>'
        f'<span class="m">played {r["play_s"]}s · video {r["video_s"]}s · '
        f'audit: {"pending" if r["flags"] is None else "clean" if not r["flags"] else str(len(r["flags"])) + " flag(s)"}</span></a>'
        for r in rows)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>8-Ball Games</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{{--bg:#12161c;--panel:#1c222b;--text:#e6e6e6;--muted:#9aa3ad;--accent:#ffd25a}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,system-ui,sans-serif}}
header{{padding:16px 20px;border-bottom:1px solid #2a313b}} h1{{margin:0;font-size:20px}}
header p{{margin:4px 0 0;color:var(--muted)}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;padding:16px 20px}}
.card{{display:flex;flex-direction:column;gap:4px;padding:14px;background:var(--panel);border-radius:10px;
color:var(--text);text-decoration:none;border-left:4px solid var(--accent)}}
.card:hover{{background:#242c37}} .m{{color:var(--muted);font-size:13px}}</style></head><body>
<header><h1>8-Ball example games</h1><p>Both players use the same policy (strong-player skill,
call your shot, no safeties). Each game is audited for missed pots. Click a game to watch.</p></header>
<main>{cards}</main></body></html>"""


def main():
    seeds = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or [3, 21, 42, 77]
    draft = "--draft" in sys.argv
    audit = "--no-audit" not in sys.argv
    os.makedirs(GAL, exist_ok=True)
    rows_path = os.path.join(GAL, "games.json")
    rows = {r["seed"]: r for r in (json.load(open(rows_path)) if os.path.exists(rows_path) else [])}
    def save():
        ordered = [rows[k] for k in sorted(rows)]
        json.dump(ordered, open(rows_path, "w"), indent=1)
        open(os.path.join(GAL, "index.html"), "w").write(page(ordered))

    for sd in seeds:  # phase 1: play + render (watchable as each lands)
        r = run(sd, draft, audit)
        rows[sd] = r
        print(f"game {sd}: winner {r['winner']} in {r['shots']} shots | play {r['play_s']}s "
              f"video {r['video_s']}s | replay {'ok' if r['replay_ok'] else 'MISMATCH'}",
              flush=True)
        save()
    if audit:  # phase 2: audit
        for sd in seeds:
            rows[sd]["flags"] = audit_flags(sd)
            print(f"audit {sd}: {'clean' if not rows[sd]['flags'] else rows[sd]['flags']}",
                  flush=True)
            save()


if __name__ == "__main__":
    main()
