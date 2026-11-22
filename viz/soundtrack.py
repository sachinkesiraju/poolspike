"""Soundtrack for a rendered game video, using the replay viewer's own table sounds.

  python soundtrack.py OUT_DIR [--speed=1.5] [--fps=30]

The sounds are client/sound.js (ball clack, cue strike, cushion thump, pocket drop, break
crack, room reverb), run unchanged in node. The events come straight from pooltool's
event list for each shot (re-simulated with the rendered stroke), with the kinds, gains
and pans that sound.js's eventsFor() assigns. Signal chain as in TableAudio: bus 0.9 into
dry + room convolver, then the same gentle compressor. Video time of a sim event =
segment start + 1.4 s aim hold + t / speed (fast_video's timing).
"""
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))  # game engine, policy
import policy as P  # noqa: E402
import pooltool as pt  # noqa: E402

SR = 44100
HOLD = 1.4
TABLE_L = 1.9812
NODE = "/opt/homebrew/bin/node"
SOUND_JS = os.path.join(HERE, "client", "sound.js")
RENDER = os.path.join(HERE, "render_sound.mjs")
ET = pt.EventType


def hit_gain(dv):
    return max(0.15, min(1.0, (dv / 5) ** 0.6))


def cue_gain(v, brk):
    g = min(1.0, 0.38 + v / 14)
    return min(1.0, g * 1.2) if brk else g


def vel(o):
    return np.asarray(o.state.rvw[1][:2], float) if o is not None and hasattr(o, "state") else np.zeros(2)


def pos(o):
    return np.asarray(o.state.rvw[0][:2], float) if o is not None and hasattr(o, "state") else np.array([0.5, TABLE_L / 2])


def pan(y):
    return float(np.clip((y / TABLE_L - 0.5) * 0.9, -0.6, 0.6))


def shot_events(it, shot):
    """Shot-local events, as sound.js's eventsFor would label them."""
    call = it.get("call") or {}
    brk = call.get("kind") == "break"
    v = float(it["stroke"][1])
    c0 = it["state"]["balls"]["cue"]
    ev = [dict(t=0.0, kind="cue", dv=v, gain=cue_gain(v, brk), brk=brk, pan=pan(c0["y"]))]
    bb = [e for e in shot.events if e.event_type == ET.BALL_BALL]
    rack = set()
    if brk and bb:
        t1 = bb[0].time
        grp = [e for e in bb if e.time - t1 <= 0.05]
        balls = {a.id for e in grp for a in e.agents}
        if len(balls) >= 4:
            dv = max(np.linalg.norm(vel(a.final)) for e in grp for a in e.agents)
            ev.append(dict(t=t1, kind="rack", dv=float(dv), gain=min(1.0, hit_gain(dv) * 1.05), n=len(balls),
                           pan=pan(pos(grp[0].agents[0].initial)[1])))
            rack = {id(e) for e in grp}
    called = call.get("ball")
    win = it.get("result") == "win"
    for e in shot.events:
        et, ag = e.event_type, e.agents
        if et == ET.BALL_BALL and id(e) not in rack:
            dv = max(np.linalg.norm(vel(a.final) - vel(a.initial)) for a in ag)
            if dv >= 0.2:
                ev.append(dict(t=e.time, kind="ball", dv=float(dv), gain=hit_gain(dv), pan=pan(pos(ag[0].initial)[1])))
        elif et in (ET.BALL_LINEAR_CUSHION, ET.BALL_CIRCULAR_CUSHION):
            dv = float(np.linalg.norm(vel(ag[0].final) - vel(ag[0].initial)))
            if dv > 0.75:                       # sound.js only hears rails above this
                ev.append(dict(t=e.time, kind="rail", dv=dv, gain=hit_gain(dv) * 0.62, pan=pan(pos(ag[0].initial)[1])))
        elif et == ET.BALL_POCKET:
            bid = ag[0].id
            if bid == "cue":
                ev.append(dict(t=e.time, kind="scratch", dv=1, gain=0.7, pan=pan(pos(ag[0].initial)[1])))
            else:
                g = 1.0 if (win and bid == "8") else 0.9 if bid == called else 0.72
                ev.append(dict(t=e.time, kind="pocket", dv=1, gain=g, pan=pan(pos(ag[0].initial)[1])))
    ev.sort(key=lambda e: (e["t"], e["kind"]))
    out = []
    for e in ev:                                # sound.js: merge same-kind hits within 40 ms
        p = out[-1] if out else None
        if p and p["kind"] == e["kind"] and e["kind"] in ("ball", "rail") and e["t"] - p["t"] < 0.04:
            if e["gain"] > p["gain"]:
                out[-1] = e
            continue
        out.append(e)
    return out


def compress(x, thr_db=-6, knee=6, ratio=1.8, att=0.012, rel=0.08):
    """WebAudio DynamicsCompressor settings from TableAudio, on 1 ms blocks."""
    blk = SR // 1000
    nb = len(x) // blk + 1
    pad = np.zeros((nb * blk, 2))
    pad[:len(x)] = x
    lvl = np.abs(pad).reshape(nb, blk, 2).max(axis=(1, 2))
    db = 20 * np.log10(np.maximum(lvl, 1e-9))
    over = db - thr_db
    red = np.where(over <= -knee / 2, 0.0,
                   np.where(over >= knee / 2, over * (1 - 1 / ratio), (over + knee / 2) ** 2 / (2 * knee) * (1 - 1 / ratio)))
    a, r = np.exp(-0.001 / att), np.exp(-0.001 / rel)
    g = np.zeros(nb)
    cur = 0.0
    for i in range(nb):
        c = a if red[i] > cur else r
        cur = c * cur + (1 - c) * red[i]
        g[i] = cur
    gain = 10 ** (-np.repeat(g, blk)[:len(x)] / 20)
    return x * gain[:, None]


def nframes(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def main(out, spd=1.5, fps=30):
    items = json.load(open(os.path.join(out, "video_items.json")))
    segs = [os.path.join(out, "segments", f"s{it['shot_no']:03d}.mp4") for it in items]
    durs = [nframes(s) / fps for s in segs]
    total = sum(durs)
    events, t0 = [], 0.0
    for it, d in zip(items, durs):
        shot = P.simulate(it["state"], tuple(it["stroke"]))
        if shot is not None:
            for e in shot_events(it, shot):
                at = t0 + HOLD + e["t"] / spd
                if at <= t0 + d:
                    events.append({**e, "at": at, "break": e.pop("brk", False)})
        t0 += d
    with tempfile.TemporaryDirectory() as tmp:
        ej = os.path.join(tmp, "ev.json")
        json.dump({"sr": SR, "dur": total, "events": events}, open(ej, "w"))
        subprocess.run([NODE, RENDER, SOUND_JS, ej, os.path.join(tmp, "s")], check=True, stdout=subprocess.DEVNULL)
        dry = np.fromfile(os.path.join(tmp, "s.dry.f32"), np.float32).reshape(-1, 2).astype(np.float64)
        ir = np.fromfile(os.path.join(tmp, "s.ir.f32"), np.float32).reshape(-1, 2).astype(np.float64)
    bus = 0.9 * dry
    wet = np.stack([fftconvolve(bus[:, c], ir[:, c])[:len(bus)] for c in range(2)], 1)
    mixd = compress(bus + wet)[:int(total * SR)]
    mixd = np.clip(mixd, -1, 1)
    wav = os.path.join(out, "game.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mixd * 32767).astype(np.int16).tobytes())
    src = os.path.join(out, "game.mp4")
    tmpv = os.path.join(out, "game_snd.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", tmpv],
                   check=True)
    os.replace(tmpv, src)
    os.remove(wav)
    kinds = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"sound (viewer's sound.js): {len(events)} events {kinds} over {total:.1f}s, peak {np.abs(mixd).max():.2f} -> {src}")


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {k: v for k, v in (x[2:].split("=") for x in a if x.startswith("--"))}
    main([x for x in a if not x.startswith("--")][0], float(kw.get("speed", 1.5)), int(kw.get("fps", 30)))
