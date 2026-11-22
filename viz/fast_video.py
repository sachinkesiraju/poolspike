"""Game video from replay_out/video_items.json — pooltool 3D render, fast.

Each shot is re-simulated from its pre-shot state + EXECUTED stroke (the sim
is deterministic) and rendered by pooltool's Panda3D scene (overhead
camera). Speed comes from: one persistent ShowBase per worker process (the
shader/scene warm-up is paid once, not per shot), vsync off (~8 ms/frame),
cloth tinted green on the scene nodes (no per-pixel recolor), overlays and
the banner/options panel composited in numpy, raw frames piped straight
into ffmpeg, shots rendered in parallel (longest first) and joined with
stream copy. No JPEG round-trips, no post-passes.

Per shot: aim hold (call + intent lines + ranked options) → motion with the
cue trail → rest hold on the re-rendered post-shot table.

--flat renders the same video top-down in pure OpenCV (no GPU needed).

Usage: python fast_video.py [--flat] [--fps=30] [--speed=1.5] [--out=...]
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from multiprocessing import get_context

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))  # game engine, policy

OUT = os.environ.get("POOL_OUT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "replay_out")
W, H = 1600, 900
S = 560.0                      # px per metre
RAIL = 36
TX, TY = 0.9906, 1.9812        # table width (x), length (y), metres
X0, Y0 = 46, 200 + RAIL      # playing-surface origin (screen px)
BALL_R = 0.028575
POCKETS = {"lb": (0.0, 0.0), "lt": (0.0, TY), "rb": (TX, 0.0),
           "rt": (TX, TY), "lc": (0.0, TY / 2), "rc": (TX, TY / 2)}
PANEL_X = int(X0 + TY * S + RAIL + 22)

COLORS = {  # RGB
    "1": (250, 200, 20), "2": (30, 80, 200), "3": (215, 40, 40),
    "4": (110, 40, 140), "5": (245, 120, 20), "6": (30, 130, 60),
    "7": (130, 30, 40), "8": (20, 20, 20), "cue": (245, 245, 240)}
CYAN, RED, GOLD = (255, 230, 0), (60, 60, 255), (60, 200, 255)  # BGR


def px(x: float, y: float) -> tuple[float, float]:
    """Table metres → screen px. Long axis (y) runs left→right."""
    return X0 + y * S, Y0 + (TX - x) * S


def _font(size, bold=False):
    from replay import font
    return font(size, bold)


# ------------------------------------------------------------------ sprites

def ball_sprite(bid: str) -> np.ndarray:
    """RGBA sprite (BGR + alpha) of one ball, 4x supersampled."""
    d = int(round(2 * BALL_R * S)) + 2
    k = 4
    D = d * k
    im = Image.new("RGBA", (D, D), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    n = int(bid) if bid.isdigit() else 0
    base = COLORS["cue"] if bid == "cue" else COLORS[str(n - 8 if n > 8 else n)]
    if n > 8:
        dr.ellipse([0, 0, D - 1, D - 1], fill=(245, 245, 240, 255))
        band = Image.new("RGBA", (D, D), (0, 0, 0, 0))
        ImageDraw.Draw(band).rectangle([0, D * 0.24, D, D * 0.76],
                                       fill=base + (255,))
        mask = Image.new("L", (D, D), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, D - 1, D - 1], fill=255)
        im.paste(band, (0, 0), Image.composite(band, Image.new("RGBA", (D, D)),
                                               mask).split()[3])
    else:
        dr.ellipse([0, 0, D - 1, D - 1], fill=base + (255,))
    dr.ellipse([0, 0, D - 1, D - 1], outline=(0, 0, 0, 200), width=k * 2)
    if bid != "cue":
        r = D * 0.26
        c = D / 2
        dr.ellipse([c - r, c - r, c + r, c + r], fill=(250, 250, 250, 255))
        f = _font(int(D * 0.36), bold=True)
        tb = dr.textbbox((0, 0), bid, font=f)
        dr.text((c - (tb[2] - tb[0]) / 2 - tb[0], c - (tb[3] - tb[1]) / 2 - tb[1]),
                bid, font=f, fill=(15, 15, 15, 255))
    im = im.resize((d, d), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    out = np.empty_like(a)
    out[..., :3] = a[..., 2::-1]
    out[..., 3] = a[..., 3] / 255.0
    return out


def blit(frame: np.ndarray, spr: np.ndarray, cx: float, cy: float) -> None:
    d = spr.shape[0]
    x0, y0 = int(round(cx - d / 2)), int(round(cy - d / 2))
    x1, y1 = x0 + d, y0 + d
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    a = spr[..., 3:4]
    frame[y0:y1, x0:x1] = (roi * (1 - a) + spr[..., :3] * a).astype(np.uint8)


# ------------------------------------------------------------- static layers

def table_layer() -> Image.Image:
    im = Image.new("RGB", (W, H), (18, 22, 28))
    d = ImageDraw.Draw(im)
    x1, y1 = px(TX, TY)[0], px(0, TY)[1]
    xa, ya = px(TX, 0)
    d.rounded_rectangle([X0 - RAIL, ya - RAIL, x1 + RAIL, y1 + RAIL], 18,
                        fill=(92, 52, 26))
    d.rectangle([X0, ya, x1, y1], fill=(22, 112, 62))
    d.rectangle([X0, ya, x1, y1], outline=(12, 80, 40), width=4)
    for i in range(1, 8):  # diamonds along the long rails
        x = X0 + i * TY * S / 8
        for y in (ya - RAIL / 2, y1 + RAIL / 2):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(235, 225, 200))
    for i in range(1, 4):
        y = ya + i * TX * S / 4
        for x in (X0 - RAIL / 2, x1 + RAIL / 2):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(235, 225, 200))
    for pk, (bx, by) in POCKETS.items():
        cx, cy = px(bx, by)
        r = 0.062 * S * (0.92 if pk[1] == "c" else 1.0)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(5, 5, 5))
    f = _font(15, True)
    for pk, (bx, by) in POCKETS.items():  # pocket ids, as used in calls
        cx, cy = px(bx, by)
        lx = cx + (26 if by == TY else -46 if by == 0 else -9)
        ly = cy + (22 if bx == 0 else -40)
        d.text((lx, ly), pk, font=f, fill=(235, 200, 120))
    hs = X0 + TY / 4 * S  # head string
    for yy in range(int(ya), int(y1), 12):
        d.line([hs, yy, hs, yy + 5], fill=(60, 150, 95), width=1)
    return im


def wrap(d, txt, f, width):
    words, lines, ln = txt.split(), [], ""
    for w_ in words:
        if d.textlength((ln + " " + w_).strip(), font=f) <= width:
            ln = (ln + " " + w_).strip()
        else:
            lines.append(ln)
            ln = w_
    lines.append(ln)
    return lines


def _names():
    # POOL_NAMES="A=Poolspike|B=EV search"; default keeps "Player A"
    return dict(x.split("=", 1) for x in os.environ.get("POOL_NAMES", "").split("|") if "=" in x)


def pname(p):
    return _names().get(p, f"Player {p}")


def pshort(p):
    n = _names().get(p)
    return n.split(" (")[0] if n else p


def shot_layer(base: Image.Image, it: dict, panel_x: int = PANEL_X,
               panel_bottom: int = 840, panel_top: int = 200) -> np.ndarray:
    im = base.copy()
    d = ImageDraw.Draw(im)
    f_b, f, f_s = _font(22, True), _font(19), _font(16)
    # banner
    head = f"Shot {it['shot_no']:02d} · {pname(it['player'])}"
    d.text((24, 14), head, font=f_b, fill=(255, 210, 90))
    y = 46
    for ln in wrap(d, it["narration"], f, base.size[0] - 48)[:4]:
        d.text((24, y), ln, font=f, fill=(235, 235, 235))
        y += 24
    # panel: groups + ranked options
    x = panel_x
    pw = base.size[0] - x - 20
    d.rounded_rectangle([x, panel_top, x + pw, panel_bottom], 10, fill=(28, 34, 42))
    d.text((x + 14, panel_top + 12), "Game state", font=f_b, fill=(255, 210, 90))
    yy = panel_top + 44
    live = {b for b, v in it["state"]["balls"].items()
            if b != "cue" and not v.get("pocketed")}
    for p in ("A", "B"):
        g = (it.get("groups") or {}).get(p)
        if g:
            left = sorted((live & set(g)), key=int)
            name = "solids" if g[0] == "1" else "stripes"
            txt = f"{pshort(p)}: {name} · {len(left)} left" + (" · on the 8" if not left else "")
        else:
            txt = f"{pshort(p)}: table open"
        col = (255, 255, 255) if p == it["player"] else (160, 165, 175)
        if p == it["player"]:
            d.polygon([(x + 14, yy + 3), (x + 14, yy + 17), (x + 25, yy + 10)],
                      fill=(255, 210, 90))
        d.text((x + 32, yy), txt, font=f_s, fill=col)
        yy += 24
    plan = it["call"].get("plan") or {}
    if plan.get("ball"):
        who = "next shot" if plan["for"] == "self" else "opponent's best look"
        d.text((x + 14, yy + 4), f"Plan: {who} {plan['ball']}→{plan['pocket']}"
               f" ~{plan['p']:.0%}", font=f_s, fill=(140, 230, 255))
        yy += 26
    yy += 12
    d.text((x + 14, yy), it["call"].get("panel_title", "Options ranked by EV"), font=f_b, fill=(255, 210, 90))
    yy += 32
    contest = it["call"].get("contest") or []
    if it["call"]["kind"] == "break":
        c = it["call"]
        rows = [f"break {it['stroke'][1]:.0f} m/s" + (f" x={c['cue_place'][0]:.2f}" if c.get("cue_place") else "")]
        if "p_continue" in c:
            rows.append(f"keep table {c['p_continue']:.0%} · scratch {c['p_scratch']:.0%}")
        if c.get("ev") is not None:
            rows.append(f"EV {c['ev']:.2f}")
        for r in rows:
            d.text((x + 14, yy), r, font=f_s, fill=(220, 220, 220))
            yy += 24
    for i, o in enumerate(contest[:7]):
        if o["kind"] == "pot":
            lab = o.get("label") or f"{o['ball']}→{o['pocket']}"
            sub = f"make {o['p_make']:.0%}   EV {o['ev']:.2f}"
        else:
            lab = f"safety off {o['ball']}" + (" (kick)" if o.get("route") == "kick" else "")
            sub = f"EV {o['ev']:.2f}"
        chosen = i == 0
        if chosen:
            d.rounded_rectangle([x + 8, yy - 4, x + pw - 8, yy + 44], 6,
                                fill=(40, 70, 52))
        # EV bar
        bw = (pw - 40) * max(0.0, min(1.0, o["ev"]))
        d.rectangle([x + 14, yy + 38, x + 14 + bw, yy + 41],
                    fill=(90, 200, 120) if chosen else (90, 110, 130))
        d.text((x + 14, yy), lab, font=f_s,
               fill=(255, 255, 255) if chosen else (200, 200, 205))
        d.text((x + 14, yy + 19), sub, font=f_s, fill=(170, 200, 180) if chosen else (150, 155, 165))
        yy += 52
    # legend
    for i, ln in enumerate(("cyan: planned cue path / actual cue path",
                            "red: planned path of the called ball",
                            "light ring: where the cue is sent for shape")):
        d.text((x + 14, panel_bottom - 72 + 22 * i), ln, font=f_s,
               fill=(150, 160, 170))
    a = np.asarray(im)
    return np.ascontiguousarray(a[..., ::-1])


# ------------------------------------------------------------------ drawing

def dashed(frame, p0, p1, color, width=2, dash=12, gap=8):
    import cv2
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    t = 0.0
    while t < L:
        t2 = min(t + dash, L)
        cv2.line(frame, (int(x0 + ux * t), int(y0 + uy * t)),
                 (int(x0 + ux * t2), int(y0 + uy * t2)), color, width,
                 cv2.LINE_AA)
        t = t2 + gap


def render_shot_flat(args) -> str:
    it, seg, fps, speed = args
    import cv2
    import pooltool as pt
    import policy as P

    base = table_layer()
    static = shot_layer(base, it)
    sprites = {b: ball_sprite(b) for b in it["state"]["balls"]}
    state = it["state"]
    shot = P.simulate(state, tuple(it["stroke"]))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
           "-pix_fmt", "yuv420p", seg]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    pre = {b: px(v["x"], v["y"]) for b, v in state["balls"].items()
           if not v.get("pocketed")}
    call = it["call"]
    cue = pre["cue"]

    def draw_balls(fr, pos):
        for b, (x, y) in pos.items():
            if b != "cue":
                blit(fr, sprites[b], x, y)
        if "cue" in pos:
            blit(fr, sprites["cue"], *pos["cue"])

    # ---- aim hold: intent only (no outcome)
    aim = static.copy()
    if call["kind"] != "break":
        phi = math.radians(it["call"]["stroke"][0])
        L = 0.45
        cx, cy = state["balls"]["cue"]["x"], state["balls"]["cue"]["y"]
        tip = px(cx + math.cos(phi) * L, cy + math.sin(phi) * L)
        dashed(aim, cue, tip, CYAN, 2)
        if call.get("ball") and call.get("pocket") and call["ball"] in pre:
            dashed(aim, pre[call["ball"]], px(*POCKETS[call["pocket"]]), RED, 3)
            pp = px(*POCKETS[call["pocket"]])
            cv2.circle(aim, (int(pp[0]), int(pp[1])), 26, RED, 3, cv2.LINE_AA)
        if call.get("cue_place"):
            cv2.circle(aim, (int(cue[0]), int(cue[1])), 30, GOLD, 2, cv2.LINE_AA)
            cv2.putText(aim, "ball in hand", (int(cue[0]) - 50, int(cue[1]) - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOLD, 2, cv2.LINE_AA)
    draw_balls(aim, pre)
    for _ in range(int(fps * 1.4)):
        ff.stdin.write(aim.tobytes())

    # ---- motion
    last = aim
    if shot is not None:
        pt.continuize(shot, dt=speed / fps, inplace=True)
        tracks = {}
        n = 0
        for b, ball in shot.balls.items():
            rvw, s, t = ball.history_cts.vectorize()
            tracks[b] = (rvw[:, 0, :2], s)
            n = max(n, len(s))
        trail = []
        for i in range(n):
            fr = static.copy()
            pos = {}
            for b, (xy, s) in tracks.items():
                k = min(i, len(s) - 1)
                if s[k] == 4:  # pocketed
                    continue
                pos[b] = px(float(xy[k, 0]), float(xy[k, 1]))
            if "cue" in pos:
                trail.append(pos["cue"])
            if len(trail) > 1:
                cv2.polylines(fr, [np.array(trail, np.int32)], False, CYAN, 2,
                              cv2.LINE_AA)
            draw_balls(fr, pos)
            ff.stdin.write(fr.tobytes())
            last = fr
    # ---- rest hold
    for _ in range(int(fps * 1.3)):
        ff.stdin.write(last.tobytes())
    ff.stdin.close()
    ff.wait()
    return seg


# ------------------------------------------------------------- 3D renderer

R3 = (1152, 720)        # pooltool render size (plane_H.json is calibrated here)
R3_Y = 132              # render's top edge on the canvas (below the banner)
H3 = R3_Y + R3[1]       # 3D canvas height
PANEL3_X = R3[0] + 16
CLOTH_TINT = (1.35, 0.98, 0.42, 1.0)  # teal pooltool cloth -> green
_IFACE = None


def _iface():
    """One ShowBase per worker process, created on first use."""
    global _IFACE
    if _IFACE is None:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "show-frame-rate-meter #f")
        loadPrcFileData("", "sync-video #f")
        import pooltool.ani.animate as anim
        from pooltool.ani.animate import FBF_FBP, FrameStepper, ShowBaseConfig
        anim._resize_offscreen_window = lambda size: None
        _IFACE = FrameStepper(config=ShowBaseConfig(
            window_type="offscreen", fb_prop=FBF_FBP, window_size=R3))
    return _IFACE


def _homography():
    H = np.linalg.inv(np.array(json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plane_H.json")))["H"]))

    def to_px(x, y):
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2] + R3_Y
    return to_px


def _frames(system, sim_fps):
    """Yield RGB frames of a simulated system from the overhead camera."""
    from pooltool.ani.camera import cam, camera_states
    from pooltool.ani.globals import Global
    from pooltool.ani.image.interface import (get_graphics_texture,
                                              image_array_from_texture)
    iface = _iface()
    iterator, n = iface.iterator(system, R3, sim_fps)
    for name in ("slate", "cushion_long", "cushion_short"):
        node = Global.render.find(f"**/{name}")
        if not node.isEmpty():
            node.setColorScale(*CLOTH_TINT)
    tex = get_graphics_texture()
    cam.load_state(camera_states["7_foot_overhead"])
    try:
        for _ in range(n):
            next(iterator)
            # image_array_from_texture returns RGB (row-flipped already)
            yield image_array_from_texture(tex)
    finally:
        iface.close_scene()


def planned_paths(state: dict, call: dict):
    """Intended paths from the NOMINAL stroke (what the player is trying to
    do — not the noisy outcome): cue up to first contact, and the called
    ball up to its pocket. Lets a viewer read banks, kicks and kisses."""
    import pooltool as pt
    import policy as P
    if call["kind"] == "break":
        return [], []
    shot = P.simulate(state, tuple(call["stroke"]))
    if shot is None:
        return [], []
    t_hit = next((e.time for e in shot.events
                  if e.event_type == pt.EventType.BALL_BALL
                  and "cue" in [a.id for a in e.agents]), None)
    t_drop = next((e.time for e in shot.events
                   if e.event_type == pt.EventType.BALL_POCKET
                   and call.get("ball") in [a.id for a in e.agents]), None)
    pt.continuize(shot, dt=0.01, inplace=True)

    def path(bid, t_end):
        rvw, s_, t = shot.balls[bid].history_cts.vectorize()
        return [(float(x), float(y)) for (x, y), tt in zip(rvw[:, 0, :2], t)
                if t_end is None or tt <= t_end + 1e-9]
    cue = path("cue", t_hit) if t_hit is not None else path("cue", 1.0)
    obj = (path(call["ball"], t_drop)
           if call["kind"] == "pot" and t_drop is not None else [])
    after = []
    if t_hit is not None and "cue" not in P.pocket_map(shot):
        rvw, s_, t = shot.balls["cue"].history_cts.vectorize()
        after = [(float(x), float(y)) for (x, y), tt in zip(rvw[:, 0, :2], t)
                 if tt >= t_hit]
    return cue, obj, after


def dashed_path(frame, pts, color, width=2):
    import cv2
    on, acc = True, 0.0
    for p0, p1 in zip(pts, pts[1:]):
        seg = math.dist(p0, p1)
        if on:
            cv2.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                     color, width, cv2.LINE_AA)
        acc += seg
        if acc > (12 if on else 8):
            on, acc = not on, 0.0


def label_balls(frame, pos: dict) -> None:
    """Ball numbers drawn on the balls: from overhead a rolling stripe can
    look solid (5 and 13 both read orange), so every ball gets its number."""
    import cv2
    for bid, (x, y) in pos.items():
        if bid == "cue":
            continue
        t = bid
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        org = (int(x - tw / 2), int(y + th / 2))
        cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)


def render_shot_3d(args) -> str:
    it, seg, fps, speed = args
    import cv2
    import pooltool as pt
    import policy as P

    to_px = _homography()
    chrome = shot_layer(Image.new("RGB", (W, H3), (18, 22, 28)), it,
                        panel_x=PANEL3_X, panel_top=R3_Y,
                        panel_bottom=H3 - 12)
    state = it["state"]
    call = it["call"]
    shot = P.simulate(state, tuple(it["stroke"]))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{W}x{H3}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", seg], stdin=subprocess.PIPE)

    def canvas(rgb):
        fr = chrome.copy()
        fr[R3_Y:R3_Y + R3[1], :R3[0]] = rgb[..., ::-1]
        return fr

    def ipt(p):
        return int(round(p[0])), int(round(p[1]))

    cue0 = to_px(state["balls"]["cue"]["x"], state["balls"]["cue"]["y"])
    pre = {b: to_px(v["x"], v["y"]) for b, v in state["balls"].items()
           if not v.get("pocketed")}
    last = None
    if shot is not None:
        sim_fps = fps / speed
        trail = []
        for i, rgb in enumerate(_frames(shot, sim_fps)):
            if i == 0:
                # ---- aim hold: intent only (no outcome)
                aim = canvas(rgb)
                if call["kind"] != "break":
                    cue_p, obj_p, after = planned_paths(state, call)
                    dashed_path(aim, [to_px(*q) for q in cue_p], CYAN, 2)
                    if len(after) > 1:
                        # position route: where the cue is sent for shape
                        pts = [to_px(*q) for q in after]
                        for q0, q1 in list(zip(pts, pts[1:]))[::3]:
                            cv2.line(aim, ipt(q0), ipt(q1), (200, 160, 90), 1,
                                     cv2.LINE_AA)
                        cv2.circle(aim, ipt(pts[-1]), 15, (255, 200, 120), 2,
                                   cv2.LINE_AA)
                    if obj_p:
                        dashed_path(aim, [to_px(*q) for q in obj_p], RED, 3)
                    if call.get("pocket"):
                        pp = to_px(*POCKETS[call["pocket"]])
                        cv2.circle(aim, ipt(pp), 24, RED, 3, cv2.LINE_AA)
                    if call.get("cue_place"):
                        cv2.circle(aim, ipt(cue0), 28, GOLD, 2, cv2.LINE_AA)
                        cv2.putText(aim, "ball in hand",
                                    (ipt(cue0)[0] - 50, ipt(cue0)[1] - 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOLD, 2,
                                    cv2.LINE_AA)
                label_balls(aim, pre)
                for _ in range(int(fps * 1.4)):
                    ff.stdin.write(aim.tobytes())
            fr = canvas(rgb)
            now = {}
            for bid, ball in shot.balls.items():
                sts = ball.history_cts.states
                st_ = sts[min(i, len(sts) - 1)]
                if st_.s != 4:
                    now[bid] = to_px(float(st_.rvw[0][0]), float(st_.rvw[0][1]))
            if "cue" in now:
                trail.append(ipt(now["cue"]))
            if len(trail) > 1:
                cv2.polylines(fr, [np.array(trail, np.int32)], False, CYAN, 2,
                              cv2.LINE_AA)
            label_balls(fr, now)
            ff.stdin.write(fr.tobytes())
            last = fr
        # ---- rest hold: re-render the post-shot table so potted balls are
        # gone (the last motion frame leaves them at the pocket lip)
        # the logged post-state is authoritative (it is the next shot's
        # start); re-sim is only a fallback for old logs without it
        post = it.get("post_state") or P.post_state(shot, state)
        rest = None
        if not post["balls"].get("cue", {}).get("pocketed"):
            # (after a scratch pooltool can't stage a cue-less table: the
            # last motion frame already shows it, so hold that instead)
            rest_sys = build_rest_system(post)
            pt.simulate(rest_sys, inplace=True, t_final=0.01)
            for rgb in _frames(rest_sys, 100):
                rest = canvas(rgb)
        if rest is not None and len(trail) > 1:
            cv2.polylines(rest, [np.array(trail, np.int32)], False, CYAN, 2,
                          cv2.LINE_AA)
        if rest is not None:
            label_balls(rest, {b: to_px(v["x"], v["y"])
                               for b, v in post["balls"].items()
                               if not v.get("pocketed")})
        last = rest if rest is not None else last
    if last is None:
        last = chrome
    for _ in range(int(fps * 1.3)):
        ff.stdin.write(last.tobytes())
    ff.stdin.close()
    ff.wait()
    return seg


def build_rest_system(post):
    from pool_env import build_system
    sys_ = build_system(post)
    # pt.Cue defaults to a V0=2 stroke: zero it or the "rest" frame fires a
    # phantom shot
    sys_.cue.set_state(V0=0.0, phi=0.0, theta=0.0, a=0.0, b=0.0)
    return sys_


def _length(it) -> float:
    return it["stroke"][1]  # proxy for shot duration (faster = longer roll)


def main():
    fps = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--fps=")), 30)
    speed = next((float(a.split("=")[1]) for a in sys.argv
                  if a.startswith("--speed=")), 1.5)
    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")),
               os.path.join(OUT, "game.mp4"))
    items = json.load(open(os.path.join(OUT, "video_items.json")))
    seg_dir = os.path.join(OUT, "segments")
    shutil.rmtree(seg_dir, ignore_errors=True)
    os.makedirs(seg_dir)
    flat = "--flat" in sys.argv
    jobs = [(it, os.path.join(seg_dir, f"s{it['shot_no']:03d}.mp4"), fps, speed)
            for it in items]
    # longest shots first so no worker is left finishing the break alone
    order = sorted(jobs, key=lambda j: -_length(j[0]))
    n = max(1, (os.cpu_count() or 2) - 1)
    if not flat:
        n = min(n, int(os.environ.get("VIDEO_WORKERS", 8)))
    with get_context("spawn").Pool(n) as pool:
        pool.map(render_shot_flat if flat else render_shot_3d, order,
                 chunksize=1)
    segs = [j[1] for j in jobs]
    lst = os.path.join(seg_dir, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    # faststart: index at the front so the browser starts playing at once
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    print(f"video: {out} ({len(segs)} shots)")
    import viewer
    viewer.build(OUT, back=bool(os.environ.get("POOL_OUT")))


if __name__ == "__main__":
    main()
