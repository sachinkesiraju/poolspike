"""Full-game 8-ball replay: two policy configs play a game, every shot is
narrated from measured quantities, and the game is rendered to panels/video.

The engine lives in `policy.py` (noise-aware EV search) and `game.py` (rules
+ turn loop). This module is presentation: narration, matplotlib panels,
pooltool hi-fi renders, and the video items consumed by fast_video.py.

Usage:
  python replay.py [seed] [--sigma=0.1] [--draft]   # play + log + video_items.json
  python replay.py --renarrate     # rebuild narration from the saved log
  python fast_video.py             # top-down game video in seconds
  REPLAY_HIFI=1 python replay.py   # also pooltool-rendered still panels
"""

import json
import math
import os
import sys

import numpy as np

import pooltool as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))  # game engine, policy
import game  # noqa: E402
import policy as P  # noqa: E402
from pool_env import build_system, phi_towards, pocket_map  # noqa: E402,F401
from pool_env import first_contact  # noqa: E402,F401

TABLE = pt.Table.default()
BALL_R = pt.BallParams().R
# pooltool coords: x across the width (0..w=0.99), y along the length (0..l=1.98).
TX, TY = float(TABLE.w), float(TABLE.l)
TW, TL = TX, TY
POCKETS = P.POCKETS


def font(size: int, bold: bool = False):
    """DejaVu on Linux, Arial/Helvetica on macOS, PIL default otherwise."""
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"]
             if bold else ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"])
    dirs = ["/usr/share/fonts/truetype/dejavu", "/System/Library/Fonts/Supplemental",
            "/System/Library/Fonts", "/Library/Fonts"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def cloth_green_mask(raw, cloth_med, inner, outer, min_comp):
    """Pixels to recolor green: cloth-colored pixels inside `inner`, plus
    the contiguous tan cushion band in the ring (outer minus inner) matched
    by hue — only large components kept so balls/pockets in the band are
    not recolored."""
    import cv2
    dist = np.linalg.norm(raw.astype(float) - cloth_med, axis=2)
    hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV)
    tan = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 30)
           & (hsv[..., 1] > 30) & (hsv[..., 2] > 50) & (hsv[..., 2] < 200))
    ring = (outer > 0) & (inner == 0)
    n, lab = cv2.connectedComponents((tan & ring).astype(np.uint8))
    cushion = np.zeros(raw.shape[:2], bool)
    for i in range(1, n):
        if np.count_nonzero(lab == i) > min_comp:
            cushion |= lab == i
    return ((dist < 45) & (inner > 0)) | cushion


def ledger(shot: pt.System) -> dict:
    contacts = []
    for ev in shot.events:
        if ev.event_type == pt.EventType.BALL_BALL:
            ids = [a.id for a in ev.agents]
            if "cue" in ids:
                other = ids[0] if ids[1] == "cue" else ids[1]
                if not contacts or contacts[-1] != other:
                    contacts.append(other)
    return {"contacts": contacts}


def cue_path(shot: pt.System) -> list[tuple[float, float]]:
    """Cue-ball trail through the event stream (from viz_steps' approach)."""
    pts = []
    for st in shot.balls["cue"].history.states:
        pts.append((float(st.rvw[0][0]), float(st.rvw[0][1])))
    return pts


def panel(state_prev: dict, turn: dict, path: str) -> None:
    """One PNG per shot: all balls pre-shot, call annotation, cue path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(-0.08, TX + 0.08)
    ax.set_ylim(-0.08, TY + 0.08)
    ax.add_patch(mp.Rectangle((0, 0), TX, TY, fc="#0a6e3f", ec="#222", lw=2))
    for pkx, pky in POCKETS.values():
        ax.add_patch(mp.Circle((pkx, pky), 0.055, fc="#111", ec="#333", zorder=3))
    grp = turn.get("player_group")
    for bid, b in state_prev["balls"].items():
        z = 5 if bid != "cue" else 6
        fc = "white" if bid == "cue" else "#111" if bid == "8" else (
            "#c77" if grp and bid in grp else "#88c")
        ax.add_patch(mp.Circle((b["x"], b["y"]), BALL_R * 1.4, fc=fc,
                               ec="#333", lw=0.5, zorder=z))
        ax.annotate(bid, (b["x"], b["y"]), ha="center", va="center",
                    fontsize=6, zorder=z + 1)
    call = turn["call"]
    shot = turn.get("shot_obj")
    if shot is not None:
        pts = cue_path(shot)
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#ffd24a", lw=1.4, zorder=4, alpha=0.9)
            ax.plot(pts[-1][0], pts[-1][1], "o", ms=6, mfc="none",
                    mec="#ffd24a", zorder=6)
    if call["kind"] == "pot":
        b = state_prev["balls"][call["ball"]]
        px, py = POCKETS[call["pocket"]]
        ax.annotate("", xy=(px, py), xytext=(b["x"], b["y"]),
                    arrowprops=dict(arrowstyle="->", color="#ffd24a", lw=2))
        title = f"{turn['player']}: {call['ball']}→{call['pocket']} (margin ≈{call['margin']:.1f}°)"
    elif call["kind"] == "break":
        title = f"{turn['player']}: break"
    else:
        b = state_prev["balls"].get(call["ball"])
        title = f"{turn['player']}: safe off {call['ball']}"
    ax.set_title(f"shot {turn['shot_no']:02d} — {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


_IFACE = None


def _hifi_iface(SIZE):
    """One FrameStepper per process — Panda3D allows a single ShowBase."""
    global _IFACE
    if _IFACE is None:
        import pooltool.ani.animate as anim
        from pooltool.ani.animate import (FrameStepper, ShowBaseConfig,
                                          FBF_FBP)
        anim._resize_offscreen_window = lambda size: None
        cfg = ShowBaseConfig(window_type="offscreen", fb_prop=FBF_FBP,
                             window_size=SIZE)
        _IFACE = FrameStepper(config=cfg)
    return _IFACE


def render_hifi_panels(items: list[dict], out_dir: str) -> None:
    """High-fidelity panels: pooltool's own rendered table per shot +
    call arrow + measured cue path + banner, overlaid in image space via
    the calibrated z=R homography (demo/plane_H.json)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    import pooltool.ani.animate as anim
    from pooltool.ani.animate import FrameStepper, ShowBaseConfig, FBF_FBP
    from pooltool.ani.image.interface import image_stack
    from pooltool.ani.camera import camera_states

    SIZE = (1152, 720)
    H = np.linalg.inv(np.array(json.load(open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_H.json")
    ))["H"]))  # table (z=R) -> image

    def to_px(x, y):
        v = H @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    iface = _hifi_iface(SIZE)
    try:
        font = globals()["font"](26, bold=True)
        small = globals()["font"](20)
    except Exception:  # noqa: BLE001
        font = small = ImageFont.load_default()

    report = []
    for it in items:
        sys_ = build_system(it["state"])
        # image_stack needs events to animate; t_final=0.01 yields ~1 frame
        # (~1.5s vs ~90s for a full settle animation).
        pt.simulate(sys_, inplace=True, t_final=0.01)
        imgs = image_stack(system=sys_, interface=iface, size=SIZE, fps=5,
                           camera_state=camera_states["7_foot_overhead"])
        # create_scene() attaches a new "scene" node per call but never removes
        # the previous one — without close_scene() every panel renders ghost
        # balls from all earlier states (looked like extra cue balls).
        iface.close_scene()
        raw = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)

        # --- per-frame validation (on the raw render, before overlays) ---
        corners = np.array([to_px(0, 0), to_px(0, TY), to_px(TX, TY),
                            to_px(TX, 0)], np.int32)
        interior = np.zeros(raw.shape[:2], np.uint8)
        cv2.fillConvexPoly(interior, corners, 255)
        inner = cv2.erode(interior, np.ones((9, 9), np.uint8))
        outer = cv2.dilate(interior,
                           np.ones((int(SIZE[0] * 0.085) | 1,) * 2,
                                   np.uint8))
        cloth_med = np.median(raw[inner > 0], axis=0)
        live = [bid for bid, b in it["state"]["balls"].items()
                if not b.get("pocketed")]
        missing = []
        for bid in live:
            bx, by = to_px(it["state"]["balls"][bid]["x"],
                           it["state"]["balls"][bid]["y"])
            if not (0 <= bx < SIZE[0] and 0 <= by < SIZE[1]):
                missing.append(bid); continue
            disc = np.zeros(raw.shape[:2], np.uint8)
            cv2.circle(disc, (int(bx), int(by)), 11, 255, -1)
            n_on = np.count_nonzero(
                np.linalg.norm(raw.astype(float) - cloth_med, axis=2) > 45)
            sub = (disc > 0) & (inner > 0)
            frac = (np.linalg.norm(raw.astype(float) - cloth_med, axis=2)
                    [sub] > 45).mean()
            if frac < 0.45:
                missing.append(bid)
        # ghost check: ball-sized blobs in interior not near any live ball
        dif = (np.linalg.norm(raw.astype(float) - cloth_med, axis=2) > 45) \
            & (inner > 0)
        n, lab = cv2.connectedComponents(dif.astype(np.uint8))
        exp_px = {to_px(it["state"]["balls"][b]["x"],
                        it["state"]["balls"][b]["y"]) for b in live}
        pocket_px = [to_px(x, y) for x, y in POCKETS.values()]
        ghosts = 0
        for i in range(1, n):
            ys, xs = np.where(lab == i)
            r = math.sqrt(len(xs) / math.pi)
            if 6 <= r <= 22:
                cx, cy = xs.mean(), ys.mean()
                near_pocket = min(
                    math.hypot(cx-px, cy-py) for px, py in pocket_px) < 32
                near_ball = min(
                    math.hypot(cx-px, cy-py) for px, py in exp_px) < 30
                if not near_pocket and not near_ball:
                    ghosts += 1
        val = (f"[{it['shot_no']:02d}] balls {len(live)-len(missing)}/{len(live)}"
               f" rendered, ghost blobs {ghosts}")
        if missing:
            val += f" MISSING: {missing}"
        report.append(val)

        # --- green felt: recolor cloth + cushion band ---
        cloth_px = cloth_green_mask(raw, cloth_med, inner, outer, 1150)
        hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0][cloth_px] = 65
        hsv[..., 1][cloth_px] = np.clip(hsv[..., 1][cloth_px] + 25, 0, 255)
        raw = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        im = Image.fromarray(raw)
        d = ImageDraw.Draw(im, "RGBA")
        call = it["call"]
        CUE = (0, 220, 255, 255)      # cyan  — cue's measured path
        POT = (255, 60, 40, 255)      # red   — called ball -> pocket
        MOVED = (255, 170, 40, 255)   # orange — other balls that moved

        def badge(x, y, text, color):
            r = 13
            d.ellipse([x-r, y-r, x+r, y+r], fill=color)
            tb = d.textbbox((0, 0), text, font=small)
            d.text((x - (tb[2]-tb[0])/2, y - (tb[3]-tb[1])/2 - tb[1]),
                   text, font=small, fill=(0, 0, 0, 255))

        # Label every live ball: the ballset renders stripes white-with-rim
        # from overhead, which otherwise reads as multiple cue balls.
        for bid, b in it["state"]["balls"].items():
            if b.get("pocketed"):
                continue
            bx, by = to_px(b["x"], b["y"])
            if bid == "cue":
                r = 10
                d.ellipse([bx-r, by-r, bx+r, by+r], fill=(0, 150, 200, 230))
                tb = d.textbbox((0, 0), "C", font=small)
                d.text((bx-(tb[2]-tb[0])/2, by-(tb[3]-tb[1])/2-tb[1]),
                       "C", font=small, fill=(255, 255, 255, 255))
                continue
            stripe = int(bid) > 8
            col = (30, 30, 30, 220) if stripe else (245, 245, 245, 230)
            fg = (255, 255, 255, 255) if stripe else (20, 20, 20, 255)
            r = 10
            d.ellipse([bx-r, by-r, bx+r, by+r], fill=col)
            tb = d.textbbox((0, 0), bid, font=small)
            d.text((bx-(tb[2]-tb[0])/2, by-(tb[3]-tb[1])/2-tb[1]),
                   bid, font=small, fill=fg)

        # moved object balls: dashed trail + ring at final rest
        for bid, ball in it["shot"].balls.items():
            if bid == "cue":
                continue
            st = ball.history.states
            if len(st) < 2:
                continue
            path = [(s.rvw[0][0], s.rvw[0][1]) for s in st]
            dist = math.dist(path[0], path[-1])
            if dist < 0.015:
                continue
            px = [to_px(x, y) for x, y in path]
            d.line(px, fill=MOVED, width=3)
            ex, ey = px[-1]
            d.ellipse([ex-9, ey-9, ex+9, ey+9], outline=MOVED, width=3)
            d.text((ex+12, ey-12), bid, font=small, fill=MOVED)

        # cue trail
        pts = cue_path(it["shot"])
        if len(pts) > 1:
            px = [to_px(x, y) for x, y in pts]
            d.line(px, fill=CUE, width=3)
            d.ellipse([px[-1][0]-8, px[-1][1]-8, px[-1][0]+8, px[-1][1]+8],
                      outline=CUE, width=3)
            d.text((px[-1][0]+11, px[-1][1]-12), "cue",
                   font=small, fill=CUE)

        # called pot arrow + called-ball badge
        if call["kind"] == "pot" and call.get("ball"):
            b = it["state"]["balls"][call["ball"]]
            bx, by = to_px(b["x"], b["y"])
            badge(bx, by, call["ball"], POT)
            px_, py_ = POCKETS[call["pocket"]]
            tx, ty = to_px(px_, py_)
            d.line([(bx, by), (tx, ty)], fill=POT, width=6)
            ang = math.atan2(ty - by, tx - bx)
            for sgn in (-1, 1):
                d.line([(tx, ty),
                        (tx - 24 * math.cos(ang + sgn * 0.5),
                         ty - 24 * math.sin(ang + sgn * 0.5))],
                       fill=POT, width=6)
            d.text((tx - 40, ty - 30), call["pocket"],
                   font=small, fill=POT)
        elif call.get("ball"):
            b = it["state"]["balls"][call["ball"]]
            bx, by = to_px(b["x"], b["y"])
            badge(bx, by, call["ball"], (160, 80, 255, 255))

        # legend
        lx, ly = 12, SIZE[1] - 88
        d.rectangle([lx, ly, lx + 250, ly + 76], fill=(0, 0, 0, 170))
        d.line([(lx+8, ly+14), (lx+40, ly+14)], fill=CUE, width=3)
        d.text((lx+48, ly+6), "cue path", font=small, fill=CUE)
        d.line([(lx+8, ly+38), (lx+40, ly+38)], fill=POT, width=4)
        d.text((lx+48, ly+30), "called pot", font=small, fill=POT)
        d.line([(lx+8, ly+62), (lx+40, ly+62)], fill=MOVED, width=3)
        d.text((lx+48, ly+54), "moved balls", font=small, fill=MOVED)
        # banner — two lines on word boundaries, no truncation
        txt = f"{it['shot_no']:02d}  {it['narration']}"
        lines, cur = [], ""
        for word in txt.split(" "):
            if len(cur) + len(word) + 1 > 105 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            lines.append(cur)
        lines = lines[:3]
        d.rectangle([0, 0, SIZE[0], 16 + 26 * len(lines)],
                    fill=(0, 0, 0, 200))
        for i, ln in enumerate(lines):
            d.text((12, 8 + 26 * i), ln, font=small,
                   fill=(255, 255, 255, 255))
        # validation stamp (bottom-right)
        ok = not missing and ghosts == 0
        d.text((SIZE[0] - 200, SIZE[1] - 30),
               "validated" if ok else "CHECK", font=small,
               fill=(80, 255, 120, 255) if ok else (255, 80, 80, 255))
        im.save(os.path.join(out_dir, f"hifi_{it['shot_no']:02d}.png"))

    print("\n".join(report))


def render_video_frames(items: list[dict], frames_dir: str) -> None:
    """Real animation: plays each shot's full simulated trajectory through
    image_stack (balls actually rolling/caroming), green-felt recolor +
    narration banner on every frame."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    import pooltool.ani.animate as anim
    from pooltool.ani.animate import FrameStepper, ShowBaseConfig, FBF_FBP
    from pooltool.ani.image.interface import image_stack
    from pooltool.ani.camera import camera_states

    SIZE = (1152, 720)
    H = np.linalg.inv(np.array(json.load(open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_H.json")
    ))["H"]))  # table (z=R) -> image

    def to_px(x, y):
        v = H @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    iface = _hifi_iface(SIZE)
    try:
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:  # noqa: BLE001
        small = ImageFont.load_default()

    os.makedirs(frames_dir, exist_ok=True)
    fi = 0
    for it in items:
        imgs = image_stack(system=it["shot"], interface=iface, size=SIZE,
                           fps=24,
                           camera_state=camera_states["7_foot_overhead"])
        iface.close_scene()
        if imgs is None or len(imgs) == 0:
            continue
        corners = np.array([to_px(0, 0), to_px(0, TY), to_px(TX, TY),
                            to_px(TX, 0)], np.int32)
        interior = np.zeros(imgs[0].shape[:2], np.uint8)
        cv2.fillConvexPoly(interior, corners, 255)
        inner = cv2.erode(interior, np.ones((9, 9), np.uint8))
        outer = cv2.dilate(interior,
                           np.ones((int(SIZE[0] * 0.085) | 1,) * 2,
                                   np.uint8))
        cloth_med = np.median(
            cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)[inner > 0], axis=0)

        # banner text: shot no + narration
        txt = f"[{it['shot_no']:02d}] {it['narration']}"
        im0 = Image.new("RGBA", (SIZE[0], 56), (0, 0, 0, 190))
        d0 = ImageDraw.Draw(im0)
        words, lines, ln = txt.split(), [], ""
        for w in words:
            if d0.textlength(ln + " " + w, font=small) < SIZE[0] - 24:
                ln = (ln + " " + w).strip()
            else:
                lines.append(ln); ln = w
        lines.append(ln)
        lines = lines[:2]
        banner = Image.new("RGBA", (SIZE[0], 56), (0, 0, 0, 0))
        db = ImageDraw.Draw(banner)
        db.rectangle([0, 0, SIZE[0], 24 * len(lines) + 10],
                     fill=(0, 0, 0, 190))
        for i, l in enumerate(lines):
            db.text((12, 5 + 24 * i), l, font=small,
                    fill=(255, 255, 255, 255))

        for img in imgs:
            raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # green felt recolor (cloth + cushion band)
            mask = cloth_green_mask(raw, cloth_med, inner, outer, 1150)
            hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = np.where(mask, 65, hsv[..., 0])
            hsv[..., 1] = np.clip(hsv[..., 1] + np.where(mask, 25, 0),
                                  0, 255)
            rec = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            im = Image.fromarray(rec).convert("RGBA")
            im.alpha_composite(banner)
            im.convert("RGB").save(
                os.path.join(frames_dir, f"f_{fi:05d}.jpg"), quality=90)
            fi += 1
        # hold the rest state ~1.2s so the viewer can read the outcome
        for _ in range(28):
            im.convert("RGB").save(
                os.path.join(frames_dir, f"f_{fi:05d}.jpg"), quality=90)
            fi += 1
        print(f"  shot {it['shot_no']:02d}: {len(imgs)} frames", flush=True)



def exec_note(stroke) -> str:
    """Human label for a stroke (phi, V, a, b)."""
    _, v, a, b = stroke
    spin = ("draw" if b <= -0.35 else "soft draw" if b < -0.1 else
            "follow" if b >= 0.35 else "soft follow" if b > 0.1 else "stun")
    side = ("left english" if a < -0.1 else
            "right english" if a > 0.1 else "center")
    pace = ("soft" if v < 1.6 else "firm" if v > 3.0 else "medium")
    return f"{spin}/{side}, {pace} {v:.1f} m/s"


def _opt(o: dict) -> str:
    if o["kind"] == "pot":
        return f"{o['label']} {o['p_make']:.0%} (EV {o['ev']:.2f})"
    return f"safe off {o['ball']}{' kick' if o.get('route') == 'kick' else ''} (EV {o['ev']:.2f})"


def narrate(rec: dict) -> str:
    """One banner line per shot, built only from measured quantities."""
    c, who = rec["call"], rec["player"]
    pm = {b: p for b, p in rec["pocketed"].items() if b != "cue"}
    down = ", ".join(f"{b}→{p}" for b, p in pm.items())
    if c["kind"] == "break":
        s = (f"{who} breaks from x={c['cue_place'][0]:.2f} at "
             f"{rec['stroke'][1]:.0f} m/s (chosen from 30 break options, "
             f"{c['p_continue']:.0%} sampled keep-the-table)")
        s += f" — {len(pm)} down ({down})" if pm else " — dry break"
        if rec.get("groups_assigned"):
            g = rec["groups_assigned"][who]
            s += f"; {who} takes {'solids' if g[0] == '1' else 'stripes'}"
        elif pm and rec["result"] != "foul":
            s += "; table stays open"
        if rec["result"] == "foul":
            s += "; SCRATCH — opponent ball in hand in the kitchen"
        return s + "."
    parts = []
    if rec.get("bih"):
        x, y = c["cue_place"]
        parts.append(f"{who} has ball in hand — places cue at ({x:.2f}, {y:.2f})")
    else:
        parts.append(who)
    if c["kind"] == "pot":
        desc = f"shoots {P._label(c)}"
        plan = c.get("plan") or {}
        shape = (f", shape for {plan['ball']}→{plan['pocket']} "
                 f"(~{plan['p']:.0%} next)"
                 if plan.get("for") == "self" and plan.get("ball") else "")
        parts.append(f"{desc}: {c['p_make']:.0%} to make (aim basin "
                     f"{c['margin']:.2f}°), {exec_note(rec['stroke'])}{shape}; "
                     f"EV {c['ev']:.2f} (if made {c['v_make']:.2f}, if missed "
                     f"{c['v_miss']:.2f})")
    else:
        kind = "kick" if c.get("route") == "kick" else "hit"
        parts.append(f"has nothing to call — plays a legal {kind} off "
                     f"{c['ball']} ({exec_note(rec['stroke'])}); EV {c['ev']:.2f}"
                     if not c.get("best_pot") else
                     f"plays safe off {c['ball']} "
                     f"({exec_note(rec['stroke'])}); EV {c['ev']:.2f}")
        if c.get("best_pot"):
            bp = c["best_pot"]
            parts.append(f"over best pot {bp['label']} "
                         f"({bp['p_make']:.0%}, EV {bp['ev']:.2f})")
        elif c.get("no_legal"):
            parts.append("— no legal hit found anywhere")
        else:
            parts.append(f"— no pot exists ({c.get('lanes_probed', 0)} "
                         f"lanes probed incl. banks/kicks/combos)")
    alts = [o for o in c.get("contest", [])[1:3]]
    s = " ".join(p for p in parts if p)
    if alts:
        s += "; next best: " + ", ".join(_opt(o) for o in alts)
    res = rec["result"]
    if res == "win":
        out = f"8→{pm['8']} — {who} WINS"
    elif res == "loss":
        out = f"LOSS of game ({rec['reason']})"
    elif res == "foul":
        out = f"FOUL ({rec['reason']}) — ball in hand"
    elif c["kind"] == "pot":
        extra = ", ".join(f"{b}→{p}" for b, p in pm.items() if b != c["ball"])
        out = (f"MADE{' (also ' + extra + ')' if extra else ''}"
               if rec["made"] else
               f"MISSED (aim error {rec['aim_error']:+.2f}°)"
               + (f", {down} dropped" if pm else ""))
    else:
        out = ("legal safety" if c.get("best_pot") else "legal hit") + (
            f", {down} dropped" if pm else "")
    # say why the table did or didn't change hands when balls dropped
    if res == "continue" and not rec["made"]:
        out += f" — own ball down, {who} keeps shooting"
    elif res == "turnover" and pm:
        if c["kind"] == "safe":
            out += (f" — no pocket was called (playing safe), so the ball "
                    f"stays down but {who}'s turn ends (call your shot)")
        else:
            out += (f" — not the called ball/pocket, so it stays down but "
                    f"{who}'s turn ends (call your shot)")
    s += f". → {out}."
    if rec.get("groups_assigned"):
        g = rec["groups_assigned"][who]
        s += f" {who} is {'solids' if g[0] == '1' else 'stripes'}."
    if rec.get("next_look") and res in ("turnover",):
        p, b, pk = rec["next_look"]
        s += (f" Opponent's best look: {b}→{pk} ~{p:.0%}." if b
              else " Opponent has no direct pot.")
    return s


def to_json(game_rec: dict) -> dict:
    """Game record without sim objects. Positions are kept at FULL float
    precision: renderers re-simulate each shot from the logged state, and a
    cluster/break shot is chaotic — rounding to 0.1 mm sent balls up to
    0.6 m elsewhere, so the video's table drifted from the real game."""
    def rnd(st):
        return {b: {"x": float(v["x"]), "y": float(v["y"]),
                    "pocketed": bool(v.get("pocketed"))}
                for b, v in st["balls"].items()}
    log = []
    for r in game_rec["log"]:
        r2 = {k: v for k, v in r.items()
              if k not in ("shot", "state", "post_state")}
        r2["state"] = {"balls": rnd(r["state"])}
        r2["post_state"] = {"balls": rnd(r["post_state"])}
        log.append(r2)
    return {**{k: v for k, v in game_rec.items() if k != "log"}, "log": log}


def write_items(rec: dict, out_dir: str) -> list[dict]:
    """video_items.json: pre-shot state + the EXECUTED stroke (the sim is
    deterministic, so renderers rebuild each shot exactly)."""
    items = [{"shot_no": r["shot_no"], "player": r["player"],
              "state": r["state"], "call": r["call"],
              "stroke": r["executed"], "result": r["result"],
              "made": r["made"], "pocketed": r["pocketed"],
              "groups": r["groups_before"], "post_state": r["post_state"],
              "narration": r["narration"]} for r in rec["log"]]
    with open(os.path.join(out_dir, "video_items.json"), "w") as f:
        json.dump(items, f)
    return items


def verify_replay(items: list[dict], tol: float = 1e-9) -> list[int]:
    """Shots whose re-simulation from the logged pre-state + executed stroke
    doesn't land on the logged post-state (pre-respot / pre-placement)."""
    bad = []
    for it in items:
        shot = P.simulate(it["state"], tuple(it["stroke"]))
        if shot is None:
            continue
        post = P.post_state(shot, it["state"])["balls"]
        want = it["post_state"]["balls"]
        for b, v in post.items():
            w = want.get(b)
            if w is None or (b == "8" and it["call"]["kind"] == "break"):
                continue
            if bool(v["pocketed"]) != bool(w["pocketed"]) or (
                    not v["pocketed"] and math.dist((v["x"], v["y"]),
                                                    (w["x"], w["y"])) > tol):
                bad.append(it["shot_no"])
                break
    return bad


def main():
    out_dir = os.environ.get("POOL_OUT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "replay_out")
    os.makedirs(out_dir, exist_ok=True)
    if "--renarrate" in sys.argv:
        path = os.path.join(out_dir, "replay_log.json")
        rec = json.load(open(path))
        for r in rec["log"]:
            r["narration"] = narrate(r)
            print(f"[{r['shot_no']:02d}] {r['narration']}")
        json.dump(rec, open(path, "w"), indent=1, default=str)
        write_items(rec, out_dir)
        return
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    sigma = next((float(a.split("=")[1]) for a in sys.argv
                  if a.startswith("--sigma=")), None)
    base = P.DRAFT if "--draft" in sys.argv else P.DEFAULT
    cfg = base if sigma is None else P.replace(
        base, name=f"{base.name}_s{sigma}", sigma_phi=sigma)
    import time
    t0 = time.time()

    def on_turn(rec):
        rec["narration"] = narrate(rec)
        if rec["shot"] is not None:
            c = rec["call"]
            if c.get("ball") and c["ball"] in rec["shot"].balls:
                c["obj_path"] = [[round(float(s.rvw[0][0]), 3),
                                  round(float(s.rvw[0][1]), 3)]
                                 for s in rec["shot"].balls[c["ball"]].history.states]
            c["cue_path"] = [[round(x, 3), round(y, 3)]
                             for x, y in cue_path(rec["shot"])]
        print(f"[{rec['shot_no']:02d}] {rec['narration']}", flush=True)

    g = game.play_game({"A": cfg, "B": cfg}, seed=seed, on_turn=on_turn)
    rec = to_json(g)
    with open(os.path.join(out_dir, "replay_log.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    items = write_items(rec, out_dir)
    bad = verify_replay(items)
    print("replay check: every logged shot re-simulates exactly"
          if not bad else f"REPLAY MISMATCH on shots {bad}")
    if os.environ.get("REPLAY_HIFI"):
        hifi_dir = os.path.join(out_dir, "hifi")
        os.makedirs(hifi_dir, exist_ok=True)
        render_hifi_panels([{**it, "shot": r["shot"]} for it, r in
                            zip(items, g["log"])], hifi_dir)
    print(f"\nWinner: {g['winner']} in {g['shots']} shots (seed {seed}) — "
          f"played in {time.time() - t0:.0f}s ({cfg.name})")
    print(f"Log: {out_dir}/replay_log.json")


if __name__ == "__main__":
    main()
