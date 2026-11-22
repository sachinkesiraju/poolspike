"""Scripted baseline: geometric greedy shot selection, no simulation.

Used three ways: the bundled `greedy` player, the action the server plays
for a seat that times out or keeps sending invalid actions, and the EV
player's instant fallback. Every function returns a legal-shaped action for
every reachable observation, in milliseconds.

Actions use the wire format of the player protocol (PROTOCOL.md):
  {"call": {"ball", "pocket"} | None, "stroke": {"phi", "speed", "side",
   "vertical"}, "cue": {"x", "y"} (break / ball in hand only)}
"""

from __future__ import annotations

import math

import policy as P
from pool_env import ghost_ball_aim, phi_towards

BREAK_SPOT = (P.TX / 2, P.HEAD_STRING - 0.04)


def _stroke(phi: float, speed: float, vertical: float = 0.0) -> dict:
    return {"phi": phi % 360.0, "speed": speed, "side": 0.0,
            "vertical": vertical}


def best_pot(state: dict, group, cfg: P.Config = P.DEFAULT):
    """(p, ball, pocket) of the geometrically easiest legal pot."""
    c = P.cue_xy(state)
    looks = [(P.shot_prob(state, c, b, pk, cfg), b, pk)
             for b in sorted(P.targets(state, group)) for pk in P.POCKETS]
    return max(looks, default=(0.0, None, None))


def break_action(state: dict) -> dict:
    apex = min(P.live(state), key=lambda b: state["balls"][b]["y"])
    A = (state["balls"][apex]["x"], state["balls"][apex]["y"])
    return {"call": None, "cue": {"x": BREAK_SPOT[0], "y": BREAK_SPOT[1]},
            "stroke": _stroke(phi_towards(BREAK_SPOT, A), 10.0, -0.2)}


def place_cue(state: dict, group, kitchen: bool) -> tuple[float, float]:
    """Cue spot behind the easiest ball-pocket line (straight-in, 25 cm)."""
    best = (-1.0, None)
    for ball in sorted(P.targets(state, group)):
        b = state["balls"][ball]
        B = (b["x"], b["y"])
        for pk, Pk in P.POCKETS.items():
            d = math.dist(B, Pk) or 1e-6
            ux, uy = (Pk[0] - B[0]) / d, (Pk[1] - B[1]) / d
            for back in (0.25, 0.15, 0.4):
                C = (B[0] - ux * (2 * P.BALL_R + back),
                     B[1] - uy * (2 * P.BALL_R + back))
                if not P.placement_ok(state, C, kitchen):
                    continue
                p = P.shot_prob(P.with_cue(state, C), C, ball, pk)
                if p > best[0]:
                    best = (p, C)
    if best[1] is not None:
        return best[1]
    return free_spot(state, kitchen)


def free_spot(state: dict, kitchen: bool) -> tuple[float, float]:
    """A legal cue placement, searched on a fine grid (always finds one:
    the kitchen alone has room for hundreds of cue balls)."""
    ys = ([P.HEAD_STRING * k / 12 for k in range(1, 12)] if kitchen
          else [P.TY * k / 24 for k in range(1, 24)])
    xs = [P.TX * k / 16 for k in range(1, 16)]
    xs.sort(key=lambda x: abs(x - P.TX / 2))
    for y in ys:
        for x in xs:
            if P.placement_ok(state, (x, y), kitchen):
                return (x, y)
    raise RuntimeError("no legal cue placement")


def shot_action(state: dict, group) -> dict:
    """Easiest geometric pot with a stun stroke; with nothing open, the
    softest full hit on the nearest legal ball (a legal-hit attempt)."""
    p, ball, pk = best_pot(state, group)
    c = P.cue_xy(state)
    if ball is not None and p > 0.02:
        b = state["balls"][ball]
        phi = phi_towards(c, ghost_ball_aim((b["x"], b["y"]), P.POCKETS[pk]))
        return {"call": {"ball": ball, "pocket": pk},
                "stroke": _stroke(phi, 2.0)}
    tg = P.targets(state, group)
    ball = min(sorted(tg), key=lambda x: math.dist(
        c, (state["balls"][x]["x"], state["balls"][x]["y"])))
    b = state["balls"][ball]
    return {"call": None, "stroke": _stroke(phi_towards(c, (b["x"], b["y"])),
                                            1.6)}


def safe_action(state: dict, phase: str) -> dict:
    """Last resort, valid in every phase: a medium no-call hit up the
    table from a legal spot (never raises on an odd table)."""
    act = {"call": None, "stroke": _stroke(90.0, 2.0)}
    if phase != "shoot":
        C = free_spot(state, kitchen=phase in ("break", "ball_in_hand_kitchen"))
        act["cue"] = {"x": C[0], "y": C[1]}
    return act


def action(obs_state: dict, phase: str, group) -> dict:
    """Baseline action for any phase of the turn protocol."""
    if phase == "break":
        return break_action(obs_state)
    if phase.startswith("ball_in_hand"):
        C = place_cue(obs_state, group, kitchen=phase == "ball_in_hand_kitchen")
        act = shot_action(P.with_cue(obs_state, C), group)
        return {**act, "cue": {"x": C[0], "y": C[1]}}
    return shot_action(obs_state, group)
