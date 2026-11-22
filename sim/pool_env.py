"""Pool position-play env core: table state -> shot call -> sim -> grade.

Coordinates are meters on the playing surface, origin at the bottom-left
cushion corner (pooltool's internal frame). The agent never sees internal
frames: it receives/returns normalized JSON only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import attrs
import numpy as np
import pooltool as pt
from pooltool.objects.ball.datatypes import BallOrientation

BALL_R = pt.BallParams().R

POCKET_NAMES = {
    "lb": "bottom-left corner",
    "lc": "left side",
    "lt": "top-left corner",
    "rb": "bottom-right corner",
    "rc": "right side",
    "rt": "top-right corner",
}


def build_table(game_type: pt.GameType = pt.GameType.NINEBALL) -> pt.Table:
    return pt.Table.from_game_type(game_type)


def serialize_state(table: pt.Table, balls: dict[str, pt.Ball]) -> dict:
    """Table state as LLM-facing JSON."""
    return {
        "table": {"width_m": float(table.w), "length_m": float(table.l)},
        "pockets": {
            pid: {"name": POCKET_NAMES[pid], "x": float(p.center[0]), "y": float(p.center[1])}
            for pid, p in sorted(table.pockets.items())
        },
        "balls": {
            bid: {
                "x": float(b.state.rvw[0, 0]),
                "y": float(b.state.rvw[0, 1]),
                "pocketed": b.state.s == pt.constants.pocketed,
            }
            for bid, b in sorted(balls.items())
        },
    }


def build_system(state: dict) -> pt.System:
    """Reconstruct a System from serialized state JSON."""
    table = build_table()
    # Pin identity orientation: random default can hide a stripe's band,
    # making it render white — reads as a second cue ball.
    IDENT = BallOrientation(pos=(1, 0, 0, 0), sphere=(1, 0, 0, 0))
    balls = {
        bid: attrs.evolve(
            pt.Ball.create(bid, xy=(b["x"], b["y"])),
            initial_orientation=IDENT,
        )
        for bid, b in state["balls"].items()
        if not b.get("pocketed", False)
    }
    cue = pt.Cue(cue_ball_id="cue")
    system = pt.System(cue=cue, table=table, balls=balls)
    # Attach the pocket ballset so rendered images show real ball colors.
    system.set_ballset(pt.layouts.DEFAULT_STANDARD_BALLSET)
    return system


def ghost_ball_aim(object_xy: tuple[float, float], pocket_xy: tuple[float, float]) -> tuple[float, float]:
    """Point the cue ball should travel through: 2R behind the object ball."""
    obj = np.asarray(object_xy, dtype=float)
    direction = obj - np.asarray(pocket_xy, dtype=float)
    direction /= np.linalg.norm(direction)
    aim = obj + direction * (2 * BALL_R)
    return float(aim[0]), float(aim[1])


def phi_towards(cue_xy: tuple[float, float], aim_xy: tuple[float, float]) -> float:
    """pooltool phi (degrees): 0=+x (right), 90=+y (toward foot rail)."""
    d = np.asarray(aim_xy) - np.asarray(cue_xy)
    return math.degrees(math.atan2(d[1], d[0]))


@dataclass
class Grade:
    pocketed_as_called: bool
    cue_in_zone: bool
    legal: bool
    scratch: bool
    pocketed: dict[str, str]  # ball id -> pocket id
    cue_rest: tuple[float, float]
    events_summary: list[str]

    @property
    def passed(self) -> bool:
        return self.pocketed_as_called and self.cue_in_zone and self.legal


def check_legal(shot: pt.System, pocketed: dict[str, str]) -> tuple[bool, bool]:
    """9-ball legality from the event stream.

    Legal = cue's first object-ball contact is the lowest live ball AND after
    contact a ball reaches a rail or a pocket. Scratch = cue ball pocketed.
    Returns (legal, scratch).
    """
    scratch = "cue" in pocketed
    first_contact = None
    rail_after = bool(pocketed)
    for e in shot.events:
        et = e.event_type
        if et == pt.EventType.BALL_BALL and first_contact is None:
            first_contact = next(a.id for a in e.agents if a.id != "cue")
        elif et in (
            pt.EventType.BALL_LINEAR_CUSHION,
            pt.EventType.BALL_CIRCULAR_CUSHION,
        ) and first_contact is not None:
            rail_after = True
    live_balls = [bid for bid, b in shot.balls.items()
                  if bid != "cue" and b.state.s != pt.constants.pocketed]
    live_balls += [bid for bid in pocketed if bid != "cue"]
    lowest = min(live_balls) if live_balls else None
    legal = (first_contact == lowest) and rail_after and not scratch
    return legal, scratch


def pocket_map(shot: pt.System) -> dict[str, str]:
    """ball id -> pocket id for every ball_pocket event."""
    out: dict[str, str] = {}
    for e in shot.events:
        if e.event_type != pt.EventType.BALL_POCKET:
            continue
        ball, pocket = e.agents
        out[ball.id] = pocket.id
    return out


def zone_contains(zone: dict, xy: tuple[float, float]) -> bool:
    """Zone shapes: circle {center:[x,y],radius} or rect {min:[x,y],max:[x,y]}."""
    if "center" in zone:
        return math.dist(zone["center"], xy) <= zone["radius"]
    mn, mx = zone["min"], zone["max"]
    return mn[0] <= xy[0] <= mx[0] and mn[1] <= xy[1] <= mx[1]


def run_called_shot(state: dict, call: dict) -> Grade:
    """Simulate one shot call and grade it.

    call = {
      "target_ball": "1",
      "pocket": "rt",
      "cue_zone": {"center": [x, y], "radius": r},   # where cue must stop
      # shot spec — either aim mode:
      "aim": {"aim_point": [x, y]}                    # cue ball travels through point
            | {"ghost_ball": true}                    # auto: contact point for the called pocket
      "speed_mps": 2.5,                                # cue impact speed
      "english": {"side": -1..1, "vertical": -1..1},   # optional
      "cue_elevation_deg": 0                           # optional masse
    }
    """
    system = build_system(state)
    cue_xy = system.balls["cue"].state.rvw[0, :2]
    aim = call["aim"]
    if aim.get("ghost_ball"):
        obj = state["balls"][call["target_ball"]]
        pocket = state["pockets"][call["pocket"]]
        aim_xy = ghost_ball_aim((obj["x"], obj["y"]), (pocket["x"], pocket["y"]))
    else:
        aim_xy = aim["aim_point"]
    english = call.get("english", {})
    system.cue.set_state(
        V0=float(call.get("speed_mps", 2.5)),
        phi=phi_towards(cue_xy, aim_xy),
        theta=float(call.get("cue_elevation_deg", 0.0)),
        a=float(english.get("side", 0.0)),
        b=float(english.get("vertical", 0.0)),
    )

    shot = pt.simulate(system)
    pocketed = pocket_map(shot)
    legal, scratch = check_legal(shot, pocketed)
    cue_rest = tuple(float(v) for v in shot.balls["cue"].state.rvw[0, :2])
    pocketed_as_called = pocketed.get(call["target_ball"]) == call["pocket"]
    cue_in_zone = zone_contains(call["cue_zone"], cue_rest)
    summary = [
        f"{e.time:.2f}s {e.event_type.value} "
        + ",".join(a.id for a in e.agents)
        for e in shot.events
    ]
    return Grade(pocketed_as_called, cue_in_zone, legal, scratch,
                 pocketed, cue_rest, summary)


# ---------- phase 2: sequences, safeties, competitive state ----------


def state_after(shot: pt.System) -> dict:
    """Serialize a post-shot system back to state-dict ball positions."""
    return {
        "balls": {
            bid: {
                "x": float(b.state.rvw[0, 0]),
                "y": float(b.state.rvw[0, 1]),
                "pocketed": b.state.s == pt.constants.pocketed,
            }
            for bid, b in shot.balls.items()
        }
    }


def first_contact(shot: pt.System) -> str | None:
    """Id of the first non-cue ball the cue contacts."""
    for e in shot.events:
        if e.event_type == pt.EventType.BALL_BALL:
            return next(a.id for a in e.agents if a.id != "cue")
    return None


def rail_after_contact(shot: pt.System) -> bool:
    seen_contact = False
    for e in shot.events:
        if e.event_type == pt.EventType.BALL_BALL:
            seen_contact = True
        elif seen_contact and e.event_type in (
            pt.EventType.BALL_LINEAR_CUSHION,
            pt.EventType.BALL_CIRCULAR_CUSHION,
        ):
            return True
    return False


def legal_groups(state: dict, my_group: list[str], table_open: bool) -> set[str]:
    """8-ball: balls the cue may contact first.

    Open table: anything but the 8. Closed: own group, or the 8 once cleared.
    """
    live = {
        bid for bid, b in state["balls"].items()
        if bid != "cue" and not b.get("pocketed", False)
    }
    if table_open:
        return live - {"8"}
    mine = live & set(my_group)
    return mine if mine else ({"8"} & live)


def check_legal_groups(
    shot: pt.System, pocketed: dict[str, str], legal_first: set[str]
) -> tuple[bool, bool]:
    """Legality for group games (8-ball). Returns (legal, scratch)."""
    scratch = "cue" in pocketed
    fc = first_contact(shot)
    rail_ok = rail_after_contact(shot) or bool(pocketed)
    return (fc in legal_first) and rail_ok and not scratch, scratch


def run_plan_shot(state: dict, call: dict, legal_first: set[str]) -> tuple[dict, dict]:
    """Simulate one shot of a plan. Returns (shot_grade, next_state).

    call.type:
      "pot"  — target_ball must fall in the called pocket; cue in cue_zone.
      "safe" — legal first contact on target_ball (or any legal ball if
               target_ball omitted); cue must rest in cue_zone; no pot needed.
    """
    system = build_system(state)
    cue_xy = system.balls["cue"].state.rvw[0, :2]

    # Resolve the called target. Ball ids can come straight from the call, or
    # the agent may reference a *position* ("target_xy") when its own
    # perception can't separate same-colour balls; the env maps it to the
    # nearest real ball so a label mix-up can't fail a correct intent.
    target = call.get("target_ball")
    resolved_from = None
    if call.get("target_xy"):
        tx, ty = call["target_xy"]
        resolved = min(
            state["balls"],
            key=lambda b: math.dist(
                (state["balls"][b]["x"], state["balls"][b]["y"]), (tx, ty)
            ),
        )
        if math.dist(
            (state["balls"][resolved]["x"], state["balls"][resolved]["y"]),
            (tx, ty),
        ) < 0.08:
            resolved_from = target
            target = resolved
    call["target_ball"] = target

    aim = call.get("aim", {"ghost_ball": True})
    if aim.get("ghost_ball"):
        obj = state["balls"][target]
        pocket = state["pockets"][call["pocket"]]
        aim_xy = ghost_ball_aim((obj["x"], obj["y"]), (pocket["x"], pocket["y"]))
    else:
        aim_xy = aim["aim_point"]
    english = call.get("english", {})
    system.cue.set_state(
        V0=float(call.get("speed_mps", 2.5)),
        phi=phi_towards(cue_xy, aim_xy),
        theta=float(call.get("cue_elevation_deg", 0.0)),
        a=float(english.get("side", 0.0)),
        b=float(english.get("vertical", 0.0)),
    )
    shot = pt.simulate(system)
    pocketed = pocket_map(shot)
    legal, scratch = check_legal_groups(shot, pocketed, legal_first)
    cue_rest = tuple(float(v) for v in shot.balls["cue"].state.rvw[0, :2])
    next_state = state_after(shot)
    next_state["pockets"] = state["pockets"]

    # Collision ledger: everything the sim says changed this shot. The plan
    # may declare expected cue contacts ("cue_contacts": [ids]); the grade
    # checks declared vs actual so sloppy lines are measured, not assumed.
    cue_contacts = [
        next(a.id for a in e.agents if a.id != "cue")
        for e in shot.events
        if getattr(e, "event_type", "") == "ball_ball"
        and "cue" in [a.id for a in e.agents]
    ]
    cue_rails = sum(
        "cushion" in getattr(e, "event_type", "")
        and "cue" in [a.id for a in e.agents]
        for e in shot.events
    )
    moved = {
        bid: (b["x"], b["y"])
        for bid, b in next_state["balls"].items()
        if bid in state["balls"]
        and math.dist((b["x"], b["y"]),
                      (state["balls"][bid]["x"], state["balls"][bid]["y"])) > 0.005
    }

    kind = call.get("type", "pot")
    grade = {
        "type": kind,
        "target_ball": target,
        "pocket": call.get("pocket"),
        "legal": legal,
        "scratch": scratch,
        "first_contact": first_contact(shot),
        "pocketed": pocketed,
        "cue_rest": cue_rest,
        "cue_contacts": cue_contacts,
        "cue_rails": cue_rails,
        "moved_balls": moved,
    }
    if resolved_from is not None and resolved_from != target:
        grade["target_resolved_from"] = resolved_from
    if "cue_contacts" in call:
        grade["contacts_declared"] = list(call["cue_contacts"])
        grade["contacts_predicted"] = set(call["cue_contacts"]) == set(cue_contacts)
    if call.get("cue_zone"):
        zc = call["cue_zone"]["center"]
        grade["cue_target"] = tuple(zc)
        grade["cue_rest_err_cm"] = round(
            100 * math.dist(zc, cue_rest), 1
        )
    if kind == "pot":
        grade["pocketed_as_called"] = pocketed.get(target) == call["pocket"]
        grade["cue_in_zone"] = zone_contains(call["cue_zone"], cue_rest)
        grade["verified"] = grade["pocketed_as_called"] and grade["cue_in_zone"] and legal
    else:  # safe
        grade["hit_declared"] = target is None or grade["first_contact"] == target
        grade["cue_in_zone"] = "cue_zone" not in call or zone_contains(
            call["cue_zone"], cue_rest
        )
        if "object_zone" in call:
            ob = next_state["balls"].get(declared, {})
            grade["object_in_zone"] = ob.get("pocketed", False) or zone_contains(
                call["object_zone"], (ob["x"], ob["y"])
            )
        else:
            grade["object_in_zone"] = True
        grade["verified"] = (
            legal and grade["hit_declared"] and grade["cue_in_zone"]
            and grade["object_in_zone"]
        )
    return grade, next_state


def segment_clear(
    p1: tuple[float, float],
    p2: tuple[float, float],
    blockers: dict[str, tuple[float, float]],
    margin: float = 0.0,
) -> bool:
    """True if no blocker center is within 2R of the p1->p2 segment.

    `margin` shrinks the checked segment at both ends so a blocker touching the
    endpoints doesn't count (the target itself is already excluded).
    """
    a, b = np.asarray(p1, float), np.asarray(p2, float)
    seg = b - a
    L = np.linalg.norm(seg)
    u = seg / L
    lo, hi = margin, L - margin
    for xy in blockers.values():
        c = np.asarray(xy, float)
        t = np.clip(np.dot(c - a, u), lo, hi)
        if np.linalg.norm(a + t * u - c) < 2 * BALL_R:
            return False
    return True


def ball_options(
    state: dict, shooter: str = "cue", group_ids: list[str] | None = None,
) -> dict[str, dict]:
    """For each live ball in group_ids: can the cue see it, which pockets are open.

    Returns {ball_id: {"cue_los": bool, "open_pockets": [pocket_id, ...]}}.
    """
    balls = {
        bid: (b["x"], b["y"])
        for bid, b in state["balls"].items()
        if not b.get("pocketed", False)
    }
    cue_xy = balls[shooter]
    live = group_ids if group_ids is not None else [k for k in balls if k != shooter]
    out: dict[str, dict] = {}
    for bid in live:
        if bid not in balls:
            continue
        others = {k: v for k, v in balls.items() if k not in (shooter, bid)}
        cue_los = segment_clear(cue_xy, balls[bid], others, margin=BALL_R)
        open_pockets = [
            pid for pid, p in state["pockets"].items()
            if segment_clear(balls[bid], (p["x"], p["y"]), others, margin=BALL_R)
        ]
        out[bid] = {"cue_los": cue_los, "open_pockets": open_pockets}
    return out


def grade_plan(state: dict, plan: dict, ctx: dict) -> dict:
    """Grade a declared plan by simulating each shot in sequence.

    plan = {"plan": "runout"|"safety", "shots": [call, ...]}
    ctx  = {"my_group": [...], "opp_group": [...], "table_open": bool}

    Every declared shot is simulated from the state the previous shot actually
    produced, so each "pot" is fully sim-verified (pocket + position + legal).
    A "safety" plan additionally gets `safety_verdict`: whether the opponent is
    left snookered or with no open pot on their resulting position.
    """
    cur = state
    shot_grades = []
    for call in plan.get("shots", []):
        legal_first = legal_groups(cur, ctx.get("my_group", []),
                                   ctx.get("table_open", False))
        grade, cur = run_plan_shot(cur, call, legal_first)
        shot_grades.append(grade)

    live_mine = [
        bid for bid, b in cur["balls"].items()
        if bid in ctx.get("my_group", []) and not b.get("pocketed", False)
    ]
    result = {
        "plan": plan.get("plan"),
        "shots": shot_grades,
        "shots_verified": sum(1 for g in shot_grades if g["verified"]),
        "shots_declared": len(plan.get("shots", [])),
        "my_balls_remaining": live_mine,
        "runout_complete": not live_mine,
    }
    if plan.get("plan") == "safety":
        if cur["balls"].get("cue", {}).get("pocketed"):
            result["safety_verdict"] = {
                "snookered": False, "no_direct_pot": False,
                "opponent_options": {}, "worked": False, "scratch": True,
            }
        else:
            opts = ball_options(cur, "cue", ctx.get("opp_group", []))
            snookered = not any(o["cue_los"] for o in opts.values())
            no_pot = not any(o["cue_los"] and o["open_pockets"] for o in opts.values())
            result["safety_verdict"] = {
                "snookered": snookered,
                "no_direct_pot": no_pot,
                "opponent_options": opts,
                "worked": snookered or no_pot,
                "scratch": False,
            }
    result["passed"] = bool(shot_grades) and all(
        g["verified"] for g in shot_grades
    ) and (
        result.get("safety_verdict", {}).get("worked", True)
    )
    return result
