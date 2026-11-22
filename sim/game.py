"""Headless full 8-ball game between two policy configs.

`play_game({"A": cfg_a, "B": cfg_b}, seed)` → game record. Rules come from
`policy.adjudicate` — the same function the policy uses to price outcomes,
so the game and the decision-maker can never disagree about what a shot
means. Every stroke is executed with the shooter's own `Config` noise.

Rules (WPA-style, documented simplifications):
  * break: cue placed in the kitchen by the breaker; any ball down = keep
    shooting; 8 on the break is re-spotted; scratch on the break = ball in
    hand behind the head string;
  * open table until a called ball is legally made; the 8 can't be hit first
    on an open table;
  * foul (scratch, wrong ball first, no contact, no rail after contact) =
    opponent ball in hand anywhere;
  * called shots: only the called ball in the called pocket keeps you at
    the table. A safety (or slop) that drops your ball still ends the turn.
    POOL_RULES=standard switches to bar rules (any own ball counts);
  * 8 legally in the called pocket after clearing = win; 8 early, 8 in the
    wrong pocket, or scratch while making the 8 = loss.
  * not modelled: three-foul rule, stalemate re-rack, jump/massé strokes.
"""

from __future__ import annotations

import random

import numpy as np

import policy as P


def rack_state(seed: int) -> dict:
    """Standard 8-ball rack, reproducible per seed.

    Positions come from pooltool's rack (its wiggle is np-seeded), but ids
    are re-assigned here: pooltool picks ids via random.choice over a set,
    whose order follows per-process string hashing, so the same seed gave
    different racks run to run. 8 stays in the centre; the two back corners
    get one solid and one stripe.
    """
    import pooltool as pt
    from pooltool.layouts import get_rack

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    rack = get_rack(pt.GameType.EIGHTBALL, pt.Table.default())
    pos = {bid: (float(b.state.rvw[0, 0]), float(b.state.rvw[0, 1]))
           for bid, b in rack.items()}
    cue, eight = pos.pop("cue"), pos.pop("8")
    spots = sorted(pos.values(), key=lambda p: (p[1], p[0]))
    rng = random.Random(seed)
    ids = [str(i) for i in range(1, 16) if i != 8]
    rng.shuffle(ids)
    back = sorted(range(len(spots)), key=lambda i: -spots[i][1])[:5]
    corners = [min(back, key=lambda i: spots[i][0]),
               max(back, key=lambda i: spots[i][0])]
    a, b = ids[corners[0]], ids[corners[1]]
    if (int(a) < 8) == (int(b) < 8):  # same group in both corners: swap one
        j = next(i for i, x in enumerate(ids)
                 if i not in corners and (int(x) < 8) != (int(a) < 8))
        ids[corners[1]], ids[j] = ids[j], ids[corners[1]]
    balls = {bid: {"x": x, "y": y, "pocketed": False}
             for bid, (x, y) in zip(ids, spots)}
    balls["8"] = {"x": eight[0], "y": eight[1], "pocketed": False}
    balls["cue"] = {"x": cue[0], "y": cue[1], "pocketed": False}
    return {"balls": balls}


def _strip(call: dict) -> dict:
    """JSON-safe copy of a call (no sim objects)."""
    return {k: v for k, v in call.items() if k not in ("shot",)}


def play_game(cfgs: dict, seed: int = 0, max_shots: int = 80,
              on_turn=None) -> dict:
    """Play one game; A breaks. `on_turn(rec)` is called after every shot.

    Each record keeps the pre-shot state (cue at its placement), the call
    with the intended stroke, the executed (noisy) stroke, the adjudication,
    and — in memory only, under "shot" — the simulated pt.System.
    """
    rng = random.Random(seed * 7919 + 1)
    state = rack_state(seed)
    groups = {"A": None, "B": None}
    turn, bih = "A", None
    log = []
    winner = None

    for shot_no in range(max_shots + 1):
        other = "B" if turn == "A" else "A"
        cfg = cfgs[turn]
        ctx = {"me": groups[turn], "opp": groups[other]}
        dseed = seed * 1000 + shot_no
        if shot_no == 0:
            call = P.choose_break(state, cfg, dseed)
        elif bih:
            call = P.choose_bih(state, ctx, cfg, dseed,
                                kitchen=(bih == "kitchen"))
        else:
            call = P.choose(state, ctx, cfg, dseed)
        if call.get("cue_place"):
            state = P.with_cue(state, call["cue_place"])
        executed = P.noisy(call["stroke"], rng, cfg)
        shot = P.simulate(state, executed)
        res = P.summarize(shot, state)
        j = P.adjudicate(state, ctx, call, res)
        made = (call["kind"] == "pot" and res is not None
                and res["pm"].get(call["ball"]) == call["pocket"])
        rec = {
            "shot_no": shot_no, "player": turn, "cfg": cfg.name,
            "groups_before": dict(groups), "bih": bih,
            "state": state, "call": _strip(call),
            "stroke": list(call["stroke"]), "executed": list(executed),
            "aim_error": executed[0] - call["stroke"][0],
            "pocketed": {} if res is None else res["pm"],
            "first_contact": None if res is None else res["fc"],
            "result": j["result"], "legal": j["legal"],
            "reason": j["reason"], "made": made, "shot": shot,
        }
        groups[turn], groups[other] = j["me"], j["opp"]
        if groups[turn] is not None and rec["groups_before"][turn] is None:
            rec["groups_assigned"] = {turn: list(groups[turn])}
        state = j["state"]
        bih = None
        r = j["result"]
        if r == "win":
            winner = turn
        elif r == "loss":
            winner = other
        elif r == "foul":
            bih = "kitchen" if shot_no == 0 else "anywhere"
            turn = other
        elif r == "turnover":
            turn = other
        if winner is None and bih is None:
            # what the next shooter is looking at (geometric best look)
            nxt = {"me": groups[turn], "opp": groups["B" if turn == "A" else "A"]}
            looks = [(P.shot_prob(state, P.cue_xy(state), b, pk, cfgs[turn]), b, pk)
                     for b in P.targets(state, nxt["me"]) for pk in P.POCKETS]
            rec["next_look"] = max(looks, default=(0.0, None, None))
        rec["post_state"] = state
        log.append(rec)
        if on_turn:
            on_turn(rec)
        if winner:
            break
    return {"winner": winner, "shots": len(log), "log": log,
            "cfgs": {k: v.name for k, v in cfgs.items()}, "seed": seed}
