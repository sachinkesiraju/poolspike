"""The agent interface, with a small example agent you can copy.

An agent is a callable `act(turn) -> action` (or an object with `.act`).
match.py builds one per game via a factory `factory(seed=int, seat=str)`:

  python -B match.py py:agent.py:Example ev --seeds 600000 600010

`turn` (what the server sends; see pool8/PROTOCOL.md):
  phase         "break" | "shoot" | "ball_in_hand" | "ball_in_hand_kitchen"
  balls         {"cue"|"1".."15": {"x", "y", "pocketed"}} in metres, origin
                at the bottom-left cushion corner; table 0.9906 x 1.9812 m
  my_group, opp_group   "solids" | "stripes" | None (open table)
  legal_targets ball ids you may contact first (and call)
  shot_no, slot, balls_left

`action`:
  {"stroke": {"phi": deg, "speed": m/s cue-ball speed (0.3-8, break 12),
              "side": a, "vertical": b},      # tip offsets in units of R,
                                              # |a|,|b| <= 0.6, hypot <= 0.7
   "call": {"ball": id, "pocket": "lb"|"lc"|"lt"|"rb"|"rc"|"rt"} or None (safety),
   "cue": {"x", "y"}}                         # only on break / ball in hand
                                              # (break & kitchen: y < 0.4953)

phi: 0 = +x, 90 = +y (toward the foot rail). The engine adds the table's
execution noise to every stroke; only the called ball in the called pocket
keeps the turn. An invalid action is replaced by the server's fallback shot
and counted as invalid.

Agents may use the simulator (pool8/policy.py: P.simulate, P.summarize,
P.adjudicate, P.noisy with P.DEFAULT) to plan; the reference agents do.
"""
from __future__ import annotations

import math


class Example:
    """Easiest straight-in pot by geometry, no simulation (about the level
    of the bundled `greedy` baseline). A template, not a strong player."""

    def __init__(self, seed=0, seat="A"):
        self.seed = seed

    def act(self, turn):
        import policy as P
        from pool_env import ghost_ball_aim, phi_towards
        from rl import encode as E
        st = E.to_state(turn)
        phase = turn["phase"]
        me = E.GROUPS[turn["my_group"]]
        if phase == "break":
            c = (P.TX / 2, P.HEAD_STRING - 0.05)
            apex = min(P.live(st), key=lambda b: st["balls"][b]["y"])
            A = (st["balls"][apex]["x"], st["balls"][apex]["y"])
            return {"stroke": {"phi": phi_towards(c, A) % 360, "speed": 10.0, "side": 0.0, "vertical": 0.0},
                    "call": None, "cue": {"x": c[0], "y": c[1]}}
        cue = None
        if phase.startswith("ball_in_hand"):
            kitchen = phase == "ball_in_hand_kitchen"
            spots = [(x, y) for x in (0.2, 0.5, 0.8) for y in ((0.15, 0.3, 0.45) if kitchen else (0.3, 0.7, 1.0, 1.3, 1.7))]
            spots = [s for s in spots if P.placement_ok(st, s, kitchen)]
            best = max(spots, key=lambda s: max((P.shot_prob(P.with_cue(st, s), s, b, pk)
                                                 for b in P.targets(st, me) for pk in P.POCKETS), default=0))
            st = P.with_cue(st, best)
            cue = {"x": best[0], "y": best[1]}
        c = P.cue_xy(st)
        opts = [(P.shot_prob(st, c, b, pk), b, pk) for b in P.targets(st, me) for pk in P.POCKETS]
        p, b, pk = max(opts)
        B = (st["balls"][b]["x"], st["balls"][b]["y"])
        if p <= 0:   # nothing to pot: just hit the nearest legal ball
            b = min(P.targets(st, me), key=lambda k: math.dist(c, (st["balls"][k]["x"], st["balls"][k]["y"])))
            B = (st["balls"][b]["x"], st["balls"][b]["y"])
            act = {"stroke": {"phi": phi_towards(c, B) % 360, "speed": 2.0, "side": 0.0, "vertical": 0.0},
                   "call": None}
        else:
            G = ghost_ball_aim(B, P.POCKETS[pk])
            act = {"stroke": {"phi": phi_towards(c, G) % 360, "speed": 2.0, "side": 0.0, "vertical": 0.0},
                   "call": {"ball": b, "pocket": pk}}
        if cue:
            act["cue"] = cue
        return act
