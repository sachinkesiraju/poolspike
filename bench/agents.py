"""Reference agents and the spec -> agent factory used by match.py.

Every agent is a callable `turn -> action` (see agent.py). Seeds follow the
original evaluation harness (poolspike/vnet2.py): the first agent of a match gets
crc32(f"{seed}{seat}"), the second crc32(f"o{seed}{seat}"), so a match here
reproduces the published numbers game for game.

  greedy        bundled baseline: easiest geometric pot, no search
  lite          EV search, budget preset
  ev            noise-aware expected-value search (pool8/policy.py, POLICY.md)
  cc[:mult]     CueCard rebuild (ref/cuecard.py); mult scales its search budget
  poolspike     Poolspike: hand-written candidates + noisy simulations, judged
                by a value net trained by 19 rounds of self-play (ref/shotnet5.py,
                weights ref/nets/poolspike.npz). `r19`, its name in the logs,
                is an alias.
  n5:<npz>      any ShotNet5 weights file
  py:<file.py>:<factory>   your agent: factory(seed=int, seat=str)
"""
from __future__ import annotations

import importlib.util
import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
POOLSPIKE = os.path.join(HERE, "ref", "nets", "poolspike.npz")


def _seed(seed, seat, role):
    return zlib.crc32(f"{'' if role == 'a' else 'o'}{seed}{seat}".encode())


def make(spec: str, seed: int, seat: str, role: str = "a"):
    from rl import env as ENV
    s = _seed(seed, seat, role)
    if spec in ("greedy", "lite", "ev"):
        return ENV.make_policy(spec)
    if spec == "cc" or spec.startswith("cc:"):
        import cuecard
        return cuecard.CueCard(spec, seed=s).act
    if spec in ("poolspike", "r19") or spec.startswith("n5:"):
        import shotnet5
        from rl import encode as E
        me = shotnet5.ShotNet5(spec[3:] if spec.startswith("n5:") else POOLSPIKE, seed=s)

        def act(turn):
            # record the judge's win chance for the position before each shot
            # (for calibration checks); the move itself is unchanged
            ph = turn["phase"]
            if ph != "break" and me.vnet is not None:
                try:
                    act.judge.append((turn["shot_no"], me.vnet(E.to_state(turn), E.GROUPS[turn["my_group"]],
                                                              E.GROUPS[turn["opp_group"]], ph.startswith("ball_in_hand"),
                                                              ph == "ball_in_hand_kitchen")))
                except Exception:
                    pass
            return me.act(turn)
        act.judge = []
        return act
    if spec.startswith("py:"):
        path, _, name = spec[3:].rpartition(":")
        mod_spec = importlib.util.spec_from_file_location(
            "user_agent_" + str(abs(hash(path))), os.path.abspath(path))
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        obj = getattr(mod, name)(seed=s, seat=seat)
        return obj.act if hasattr(obj, "act") else obj
    raise ValueError(f"unknown agent spec {spec!r}")
