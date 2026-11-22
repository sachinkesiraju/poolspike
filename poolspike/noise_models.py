"""Alternative execution-noise models and rule sets, switched on by env vars.

Nothing happens unless one of these is set, so the default game (our noise,
our call-shot rules) is untouched:

  POOL_NOISE=icga2008   2008 ICGA computational-pool noise (Archibald et al.,
                        AI Magazine 2010; same values as the 2006 Olympiad
                        model E_Italy in Smith's PickPocket thesis, Sec. 4.5):
                        zero-mean Gaussian, sd phi 0.125 deg, V 0.075 m/s,
                        a = b = 0.5 mm, theta 0.1 deg.
  POOL_RULESET=icga     BCA-style open table after the break (the ICGA rule):
                        a ball pocketed on the break keeps the turn but does
                        not assign groups.

`apply_from_env()` must run in every process before the game engine and the
agents are built (vnet2.play_ and bench/match.py call it). It mutates the
shared `policy.DEFAULT` config in place (every agent and the engine read their
noise from it or from copies made later) and swaps `policy.noisy`, so the
engine's execution noise and every agent's internal noise model agree.

Mapping onto our stroke space (phi, V, a, b; level cue):
  * phi: 0.125 deg absolute, no extra error for pace or english
    (our default grows aim error with speed and side spin).
  * V: 0.075 m/s absolute. ICGA perturbs the cue-stick speed of POOLFIZ;
    our V is the cue-ball speed, so this is applied to the cue-ball speed.
  * a, b: 0.5 mm / R (28.575 mm) = 0.0175 tip-offset units.
  * theta: not modelled; our engine always strikes with a level cue.
"""

from __future__ import annotations

import os
import sys

_COWORLD = os.environ.get("POOL_COWORLD", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sim"))
if os.path.isdir(_COWORLD) and _COWORLD not in sys.path:
    sys.path.insert(0, _COWORLD)

ICGA_SIGMA_PHI = 0.125     # deg  (AI Magazine 2010; thesis E_Italy)
ICGA_SIGMA_V = 0.075       # m/s  absolute
ICGA_SIGMA_AB_MM = 0.5     # mm
ICGA_SIGMA_THETA = 0.1     # deg  (not applied: level cue)

ACTIVE = {"noise": "ours", "rules": "ours"}


def _icga_noise(P):
    sab = ICGA_SIGMA_AB_MM / 1000.0 / P.BALL_R
    fields = dict(sigma_phi=ICGA_SIGMA_PHI, sigma_ab=sab,
                  sigma_v=ICGA_SIGMA_V / 2.5,   # for any relative-speed use
                  speed_gain=0.0, english_gain=0.0)
    cfgs = {id(P.DEFAULT): P.DEFAULT}
    for c in list(getattr(P, "PRESETS", {}).values()) + [P.DRAFT, P.LITE]:
        cfgs[id(c)] = c
    for c in cfgs.values():
        for k, v in fields.items():
            object.__setattr__(c, k, v)

    def noisy(stroke, rng, cfg=P.DEFAULT):
        phi, v, a, b = stroke
        return (phi + rng.gauss(0, cfg.sigma_phi),
                max(0.3, v + rng.gauss(0, ICGA_SIGMA_V)),
                max(-0.6, min(0.6, a + rng.gauss(0, cfg.sigma_ab))),
                max(-0.6, min(0.6, b + rng.gauss(0, cfg.sigma_ab))))
    P.noisy = noisy


def _icga_rules(P):
    orig = P.adjudicate

    def adjudicate(state, ctx, call, res):
        j = orig(state, ctx, call, res)
        if call.get("kind") == "break":
            j = dict(j)
            j["me"] = j["opp"] = None      # table stays open after the break
        return j
    P.adjudicate = adjudicate


def apply(noise: str = "ours", rules: str = "ours") -> dict:
    import policy as P
    if noise == "icga2008" and ACTIVE["noise"] != "icga2008":
        _icga_noise(P)
        ACTIVE["noise"] = "icga2008"
    elif noise not in ("ours", "icga2008"):
        raise ValueError(f"unknown POOL_NOISE {noise!r}")
    if rules == "icga" and ACTIVE["rules"] != "icga":
        _icga_rules(P)
        ACTIVE["rules"] = "icga"
    elif rules not in ("ours", "icga"):
        raise ValueError(f"unknown POOL_RULESET {rules!r}")
    return dict(ACTIVE)


def apply_from_env() -> dict:
    return apply(os.environ.get("POOL_NOISE", "ours") or "ours",
                 os.environ.get("POOL_RULESET", "ours") or "ours")
