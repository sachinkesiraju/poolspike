"""Independent audit of a played game (replay_out/replay_log.json).

For every non-break turn, brute-force each legal ball × pocket with a dense
aim sweep (±3° around the ghost-ball line, 0.15° steps, three speeds) —
independent of the policy's lane generator — and flag:

  MISSED_LANE  a direct pot the brute force finds (basin ≥ 0.3°) that the
               policy never verified, that really makes ≥ 60% when executed
               40× under the game's noise, and that beats the shot taken
  DECLINED     the policy played safe although the brute force has a pot of
               ≥ 85% — printed with both EVs so the call can be judged
  THIN_TAKE    the chosen pot is ≥ 30 points less makeable than the best
               brute-force pot — printed with EVs (position may justify it)

Usage: python audit_policy.py [replay_out/replay_log.json]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))  # game engine, policy
import policy as P  # noqa: E402

STEP, SPAN, SPEEDS = 0.15, 3.0, (1.3, 1.8, 3.0)


def _lane_sweep(job):
    """Longest contiguous potting run (deg) for one ball→pocket lane, with
    the stroke at the centre of that run."""
    state, ball, pk = job
    b = state["balls"][ball]
    c = P.cue_xy(state)
    base = P.phi_towards(c, P.ghost_ball_aim((b["x"], b["y"]), P.POCKETS[pk]))
    best, stroke = 0.0, None
    n = int(SPAN / STEP)
    for v in SPEEDS:
        run = 0
        for k in range(-n, n + 1):
            res = P.summarize(P.simulate(state, (base + k * STEP, v, 0.0, 0.0)),
                              state)
            if P._pots_called(res, ball, pk, ball):
                run += 1
                if run * STEP > best:
                    best = run * STEP
                    stroke = (base + (k - (run - 1) / 2) * STEP, v, 0.0, 0.0)
            else:
                run = 0
    return best, stroke


def _true_make(job):
    """Make-rate of a stroke under the player's full execution noise."""
    state, ball, pk, stroke, cfg, seed = job
    import random
    rng = random.Random(seed)
    return sum(P._pots_called(P.summarize(P.simulate(state, P.noisy(stroke, rng, cfg)),
                                          state), ball, pk, ball)
               for _ in range(40)) / 40


def audit(log: list, cfg: P.Config = P.DEFAULT) -> list[str]:
    """Flags per turn; a never-verified lane only counts if it really makes
    ≥ 60% under `cfg` noise (fragmented kiss/rattle basins don't)."""
    flags = []
    for rec in log:
        c = rec["call"]
        if c["kind"] == "break":
            continue
        state = rec["state"]
        me = rec["groups_before"][rec["player"]]
        tg = sorted(P.targets(state, me))
        jobs = [(state, b, pk) for b in tg for pk in P.POCKETS]
        sweeps = P.pmap(_lane_sweep, jobs)
        strokes = {(b, pk): st for (_, b, pk), (_, st) in zip(jobs, sweeps)}
        lanes = sorted(((bas, b, pk) for (_, b, pk), (bas, _) in zip(jobs, sweeps)
                        if bas > 0), reverse=True)
        # match on ball→pocket: a "direct" lane that touches a cushion first
        # is relabelled "X→pk bank rail-first" by the policy
        found = {lab.split(" ")[0] for lab in c.get("lanes_found", [])}
        n = rec["shot_no"]
        best_bf = lanes[0] if lanes else None
        unseen = [(bas, b, pk) for bas, b, pk in lanes
                  if bas >= 0.3 and f"{b}→{pk}" not in found]
        trues = P.pmap(_true_make, [(state, b, pk, strokes[(b, pk)], cfg, n)
                                    for _, b, pk in unseen])
        for (bas, b, pk), pt_ in zip(unseen, trues):
            # same yardstick as the policy: a lane is "direct" only if the
            # ball drops without kissing or banking (else it is a deep route)
            info = P.classify_route(P.simulate(state, strokes[(b, pk)]), b, pk)
            if info["kisses"] or info["rails"]:
                pt_ *= P.route_factor(info, cfg)
            chosen_p = c.get("p_make", 0.0) if c["kind"] == "pot" else 0.0
            if pt_ >= 0.6 and pt_ > chosen_p + 0.05:
                flags.append(f"[{n:02d}] MISSED_LANE {b}→{pk}: brute basin "
                             f"{bas:.2f}°, makes {pt_:.0%} under noise, never "
                             f"verified (chose {chosen_p:.0%})")
            elif pt_ >= 0.6:
                print(f"     (unverified {b}→{pk} ~{pt_:.0%}, but a "
                      f"{chosen_p:.0%} shot was taken instead)")
            else:
                print(f"     (ignored {b}→{pk}: {bas:.2f}° basin but only "
                      f"{pt_:.0%} under noise — fragmented)")
        if best_bf:
            # same yardstick as the policy: full execution noise (aim, speed,
            # spin), not aim alone — spin error alone can cost 10–30 points
            pb = _true_make((state, best_bf[1], best_bf[2],
                             strokes[(best_bf[1], best_bf[2])], cfg, n))
            lab = f"{best_bf[1]}→{best_bf[2]}"
            # same yardstick as the policy: kiss/bank routes carry real-table risk
            info = P.classify_route(P.simulate(state, strokes[(best_bf[1], best_bf[2])]),
                                    best_bf[1], best_bf[2])
            pb *= P.route_factor(info, cfg)
            if c["kind"] == "safe" and pb >= 0.85:
                bp = c.get("best_pot") or {}
                flags.append(f"[{n:02d}] DECLINED {lab} (~{pb:.0%} brute): "
                             f"safety EV {c['ev']:.2f} vs best pot EV "
                             f"{bp.get('ev', float('nan')):.2f}")
            if c["kind"] == "pot" and pb - c["p_make"] >= 0.30:
                alt = next((o for o in c.get("contest", [])
                            if o.get("label") == lab), None)
                flags.append(f"[{n:02d}] THIN_TAKE {c['ball']}→{c['pocket']} "
                             f"{c['p_make']:.0%} (EV {c['ev']:.2f}) over {lab} "
                             f"~{pb:.0%}" + (f" (EV {alt['ev']:.2f})" if alt
                                             else " (not in top contest)"))
        print(f"[{n:02d}] {rec['player']} {c['kind']:<4} "
              f"{c.get('ball')}→{c.get('pocket')}  brute lanes: "
              + ", ".join(f"{b}→{pk} {bas:.2f}°" for bas, b, pk in lanes[:5]),
              flush=True)
    return flags


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "replay_out",
        "replay_log.json")
    rec = json.load(open(path))
    log = rec["log"]
    # audit at the noise level the game was played with (cfg "ev_s0.18")
    name = log[0].get("cfg", "")
    sigma = float(name.rsplit("_s", 1)[1]) if "_s" in name else P.DEFAULT.sigma_phi
    flags = audit(log, P.replace(P.DEFAULT, sigma_phi=sigma))
    print("\n" + ("\n".join(flags) if flags else "no flags"))


if __name__ == "__main__":
    main()
