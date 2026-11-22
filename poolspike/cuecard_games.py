"""CueCard full-game sanity: plays CueCard (agent) vs OPP on a few seeds and
reports per-decision CPU time, how often each mechanism fires, rule outcomes
and any exceptions (CueCard errors are NOT masked here, unlike PoolEnv).

  cuecard_games.py OPP LO HI WORKERS [SPEC] [OUT.jsonl]
"""
import collections as C
import json
import multiprocessing as mp
import os
import statistics as S
import sys
import time
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["POOL_WORKERS"] = "1"


def one(job):
    opp, seed, seat, spec = job
    import noise_models
    noise_models.apply_from_env()
    import cuecard
    import judge_net
    from rl import env as ENV
    cc = cuecard.CueCard(spec, seed=zlib.crc32(f"{seed}{seat}".encode()))
    times, kinds, errs = [], C.Counter(), []

    def act(turn):
        t = time.process_time()
        try:
            w = cc.act(turn)
        except Exception:
            errs.append(traceback.format_exc()[-600:])
            raise
        times.append((turn["phase"], time.process_time() - t))
        kinds[(w.get("meta") or {}).get("kind")] += 1
        return w
    env = ENV.PoolEnv(opponent=judge_net.opponent(opp, seed, seat, 0.0), agent_seat=seat)
    obs = env.reset(seed)
    while obs is not None:
        obs, _, _, _ = env.step(act(obs))
    hs = [h for h in env.history if h["seat"] == env.seat]
    pots = [h for h in hs if h["kind"] == "pot"]
    return {"ccseed": cc.seed, "shots_detail": [(h["shot_no"], h["kind"], bool(h["made"]), h["result"]) for h in hs],
            "seed": seed, "seat": seat, "score": (env.outcome() + 1) / 2, "end": env.m.end_reason,
            "times": times, "kinds": dict(kinds), "stats": cc.stats, "errs": errs,
            "pots": len(pots), "made": sum(bool(h["made"]) for h in pots),
            "fouls": sum(h["result"] == "foul" for h in hs), "shots": len(hs),
            "invalid": sum(h["invalid"] is not None for h in hs),
            "opp_err": sum(1 for h in env.history if h["seat"] != env.seat and h["source"] != "player")}


def main(opp, lo, hi, workers, spec="cc", out=None):
    jobs = [(opp, s, seat, spec) for s in range(int(lo), int(hi)) for seat in "AB"]
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        rs = pool.map(one, jobs, chunksize=1)
    if out:
        with open(out, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    ts = [t for r in rs for ph, t in r["times"] if ph == "shoot"]
    tb = [t for r in rs for ph, t in r["times"] if ph.startswith("ball_in_hand")]
    st = C.Counter()
    for r in rs:
        st.update(r["stats"])
    print(f"{spec} vs {opp}: {len(rs)} games, score {S.mean(r['score'] for r in rs):.3f} [{time.time() - t0:.0f}s]")
    print(f"  make {sum(r['made'] for r in rs)}/{sum(r['pots'] for r in rs)}, fouls {sum(r['fouls'] for r in rs)}"
          f"/{sum(r['shots'] for r in rs)}, invalid {sum(r['invalid'] for r in rs)}, errors "
          f"{sum(len(r['errs']) for r in rs)}, opp fallbacks {sum(r['opp_err'] for r in rs)}")
    if ts:
        q = sorted(ts)
        print(f"  shoot cpu/decision mean {S.mean(ts):.2f}s median {q[len(q) // 2]:.2f} p90 {q[int(.9 * len(q))]:.2f}"
              f" max {q[-1]:.2f} (n={len(ts)})")
    if tb:
        print(f"  ball-in-hand cpu/decision mean {S.mean(tb):.2f}s max {max(tb):.2f} (n={len(tb)})")
    print("  stats", dict(st))
    print("  ends", dict(C.Counter(("W " if r["score"] > .5 else "L ") + str(r["end"]) for r in rs)))
    for r in rs:
        for e in r["errs"][:1]:
            print("ERROR seed", r["seed"], r["seat"], e)


if __name__ == "__main__":
    main(*sys.argv[1:])
