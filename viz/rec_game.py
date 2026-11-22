"""Replay one Poolspike-vs-EV game exactly as the eval played it and write the video
pipeline's replay_log.json + video_items.json.

  rec_game.py NET SEED SEAT OUT_DIR
"""
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "poolspike"))  # the player
import poolspike_player as N5  # noqa: E402
from rl import env as ENV  # noqa: E402
import policy as P  # noqa: E402
import replay as R  # noqa: E402

net, seed, seat, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
log = []
me = N5.ShotNet5(net, tau=0.0, seed=zlib.crc32(f"{seed}{seat}".encode()), log=log)
picks = {}


def act(turn):
    """ShotNet5.act verbatim, but keeps the shortlist for the video panel."""
    N5.VNET = me.vnet
    if turn["phase"] == "break":
        return N5.SN.break_wire(me.book, turn)
    opts, st0, mg, ctx = N5.options(turn)
    if not opts:
        return me.fallback(turn, st0, mg, ctx)
    for o in opts:
        o.sample(N5.NOISY0)
    s = me.score(N5.np.array([o.vector() for o in opts]))
    keep = [opts[i] for i in N5.np.argsort(-s)[:N5.TOP]]
    for o in keep:
        o.refine()
        o.sample(N5.NOISY0 + N5.NOISY1)
    X = N5.np.array([o.vector() for o in keep])
    s = me.score(X)
    i = int(N5.np.argmax(s))
    log.append({"X": X, "i": i})
    rank = sorted(range(len(keep)), key=lambda k: -s[k])
    picks[turn.get("shot_no", 0)] = {
        "value": float(X[i][N5.F["value_mean"]]), "make": float(X[i][N5.F["noisy_make"]]),
        "n_options": len(opts),
        "contest": [{"kind": "pot", "label": f"{keep[k].call['ball']}→{keep[k].call['pocket']}"
                     + f" · {keep[k].stroke[1]:.1f} m/s " + ("follow" if keep[k].stroke[3] > 0.1 else "draw" if keep[k].stroke[3] < -0.1 else "center")
                     + (" + side" if abs(keep[k].stroke[2]) > 0.1 else "")
                     + (" (bank/kick)" if X[k][N5.F["deep"]] else ""),
                     "ball": keep[k].call["ball"], "pocket": keep[k].call["pocket"],
                     "p_make": float(X[k][N5.F["noisy_make"]]), "ev": float(X[k][N5.F["value_mean"]])}
                    for k in rank]}
    return keep[i].wire


ev_calls = {}
_ctw = ENV.call_to_wire


def _spy(call):                           # EV's full call (options, plan) for the panel
    ev_calls[len(env.m.log)] = call
    return _ctw(call)


ENV.call_to_wire = _spy
agent = act


env = ENV.PoolEnv(opponent=ENV.make_policy("ev"), agent_seat=seat)
obs = env.reset(seed)
while obs is not None:
    obs, _, _, _ = env.step(agent(obs))
score = (env.outcome() + 1) / 2
m = env.m
who = {seat: "Poolspike", ("B" if seat == "A" else "A"): "EV"}


def groups(g):
    return {s: (sorted(v, key=int) if v else None) for s, v in g.items()}


def narr(r):
    c = r["call"]
    name = who[r["player"]]
    if c["kind"] == "break":
        s = f"{name} breaks"
    elif c["kind"] == "pot":
        s = f"{name} calls {c['ball']} in {c['pocket']}"
    else:
        s = f"{name} plays safe off {c.get('ball', '?')}"
    pk = picks.get(r["shot_no"])
    if r["player"] == seat and pk:
        s += (f" — best of {pk['n_options']} candidate shots; made {pk['make']:.0%} of 24 noisy "
              f"simulations, learned win chance after it {pk['value']:.0%}")
    res = r["result"]
    s += {"continue": " → made, shoots again", "turnover": " → missed, turn passes",
          "foul": " → foul, ball in hand", "win": " → wins the game", "loss": " → loses the game"}.get(res, f" → {res}")
    return s + "."


recs = []
for r in m.log:
    shot = P.simulate(r["state"], tuple(r["executed"]))
    c = dict(r["call"])
    pk = picks.get(r["shot_no"])
    if r["player"] == seat:
        c["panel_title"] = "Net's shortlist · learned win chance"
        if pk:
            c["contest"] = pk["contest"]
    else:
        e = ev_calls.get(r["shot_no"], {})
        c["panel_title"] = "EV's options ranked by EV"
        for k in ("contest", "plan", "ev", "p_continue", "p_scratch"):
            if k in e:
                c[k] = e[k]
    if shot is not None:
        if c.get("ball") and c["ball"] in shot.balls:
            c["obj_path"] = [[round(float(s.rvw[0][0]), 3), round(float(s.rvw[0][1]), 3)]
                             for s in shot.balls[c["ball"]].history.states]
        c["cue_path"] = [[round(x, 3), round(y, 3)] for x, y in R.cue_path(shot)]
    c["stroke"] = [float(v) for v in c["stroke"]]
    if c.get("cue_place"):
        c["cue_place"] = [float(v) for v in c["cue_place"]]
    recs.append({**{k: v for k, v in r.items() if k not in ("shot", "call", "groups_before", "groups_after", "planned")},
                 "call": c, "groups_before": groups(r["groups_before"]),
                 "groups_after": groups(r["groups_after"]), "narration": narr(r)})
game = {"seed": seed, "winner": m.winner, "end_reason": m.end_reason, "shots": len(recs),
        "players": who, "net_score": score, "log": recs}
rec = R.to_json(game)
json.dump(rec, open(os.path.join(out, "replay_log.json"), "w"), indent=1, default=str)
items = R.write_items(rec, out)
bad = R.verify_replay(items)
for r in recs:
    print(f"[{r['shot_no']:02d}] {r['narration']}")
print("replay check:", "every shot re-simulates exactly" if not bad else f"MISMATCH {bad}")
print(f"winner {who[m.winner]} ({m.end_reason}), net score {score}, {len(recs)} shots")
