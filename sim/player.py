"""Bundled 8-ball players (platform-hosted WebSocket clients).

  python player.py --strategy ev       full EV policy, anytime (policy.py)
  python player.py --strategy lite     EV policy on the budgeted search only
  python player.py --strategy pattern  ev plus a second ply on its best pots
                                       (pattern.py)
  python player.py --strategy pro      safeties, clock-weighted search, break
                                       book (pro.py)
  python player.py --strategy greedy   scripted geometric baseline (baseline.py)

`ev` is anytime: every shot it first runs the `lite` search (about a second
on a few cores), then the full search (POOL_PRESET, default `ev`) in the
time left on the shot clock, and plays the deepest answer that finished.
`pattern` is the same, with the full search's answer followed by its
two-ply refinement in the remaining time.
The search runs in a killable child process, so a slow search never costs
a timeout. POOL_WORKERS sets the search processes; by default it is the
CPUs this container may use (cgroup quota). Protocol: PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace



def _cpu_budget(cap: int = 8, unlimited: int = 4) -> int:
    """Search processes for this container: its CPU quota (cgroup v2 or v1)
    within its CPU affinity, capped. With no quota the pod may land on a
    big node with a small memory request, so use a modest default."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        cores = os.cpu_count() or 1
    quota = None
    try:
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    n = int(quota) if quota else min(cores, unlimited)
    return max(1, min(n, cores, cap))


os.environ.setdefault("POOL_WORKERS", str(_cpu_budget()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline  # noqa: E402
import pattern  # noqa: E402
import pro  # noqa: E402
import policy as P  # noqa: E402

RULES = {"call-shot": "wpa", "bar": "standard"}
GROUPS = {"solids": P.SOLIDS, "stripes": P.STRIPES, None: None}


def log(msg: str) -> None:
    print(f"[player] {msg}", flush=True)


def to_state(balls: dict) -> dict:
    return {"balls": {b: {"x": bl["x"], "y": bl["y"],
                          "pocketed": bl["pocketed"]}
                      for b, bl in balls.items()}}


def to_wire(call: dict) -> dict:
    """Policy call -> protocol action (+ a human-readable note)."""
    phi, v, a, b = call["stroke"]
    act = {"stroke": {"phi": phi % 360.0, "speed": v, "side": a,
                      "vertical": b},
           "call": ({"ball": call["ball"], "pocket": call["pocket"]}
                    if call["kind"] == "pot" else None)}
    if call.get("cue_place"):
        act["cue"] = {"x": call["cue_place"][0], "y": call["cue_place"][1]}
    if call["kind"] == "pot":
        act["note"] = (f"{P._label(call)} · make {call.get('p_make', 0):.0%}"
                       f" · EV {call.get('ev', 0):.2f}")
        plan = call.get("plan") or {}
        if plan.get("ball") and plan.get("for") == "self":
            act["note"] += f" · shape for {plan['ball']}→{plan['pocket']}"
    elif call["kind"] == "break":
        act["note"] = f"break · continue {call.get('p_continue', 0):.0%}"
    else:
        act["note"] = "nothing to call: legal hit"
    act["meta"] = _meta(call)
    return act


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _meta(call: dict) -> dict:
    """The decision's own numbers, for viewers' shot descriptions."""
    m = {"kind": call["kind"], "ev": _r(call.get("ev"))}
    if call["kind"] == "break":
        m["p_continue"] = _r(call.get("p_continue"))
        m["p_scratch"] = _r(call.get("p_scratch"))
        return m
    if call["kind"] == "pot":
        m.update(label=P._label(call), p_make=_r(call.get("p_make")),
                 margin=_r(call.get("margin")), v_make=_r(call.get("v_make")),
                 v_miss=_r(call.get("v_miss")))
    else:
        m.update(ball=call.get("ball"), route=call.get("route"),
                 no_legal=bool(call.get("no_legal")),
                 lanes_probed=call.get("lanes_probed"))
        if call.get("best_pot"):
            bp = call["best_pot"]
            m["best_pot"] = {k: bp.get(k) for k in ("label", "p_make", "ev")}
    plan = call.get("plan") or {}
    if plan.get("ball"):
        m["plan"] = {"for": plan.get("for"), "ball": plan["ball"],
                     "pocket": plan["pocket"], "p": _r(plan.get("p"))}
    m["alts"] = [{k: o.get(k) for k in ("kind", "label", "ball", "route",
                                        "p_make", "ev")}
                 for o in (call.get("contest") or [])[1:3]]
    return m


def decide(obs: dict, cfg: P.Config, seed: int) -> dict:
    state = to_state(obs["balls"])
    ctx = {"me": GROUPS[obs["my_group"]], "opp": GROUPS[obs["opp_group"]]}
    phase = obs["phase"]
    if phase == "break":
        if cfg.name == "plus":
            return to_wire(pro.choose_break(state, cfg, seed))
        return to_wire(P.choose_break(state, cfg, seed))
    if phase.startswith("ball_in_hand"):
        return to_wire(P.choose_bih(state, ctx, cfg, seed,
                                    kitchen=phase == "ball_in_hand_kitchen"))
    return to_wire(P.choose(state, ctx, cfg, seed))


def answers(obs: dict, cfg: P.Config, seed: int, deadline: float):
    """(search name, action) for one stage; the pattern stage answers
    twice (its first ply is the full search)."""
    if cfg.name == "pro":
        state = to_state(obs["balls"])
        ctx = {"me": GROUPS[obs["my_group"]], "opp": GROUPS[obs["opp_group"]]}
        phase = obs["phase"]
        if phase == "break":
            yield "pro-book", to_wire(pro.choose_break(state, cfg, seed))
        elif phase.startswith("ball_in_hand"):
            yield "pro-bih", to_wire(pro.choose_bih(
                state, ctx, cfg, seed, kitchen=phase == "ball_in_hand_kitchen"))
        else:
            for name, call in pro.answers(state, ctx, cfg, seed):
                yield name, to_wire(call)
    elif cfg.name == "pattern" and obs["phase"] == "shoot":
        state = to_state(obs["balls"])
        ctx = {"me": GROUPS[obs["my_group"]], "opp": GROUPS[obs["opp_group"]]}
        for name, call in pattern.answers(state, ctx, cfg, seed, deadline):
            act = to_wire(call)
            if name == "pattern":  # the audit replays the same cut
                act["note"] += f" · looked {call['looked']} pots ahead"
            yield name, act
    else:
        yield ("ev" if cfg.name == "pattern" else cfg.name,
               decide(obs, cfg, seed))


def _worker(conn, rules: str) -> None:
    import signal

    def _quit(*_):  # take the search pool down with us
        if P._POOL is not None:
            P._POOL.terminate()
        os._exit(0)

    signal.signal(signal.SIGTERM, _quit)
    P.RULES = rules
    os.environ["POOL_RULES"] = rules  # the search pool's children read this
    while True:
        job = conn.recv()
        if job is None:
            return
        obs, cfgs, seed, budget = job
        deadline = time.monotonic() + budget - 1.5  # leave time to answer
        for stage, cfg in enumerate(cfgs):  # cheapest first: anytime
            try:
                for name, out in answers(obs, cfg, seed, deadline):
                    conn.send(("ok", stage, name, out))
            except Exception as e:  # noqa: BLE001 — report, never hang
                conn.send(("error", stage, cfg.name, repr(e)))
        conn.send(("done", None, None, None))


class Searcher:
    """The EV search in a killable child process, deepest answer in time."""

    def __init__(self, rules: str):
        self.rules = rules
        self.proc = None

    def _start(self):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, self.rules))
        self.proc.start()

    async def run(self, obs: dict, cfgs: list, seed: int, budget: float):
        """(action or None, status): the deepest stage done within budget."""
        loop = asyncio.get_running_loop()
        best, status = None, "timeout"
        try:
            if self.proc is None or not self.proc.is_alive():
                self._start()
            self.conn.send((obs, cfgs, seed, budget))
            end = time.monotonic() + budget
            while True:
                left = end - time.monotonic()
                ready = left > 0 and await loop.run_in_executor(
                    None, self.conn.poll, left)
                if not ready:
                    # out of time: kill it and warm a fresh one for next shot
                    await loop.run_in_executor(None, self.restart)
                    return best, f"{status}, deeper search timed out"
                kind, _, name, out = self.conn.recv()
                if kind == "done":
                    break
                if kind == "ok":
                    best, status = out, name
                else:
                    status = f"{name} failed: {out}"
        except (EOFError, OSError) as e:  # the worker died: replace it
            await loop.run_in_executor(None, self.restart)
            return best, f"{status}, search worker died ({e!r})"
        return best, status

    def restart(self):
        self.stop()
        self._start()

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.join(5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
            self.proc = None


async def play(strategy: str) -> None:
    import websockets

    url = os.environ["COWORLD_PLAYER_WS_URL"]
    preset = os.environ.get("POOL_PRESET", "ev")
    stages = {"ev": ["lite", preset], "lite": ["lite"],
              "pattern": ["lite", "pattern"], "pro": ["pro"],
              "plus": ["lite", "plus"]}.get(strategy, [])
    cfgs = [P.PRESETS[n] for n in dict.fromkeys(stages)]
    searcher, slot = None, None
    fallback = {}  # baseline action of the shot in progress
    log(f"strategy={strategy} stages={[c.name for c in cfgs]} "
        f"workers={os.environ['POOL_WORKERS']} connecting")
    # ping_timeout=None: keep the socket through long searches even if the
    # engine is slow to answer pings (PLAYER.md)
    async with websockets.connect(url, ping_timeout=None,
                                  max_size=2 ** 24) as ws:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "hello":
                    slot = msg["slot"]
                    st = msg["settings"]
                    P.RULES = RULES.get(st["rules"], "wpa")
                    # plan with the table's real execution noise
                    cfgs = [replace(c, **st["noise"]) for c in cfgs]
                    if cfgs:
                        searcher = Searcher(P.RULES)
                        searcher._start()  # import + warm before the break
                    log(f"slot {slot}, rules {st['rules']}, noise {st['noise']}")
                elif kind == "turn":
                    t0 = time.monotonic()
                    obs_state = to_state(msg["balls"])
                    act = baseline.action(obs_state, msg["phase"],
                                          GROUPS[msg["my_group"]])
                    act.update(note="baseline (greedy geometric)",
                               type="shot", shot_no=msg["shot_no"])
                    fallback = dict(act)
                    if searcher is not None:
                        budget = msg["deadline_s"] * 0.85 - 1.0
                        out, status = await searcher.run(
                            msg, cfgs, msg["shot_no"] * 7 + slot, budget)
                        if out is not None:
                            act = out
                            # say which search produced the shot
                            tag = ("full search" if status == "ev" else
                                   "full search (pattern ran out of clock)"
                                   if status.startswith("ev,") else
                                   "lite search (full ran out of clock)"
                                   if "timed out" in status else
                                   f"{status} search")
                            act["note"] = f"{tag} · {act['note']}"
                            act["meta"]["search"] = tag
                        else:
                            act["note"] = f"search {status}: baseline shot"
                        log(f"shot {msg['shot_no']}: {status}")
                    act.update(type="shot", shot_no=msg["shot_no"])
                    await ws.send(json.dumps(act))
                    log(f"shot {msg['shot_no']} ({msg['phase']}) "
                        f"{time.monotonic() - t0:.1f}s: {act.get('note')}")
                elif kind == "error":
                    log(f"server rejected action: {msg.get('message')}")
                    # resend the baseline shot, which is always valid
                    if msg.get("retries_left", 0) > 0 and \
                            msg.get("shot_no") == fallback.get("shot_no"):
                        await ws.send(json.dumps(fallback))
                elif kind == "final":
                    log(f"final: {msg['results'].get('end_reason')} "
                        f"scores {msg['results'].get('scores')}")
                    return
        finally:
            if searcher is not None:
                searcher.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy",
                    choices=("ev", "lite", "pattern", "pro", "plus",
                             "greedy"),
                    default="ev")
    args = ap.parse_args()
    if args.strategy != "greedy":
        t = time.time()  # compile pooltool's kernels before the clock runs
        import engine
        engine.warm_up()
        log(f"physics warm in {time.time() - t:.1f}s")
    asyncio.run(play(args.strategy))


if __name__ == "__main__":
    main()
