"""Speed patches for pooltool 0.6.0's event loop. Results are identical.

Each patch replaces pure bookkeeping (cache scans, min-searches) with an
equivalent faster form. Event times, event order and tie-breaking are
unchanged, so every simulation is bit-for-bit the same as stock pooltool.
`tests/test_fastsim.py` checks this on random strokes.

Import this module before simulating (policy.py does).
"""

from __future__ import annotations

import pooltool.evolution.event_based.cache as _cache
from pooltool.events.utils import event_type_to_ball_indices

_IDX = {et: tuple(sorted(ix)) for et, ix in event_type_to_ball_indices.items()}


def _invalidate(self, event) -> None:
    """CollisionCache.invalidate without the per-key generator: drop every
    cached time involving a ball the event changed."""
    bad = self._get_invalid_ball_ids(event)
    for et, times in self.times.items():
        idx = _IDX.get(et, ())
        if not idx or not times:
            continue
        if len(idx) == 1:
            i = idx[0]
            dead = [k for k in times if k[i] in bad]
        else:
            i, j = idx
            dead = [k for k in times if k[i] in bad or k[j] in bad]
        for k in dead:
            del times[k]


_cache.CollisionCache.invalidate = _invalidate


# --------------------------------------------------------- next-event search
# Stock pooltool tests every (ball, segment) key against the cache on every
# event. Keys for one ball are always added together (here) and dropped
# together (invalidate), so one probe per ball tells whether that ball needs
# recomputing. Insertion order, and so min()'s tie-breaking, is unchanged.

import numpy as np  # noqa: E402

import pooltool.constants as const  # noqa: E402
import pooltool.evolution.event_based.simulate as _sim  # noqa: E402
from pooltool.events import EventType  # noqa: E402

_solve_t = _sim.solve


def _mu(state, params):
    return params.u_s if state.s == const.sliding else params.u_r


def get_next_ball_linear_cushion_collision(shot, collision_cache):
    if not shot.table.has_linear_cushions:
        return _sim.null_event(np.inf)
    cache = collision_cache.times.setdefault(EventType.BALL_LINEAR_CUSHION, {})
    cushions = list(shot.table.cushion_segments.linear.values())
    probe = cushions[0].id
    t = shot.t
    for ball in shot.balls.values():
        if (ball.id, probe) in cache:
            continue
        state, params = ball.state, ball.params
        if state.s in const.nontranslating:
            for cushion in cushions:
                cache[(ball.id, cushion.id)] = np.inf
            continue
        mu = _mu(state, params)
        for cushion in cushions:
            cache[(ball.id, cushion.id)] = t + _solve_t.ball_linear_cushion_collision_time(
                rvw=state.rvw, s=state.s, lx=cushion.lx, ly=cushion.ly,
                l0=cushion.l0, p1=cushion.p1, p2=cushion.p2,
                direction=cushion.direction, mu=mu, m=params.m, g=params.g,
                R=params.R)
    obj_ids = min(cache, key=cache.__getitem__)
    return _sim.ball_linear_cushion_collision(
        ball=shot.balls[obj_ids[0]],
        cushion=shot.table.cushion_segments.linear[obj_ids[1]],
        time=cache[obj_ids])


def get_next_ball_circular_cushion_event(shot, collision_cache):
    if not shot.table.has_circular_cushions:
        return _sim.null_event(np.inf)
    cache = collision_cache.times.setdefault(EventType.BALL_CIRCULAR_CUSHION, {})
    cushions = list(shot.table.cushion_segments.circular.values())
    probe = cushions[0].id
    t = shot.t
    for ball in shot.balls.values():
        if (ball.id, probe) in cache:
            continue
        state, params = ball.state, ball.params
        if state.s in const.nontranslating:
            for cushion in cushions:
                cache[(ball.id, cushion.id)] = np.inf
            continue
        mu = _mu(state, params)
        for cushion in cushions:
            cache[(ball.id, cushion.id)] = t + _solve_t.ball_circular_cushion_collision_time(
                rvw=state.rvw, s=state.s, a=cushion.a, b=cushion.b,
                r=cushion.radius, mu=mu, m=params.m, g=params.g, R=params.R)
    ball_id, cushion_id = min(cache, key=cache.__getitem__)
    return _sim.ball_circular_cushion_collision(
        ball=shot.balls[ball_id],
        cushion=shot.table.cushion_segments.circular[cushion_id],
        time=cache[(ball_id, cushion_id)])


def get_next_ball_pocket_collision(shot, collision_cache):
    if not shot.table.has_pockets:
        return _sim.null_event(np.inf)
    cache = collision_cache.times.setdefault(EventType.BALL_POCKET, {})
    pockets = list(shot.table.pockets.values())
    probe = pockets[0].id
    t = shot.t
    for ball in shot.balls.values():
        if (ball.id, probe) in cache:
            continue
        state, params = ball.state, ball.params
        if state.s in const.nontranslating:
            for pocket in pockets:
                cache[(ball.id, pocket.id)] = np.inf
            continue
        mu = _mu(state, params)
        for pocket in pockets:
            cache[(ball.id, pocket.id)] = t + _solve_t.ball_pocket_collision_time(
                rvw=state.rvw, s=state.s, a=pocket.a, b=pocket.b,
                r=pocket.radius, mu=mu, m=params.m, g=params.g, R=params.R)
    ball_id, pocket_id = min(cache, key=cache.__getitem__)
    return _sim.ball_pocket_collision(
        ball=shot.balls[ball_id], pocket=shot.table.pockets[pocket_id],
        time=cache[(ball_id, pocket_id)])


_sim.get_next_ball_linear_cushion_collision = get_next_ball_linear_cushion_collision
_sim.get_next_ball_circular_cushion_event = get_next_ball_circular_cushion_event
_sim.get_next_ball_pocket_collision = get_next_ball_pocket_collision


# ------------------------------------------------------------ cushion model
# Stock _solve builds six debug f-strings of numpy arrays per cushion hit,
# whether or not debug logging is on. Same maths, no strings.

import pooltool.physics.resolve.ball_cushion.stronge_compliant.model as _scm  # noqa: E402
import pooltool.ptmath as ptmath  # noqa: E402
from pooltool.objects.ball.datatypes import BallState  # noqa: E402


def _cushion_solve(ball, cushion, omega_ratio):
    rvw = ball.state.rvw.copy()
    normal_direction = cushion.get_normal_3d(ball.xyz)
    relative_contact_velocity = ptmath.surface_velocity(
        rvw, -normal_direction, ball.params.R)
    v_n_0, v_t_0, tangent_direction = ptmath.decompose_normal_tangent(
        relative_contact_velocity, normal_direction, True)
    beta_t, beta_n = 3.5, 1.0
    v_t_f, v_n_f = _scm.resolve_collinear_compliant_frictional_inelastic_collision(
        v_t_0=v_t_0, v_n_0=v_n_0, m=ball.params.m, beta_t=beta_t,
        beta_n=beta_n, mu=ball.params.f_c, e_n=ball.params.e_c, k_n=1e3,
        eta_squared=(beta_t / beta_n / omega_ratio ** 2))
    Dv_n = (v_n_f - v_n_0) / beta_n
    rvw[1] += Dv_n * normal_direction
    Dv_t = (v_t_f - v_t_0) / beta_t
    rvw[1] += Dv_t * tangent_direction
    rvw[2] += (2.5 / ball.params.R) * ptmath.cross(
        -normal_direction, Dv_t * tangent_direction)
    rvw[1][2] = 0.0
    ball.state = BallState(rvw, const.sliding)
    return ball, cushion


_scm._solve = _cushion_solve


# -------------------------------------------------------- early-exit loop

def simulate_until(shot, stop) -> bool:
    """pt.simulate(shot, inplace=True) that can stop early: `stop(event)` is
    called after each resolved event; True ends the run. Returns whether it
    stopped early. The loop is pooltool's own, so a run that isn't stopped
    is identical to pt.simulate."""
    sim = _sim._SimulationState(shot, _sim.DEFAULT_ENGINE, None,
                                _sim.INCLUDED_EVENTS, 0)
    sim.init()
    while not sim.done:
        event = sim.step()
        if stop(event):
            return True
        if not sim.done:
            sim.update_caches(event)
    return False
