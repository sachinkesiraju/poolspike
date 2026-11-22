# Pool8 player protocol

One WebSocket per seat at `COWORLD_PLAYER_WS_URL` (`/player?slot=N&token=…`).
Messages are JSON text frames. Units are metres, m/s and degrees, in table
coordinates: `x` runs across the table width, `y` along its length, and the
kitchen (break area) is at `y < head_string_y`. Slot 0 and slot 1 are seats;
`breaker_slot` breaks.

## Server → player

### `hello` (once, on connect)
```json
{"type": "hello", "slot": 0, "seat": "A", "breaker_slot": 0,
 "players": ["Alice", "Bob"],
 "table": {"width": 0.9906, "length": 1.9812, "ball_radius": 0.028575,
           "head_string_y": 0.4953,
           "pockets": {"lb": {"x": …, "y": …, "r": …}, …},
           "pocket_aim_points": {"lb": [0, 0], "lc": [0, 0.99], …}},
 "settings": {"rules": "call-shot", "noise": {"sigma_phi": 0.1, …},
              "max_shots": 80, "shot_timeout_s": 40, "time_bank_s": 420,
              "min_shot_s": 5, "max_invalid": 2},
 "balls": {"cue": {"x": …, "y": …, "pocketed": false}, "1": …},
 "shots": [ …shot records so far (reconnects)… ]}
```

### `turn` (your move)
```json
{"type": "turn", "slot": 0, "shot_no": 7,
 "phase": "shoot" | "break" | "ball_in_hand" | "ball_in_hand_kitchen",
 "balls": {"cue": {"x": 0.41, "y": 1.2, "pocketed": false}, "1": …},
 "my_group": "solids" | "stripes" | null, "opp_group": …,
 "legal_targets": ["1", "3", "6"],
 "balls_left": {"me": 3, "opp": 5},
 "deadline_s": 40, "time_bank_s": [380.2, 401.7]}
```
`legal_targets` are the balls you may hit first and call: anything but the
8 on an open table, your group otherwise, and only the 8 once your group is
cleared.

### `shot` (both players and viewers, after every shot)
This is the public record of what happened. It is also the replay entry.
```json
{"type": "shot", "shot_no": 7, "slot": 0, "phase": "shoot", "kind": "pot",
 "call": {"ball": "3", "pocket": "rt"},
 "intended": {"phi": 51.9, "speed": 2.4, "side": 0, "vertical": -0.4},
 "executed": {"phi": 51.96, "speed": 2.43, "side": 0.01, "vertical": -0.42},
 "result": "continue" | "turnover" | "foul" | "win" | "loss",
 "reason": null | "scratch" | "wrong ball first (9)" | "8 early" | …,
 "pocketed": {"3": "rt"}, "first_contact": "3",
 "balls_before": {…}, "balls_after": {…},
 "groups_before": {"0": null, "1": null}, "groups_after": {"0": "solids", "1": "stripes"},
 "traj": {"cue": {"t0": 0, "xy": [[x, y], …], "down": null}, …},
 "planned": {…same shape, the noise-free intended stroke…},
 "source": "player" | "timeout" | "invalid" | "absent",
 "think_s": 3.4, "note": "3→rt · make 100% · shape for 7→lb", "meta": {…},
 "next_look": {"p": 0.99, "ball": "13", "pocket": "rt"} | null}
```
`traj` samples each moving ball every 1/30 s. `down` is the time the ball
dropped, if it did. `next_look` (after a turnover) is the incoming shooter's
easiest geometric pot.

### `error`
```json
{"type": "error", "shot_no": 7, "message": "ball 9 is not yours to call (legal: [1, 3, 6])", "retries_left": 1}
```

### `final`
```json
{"type": "final", "results": {"scores": [1.0, 0.0], "winner_slot": 0, "end_reason": "8 legally pocketed", …}}
```
Close the socket and exit after `final`.

## Player → server: `shot`

```json
{"type": "shot", "shot_no": 7,
 "call": {"ball": "3", "pocket": "rt"},
 "stroke": {"phi": 51.9, "speed": 2.4, "side": 0.0, "vertical": -0.4},
 "cue": {"x": 0.3, "y": 0.4},
 "note": "optional, ≤300 chars, shown in the replay",
 "meta": {"label": "3→rt", "p_make": 0.97, "ev": 0.74, "plan": {"for": "self", "ball": "7", "pocket": "lb", "p": 0.9}, "...": "optional, ≤4 KB"}}
```

- `shot_no` must echo the `turn` message.
- `stroke.phi` is the cue direction in degrees, counter-clockwise from +x.
  `speed` is the cue-ball speed in m/s: 0.3–8, or up to 12 on the break.
  `side` (english, + = right) and `vertical` (+ = follow, − = draw) are tip
  offsets in ball radii: each within ±0.6, and hypot(side, vertical) ≤ 0.7.
- `call`: the ball and pocket you call. `null` means no call (a legal-hit
  attempt that can never keep you at the table). The break takes no call.
- `cue`: required for the `break` and both ball-in-hand phases, forbidden
  otherwise. Kitchen phases need `y < head_string_y`. The cue may not
  overlap a ball or the rails.
- `note` and `meta`: optional commentary for viewers, never read by the
  rules. `meta` is whitelisted: only the keys below survive, typed and
  truncated. The replay viewer turns these `meta` keys into the shot
  description:
  - `label`, `p_make`, `margin` (aim margin in degrees), `ev`, `v_make` and
    `v_miss`;
  - `plan` {`for`, `ball`, `pocket`, `p`};
  - `alts` (up to 2 alternatives: `kind`, `label`, `p_make`, `ev`);
  - `best_pot`, `no_legal` and `lanes_probed`, for no-call shots;
  - `p_continue` and `p_scratch`, on the break;
  - `search`.

### Delivery rules

- Frames are limited to 64 KiB and must be standard JSON (no `NaN` or
  `Infinity`).
- Messages sent while it isn't your turn are ignored unread. Answer each
  `turn` message only.
- If you reconnect during your turn, the current `turn` is sent again after
  `hello`.
- A client that stops reading its socket is disconnected. The game never
  waits on a slow reader.

### What the server does with it

1. It validates the action. If invalid, you get an `error` with
   `retries_left`. After `max_invalid` retries, the next invalid action plays
   the baseline shot for you.
2. It adds the table's execution noise to your intended stroke. Everyone
   gets the same noise: aim σ = `sigma_phi`° × (1 + `speed_gain`·max(0, v − 2.5)
   + `english_gain`·|side|), speed ×(1 + N(0, `sigma_v`)), tip offsets
   + N(0, `sigma_ab`).
3. It simulates the shot with pooltool's event-based physics and applies
   the rules (README.md).

### Clock

Each shot must arrive within `deadline_s` = max(`min_shot_s`, min(`shot_timeout_s`,
your remaining `time_bank_s`)). Thinking time is deducted from your bank. The
whole match also has a wall clock (`time_limit_s`, default 900 s). When it
runs out, the player with more `time_bank_s` left wins (within 1 s is a draw).

### Baseline shot

On a timeout, an absent seat, or too many invalid actions, the server plays
the scripted baseline for you (`baseline.py`):
- **Shoot:** the geometrically easiest pot with a stun stroke, or a soft
  full hit on the nearest legal ball.
- **Ball in hand:** the cue is placed straight behind the easiest pot.
- **Break:** a centre break at 10 m/s.

## Global viewer: `/global`

This is read-only. You get `hello` (no slot, plus `shots`, `status` and
`results`), then `shot`, `status` and `final` messages:
```json
{"type": "status", "to_move": 1, "phase": "shoot", "shot_no": 8, "deadline_s": 40, "time_bank_s": [...]}
```
