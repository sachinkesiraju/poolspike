# viz: watch Poolspike play, with sound

Renders a game as a video: each shot's call, the planned paths, the ranked
shortlist with make rates and learned win chances, then the shot itself, with
table sounds (cue strike, ball clacks, cushions, pocket drops, room reverb).
`../media/poolspike_vs_ev.mp4` was made this way.

Setup: the base install from `../bench/pool8/requirements.txt`, then
`pip install -r requirements.txt`, plus `ffmpeg` and Node.js on your PATH.

## One Poolspike game vs EV

```bash
cd viz
../.venv/bin/python -B rec_game.py ../poolspike/nets/poolspike.npz 110001 A out/
POOL_OUT=out POOL_NAMES="A=Poolspike|B=EV search" ../.venv/bin/python -B fast_video.py
../.venv/bin/python -B soundtrack.py out/
open out/game.mp4          # or serve out/ and open index.html for a clickable shot list
```

`rec_game.py NET SEED SEAT OUT` replays one game exactly as the evaluation
plays it (seed 110001, seat A is the game in `media/`) and checks that every
shot re-simulates identically. `fast_video.py` renders it with pooltool's 3D
scene (`--flat` for a top-down OpenCV render without a GPU; `--speed=`,
`--fps=`). `soundtrack.py` synthesizes the audio from the simulator's own
collision events and muxes it in.

## A gallery of EV games

```bash
../.venv/bin/python -B gallery.py 3 21 42 --draft
```

Plays each seed EV vs EV (`replay.py`), renders and scores each one, checks
every shot with `audit_policy.py`, and writes `gallery/index.html`. Serve
`gallery/` with a server that supports HTTP Range requests so seeking works.

| File | What it does |
|---|---|
| `rec_game.py` | records a Poolspike vs EV game, with Poolspike's shortlist for the panel |
| `replay.py` | plays and narrates an EV vs EV game; writes the video items |
| `fast_video.py` | renders video items to `game.mp4` (3D or `--flat`) |
| `soundtrack.py`, `render_sound.mjs`, `client/sound.js` | table sounds, rendered offline in Node |
| `viewer.py` | `index.html` with the video and a clickable shot list |
| `gallery.py`, `audit_policy.py` | multi-game gallery and per-shot audit |
| `plane_H.json` | camera calibration for drawing overlays on the 3D render |
