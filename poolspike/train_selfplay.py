"""Self-improvement loop. One round: play (self + EV), refit on everything so far,
test the new net against the current champion, promote it only if it scores above
0.5. Every few rounds the champion plays EV on fresh seeds. Resumes from loop.json.

  train_selfplay.py START_NET ROUNDS
"""
import json
import os
import subprocess
import sys
import time

PY = "../.venv/bin/python"
R = "../results/training"
N = "nets"
WORKERS = "9"
SELF, EVG, GATE, EVTEST = 150, 50, 150, 150     # seeds per round; each seed is 2 games
EVERY = 3


def sh(*a, env=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run([PY, "-B", "judge_net.py", *a], capture_output=True, text=True, env=e)
    out = "\n".join(l for l in p.stdout.splitlines() if l.strip())
    with open(f"{R}/loop.log", "a") as f:
        f.write(out + "\n")
        if p.returncode:
            f.write("FAILED " + " ".join(a) + "\n" + p.stderr[-1500:] + "\n")
    if p.returncode:
        raise SystemExit("step failed: " + " ".join(a))
    return out


def note(s):
    with open(f"{R}/loop.log", "a") as f:
        f.write(s + "\n")


def score(out):
    return float(out.split(" score ")[1].split()[0])


def main(start, rounds):
    os.makedirs(R, exist_ok=True)
    os.makedirs(N, exist_ok=True)
    sp = f"{R}/loop.json"
    st = json.load(open(sp)) if os.path.exists(sp) else {"round": 0, "champ": start, "data": [], "seed": 40000, "history": []}
    while st["round"] < rounds:
        r = st["round"] + 1
        t = time.time()
        note(f"=== round {r}, champion {st['champ']}, {time.strftime('%c')}")
        s = st["seed"]
        fast = {"SN3_N1": "12"}          # data games use fewer samples, tests use the real setting
        d1, d2 = f"{R}/self{r}.npz", f"{R}/ev{r}.npz"
        sh("gen", st["champ"], "n5:" + st["champ"], str(s), str(s + SELF), WORKERS, "0.3", d1, env=fast)
        s += SELF
        sh("gen", st["champ"], "ev", str(s), str(s + EVG), WORKERS, "0.3", d2, env=fast)
        s += EVG
        st["data"] += [d1, d2]
        new = f"{N}/r{r}.npz"
        sh("fit", new, st["champ"], *st["data"])
        out = sh("eval", new, "n5:" + st["champ"], str(s), str(s + GATE), WORKERS, f"{R}/gate{r}.jsonl")
        s += GATE
        g = score(out)
        if g > 0.5:
            st["champ"] = new
            note(f"promoted {new}, gate {g:.3f}")
        else:
            note(f"kept {st['champ']}, gate {g:.3f}")
        rec = {"round": r, "gate": g, "champ": st["champ"]}
        if r % EVERY == 0:
            out = sh("eval", st["champ"], "ev", str(s), str(s + EVTEST), WORKERS, f"{R}/ev_test{r}.jsonl")
            s += EVTEST
            rec["ev"] = score(out)
        st["history"].append(rec)
        st["round"], st["seed"] = r, s
        with open(sp, "w") as f:
            json.dump(st, f, indent=0)
        note(f"round {r} done in {(time.time() - t) / 60:.0f} min: {rec}")
    note("LOOP FINISHED")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
