// Table sounds synthesized from the replay trajectories. The record has no
// collision list, so hits are read off direction changes, balls leaving rest,
// and the pocket time (`down`). DT matches table.js / engine.TRAJ_DT.
const DT = 1 / 30

const STORE = 'pool8-sound'

function hitGain(dv) {
  return Math.max(0.15, Math.min(1, (dv / 5) ** 0.6))
}

function cueGain(speed, isBreak) {
  const g = Math.min(1, 0.38 + speed / 14)
  return isBreak ? Math.min(1, g * 1.2) : g
}

// Events in shot-local time (0 = the cue strikes, same clock as positionsAt).
export function eventsFor(shot, table) {
  if (!shot || !table) return []
  const W = table.width, L = table.length, R = table.ball_radius
  const traj = shot.traj || {}
  const starts = []
  for (const [id, tr] of Object.entries(traj)) {
    if (id === 'cue' || tr.xy.length < 2) continue
    const dx = tr.xy[1][0] - tr.xy[0][0], dy = tr.xy[1][1] - tr.xy[0][1]
    const sp = Math.hypot(dx, dy) / DT
    if (sp < 0.2) continue
    starts.push({ id, t: tr.t0, x: tr.xy[0][0], y: tr.xy[0][1], s: sp })
  }
  starts.sort((a, b) => a.t - b.t)

  const speed = shot.executed?.speed ?? shot.intended?.speed ?? 2
  const isBreak = shot.kind === 'break'
  const c0 = traj.cue?.xy?.[0] || [W / 2, L / 4]
  const events = [{ t: 0, kind: 'cue', dv: speed, gain: cueGain(speed, isBreak), break: isBreak, x: c0[0], y: c0[1] }]

  let i = 0
  while (i < starts.length) {
    let j = i
    while (j + 1 < starts.length && starts[j + 1].t - starts[i].t <= 0.05) j++
    const group = starts.slice(i, j + 1)
    if (group.length >= 4) {
      const dv = Math.max(...group.map((g) => g.s))
      events.push({ t: group[0].t, kind: 'rack', dv, n: group.length, gain: Math.min(1, hitGain(dv) * 1.05), x: group[0].x, y: group[0].y })
    } else {
      for (const g of group) events.push({ t: g.t, kind: 'ball', dv: g.s * 1.4, gain: hitGain(g.s * 1.4), x: g.x, y: g.y })
    }
    i = j + 1
  }

  const launched = (id, t) => starts.some((s) => s.id === id && Math.abs(s.t - t) < 0.12)
  const kinks = []
  for (const [id, tr] of Object.entries(traj)) {
    const xy = tr.xy
    const vels = []
    for (let k = 1; k < xy.length; k++) {
      const vx = (xy[k][0] - xy[k - 1][0]) / DT
      const vy = (xy[k][1] - xy[k - 1][1]) / DT
      vels.push({ t: tr.t0 + k * DT, x: xy[k][0], y: xy[k][1], vx, vy, s: Math.hypot(vx, vy) })
    }
    for (let k = 1; k < vels.length; k++) {
      const a = vels[k - 1], b = vels[k]
      if (a.s < 0.15 && b.s < 0.15) continue
      const dv = Math.hypot(b.vx - a.vx, b.vy - a.vy)
      const ang = Math.acos(Math.max(-1, Math.min(1, (a.vx * b.vx + a.vy * b.vy) / ((a.s * b.s) || 1)))) * 180 / Math.PI
      if (dv < 0.7 && ang < 35) continue
      if (tr.down != null && b.t > tr.down - DT * 2) continue
      if (b.x < -R || b.y < -R || b.x > W + R || b.y > L + R) continue
      if (launched(id, b.t)) continue
      kinks.push({ id, t: b.t, x: b.x, y: b.y, dv, pvx: a.vx, pvy: a.vy, vx: b.vx, vy: b.vy })
    }
  }

  const used = new Set()
  const margin = R * 3.2
  for (let a = 0; a < kinks.length; a++) {
    if (used.has(a)) continue
    const k = kinks[a]
    if (k.id === 'cue' && starts.some((s) => Math.hypot(k.x - s.x, k.y - s.y) < R * 8 && Math.abs(k.t - s.t) < 0.08)) {
      used.add(a)
      continue
    }
    let partner = -1, best = Infinity
    for (let b = a + 1; b < kinks.length; b++) {
      if (used.has(b) || kinks[b].id === k.id || Math.abs(kinks[b].t - k.t) > DT + 1e-6) continue
      const d = Math.hypot(kinks[b].x - k.x, kinks[b].y - k.y)
      if (d < best) { best = d; partner = b }
    }
    const edgeX = Math.min(k.x, W - k.x), edgeY = Math.min(k.y, L - k.y)
    const rail = ((k.pvx * k.vx < 0 && edgeX < margin) || (k.pvy * k.vy < 0 && edgeY < margin))
      && !(edgeX < margin && edgeY < margin) && k.dv > 0.75
    if (partner >= 0 && best < R * 6) {
      used.add(a); used.add(partner)
      const dv = Math.max(k.dv, kinks[partner].dv)
      events.push({ t: Math.min(k.t, kinks[partner].t), kind: 'ball', dv, gain: hitGain(dv), x: k.x, y: k.y })
    } else if (rail) {
      used.add(a)
      events.push({ t: k.t, kind: 'rail', dv: k.dv, gain: hitGain(k.dv) * 0.62, x: k.x, y: k.y })
    } else if (k.dv > 1.6) {
      used.add(a)
      events.push({ t: k.t, kind: 'ball', dv: k.dv, gain: hitGain(k.dv), x: k.x, y: k.y })
    }
  }

  const called = shot.call?.ball
  for (const [id, tr] of Object.entries(traj)) {
    if (tr.down == null) continue
    const win = shot.result === 'win' && id === '8'
    const end = tr.xy[tr.xy.length - 1] || [W / 2, L / 2]
    events.push({
      t: tr.down,
      kind: id === 'cue' ? 'scratch' : 'pocket',
      dv: 1,
      gain: id === 'cue' ? 0.7 : win ? 1 : called === id ? 0.9 : 0.72,
      win,
      x: end[0],
      y: end[1],
    })
  }

  events.sort((a, b) => a.t - b.t || (a.kind < b.kind ? -1 : 1))
  const out = []
  for (const e of events) {
    const prev = out[out.length - 1]
    if (prev && prev.kind === e.kind && (e.kind === 'ball' || e.kind === 'rail') && e.t - prev.t < 0.04) {
      if (e.gain > prev.gain) out[out.length - 1] = e
      continue
    }
    out.push(e)
  }
  return out
}


// --- Synthesis -------------------------------------------------------------
// Every sound is rendered from contact physics at the speed of that hit,
// rather than replaying a few fixed samples at different rates.
//
// Ball on ball: two phenolic spheres (57.15 mm, 170 g) stay in Hertzian
// contact for ~0.3 ms, a little shorter the harder they meet. What you hear
// is the rigid balls accelerating: a sphere radiates its acceleration through
// a band-pass at sqrt(2) c / 2πR ≈ 2.7 kHz (Q 0.707), and the two balls push
// apart, so the far one cancels part of the near one after up to 2R / c. That
// gives the crisp, bright, very short clack. The balls' own vibration modes
// sit near 17 kHz and add only a faint glassy edge; the weight comes from the
// room, not from a ringing tone.
const C_AIR = 343
const BALL_R = 0.028575
const RADIATION_HZ = Math.SQRT2 * C_AIR / (2 * Math.PI * BALL_R)

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

function rng(seed) {
  let state = (seed >>> 0) || 1
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return (state / 4294967296) * 2 - 1
  }
}

// RBJ biquad, in place.
function biquad(data, kind, freq, q, sr) {
  const w0 = 2 * Math.PI * Math.min(freq, sr * 0.45) / sr
  const cw = Math.cos(w0), alpha = Math.sin(w0) / (2 * q)
  const b = kind === 'bandpass' ? [alpha, 0, -alpha]
    : kind === 'lowpass' ? [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
      : [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
  const a0 = 1 + alpha, a1 = -2 * cw / a0, a2 = (1 - alpha) / a0
  const b0 = b[0] / a0, b1 = b[1] / a0, b2 = b[2] / a0
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0
  for (let i = 0; i < data.length; i++) {
    const x = data[i]
    const y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    x2 = x1; x1 = x; y2 = y1; y1 = y
    data[i] = y
  }
  return data
}

// Contact force sin(πt/τ)^shape from t0, scaled so its samples sum to amp:
// a push much shorter than a mode's period then drives it like an impulse.
function addPulse(out, sr, t0, tau, amp, shape = 1.5) {
  const i0 = Math.max(0, Math.floor(t0 * sr)), i1 = Math.min(out.length - 1, Math.ceil((t0 + tau) * sr))
  const vals = []
  let sum = 0
  for (let i = i0; i <= i1; i++) {
    const t = i / sr - t0
    const f = t > 0 && t < tau ? Math.sin(Math.PI * t / tau) ** shape : 0
    vals.push(f)
    sum += f
  }
  if (!sum) return out
  for (let i = i0; i <= i1; i++) out[i] += amp * vals[i - i0] / sum
  return out
}

// Modal bank: each mode a decaying sine whose impulse response has amplitude
// `amp`, so a slow push only excites the modes below 1/τ.
function resonate(force, sr, modes, out = new Float32Array(force.length)) {
  for (const m of modes) {
    if (m.freq > sr * 0.45) continue
    const w = 2 * Math.PI * m.freq / sr
    const r = Math.exp(-1 / (m.tau * sr))
    const c1 = 2 * r * Math.cos(w), c2 = -r * r, g = m.amp * Math.sin(w)
    let y1 = 0, y2 = 0
    for (let i = 0; i < force.length; i++) {
      const y = c1 * y1 + c2 * y2 + g * force[i]
      out[i] += y
      y2 = y1; y1 = y
    }
  }
  return out
}

function peakOf(data) {
  let p = 0
  for (let i = 0; i < data.length; i++) p = Math.max(p, Math.abs(data[i]))
  return p || 1
}

function mix(out, src, gain) {
  const g = gain / peakOf(src)
  for (let i = 0; i < out.length; i++) out[i] += src[i] * g
  return out
}

function noiseBurst(n, sr, rnd, len, freq, q) {
  const d = new Float32Array(n)
  const m = Math.min(n, Math.round(len * sr))
  for (let i = 0; i < m; i++) d[i] = rnd() * (1 - i / m) ** 2
  return biquad(d, 'bandpass', freq, q, sr)
}

const normalized = (data, peak = 0.9) => mix(new Float32Array(data.length), data, peak)

// v: closing speed (m/s).
export function ballClack(sr, v, seed) {
  const rnd = rng(seed)
  v = clamp(v, 0.05, 12)
  const tau = 3.9e-4 * v ** -0.2 * (1 + 0.05 * rnd())
  const n = Math.round(sr * 0.012)
  const t0 = 0.0002
  const cos = 0.35 + 0.65 * Math.abs(rnd()) // listener vs the line of centres
  const acc = new Float32Array(n)
  addPulse(acc, sr, t0, tau, 1)
  addPulse(acc, sr, t0 + 2 * BALL_R * cos / C_AIR, tau, -1)
  biquad(acc, 'bandpass', RADIATION_HZ, Math.SQRT1_2, sr)
  const force = addPulse(new Float32Array(n), sr, t0, tau, 1)
  const k = 1 + 0.03 * rnd()
  const ring = resonate(force, sr, [
    { freq: 16900 * k, tau: 0.0018, amp: 1 },
    { freq: 18300 * k, tau: 0.0014, amp: 0.7 },
  ])
  const out = mix(new Float32Array(n), acc, 1)
  mix(out, ring, 0.1 * clamp(v / 3, 0.2, 1) ** 0.5)
  return normalized(out)
}

// Leather tip on the cue ball: contact lasts ~1–1.5 ms, so the same sphere
// radiation gives a rounder "tok". The maple shaft adds a faint longitudinal
// ring (c / 2L ≈ 1.6 kHz) and the bridge hand a soft low knock.
export function cueStrike(sr, v, seed) {
  const rnd = rng(seed)
  v = clamp(v, 0.3, 12)
  const tau = clamp(1.2e-3 * (v / 2) ** -0.25, 0.6e-3, 2e-3) * (1 + 0.05 * rnd())
  const n = Math.round(sr * 0.05)
  const force = addPulse(new Float32Array(n), sr, 0.0002, tau, 1, 1.2)
  const acc = biquad(force.slice(), 'bandpass', RADIATION_HZ, Math.SQRT1_2, sr)
  const wood = resonate(force, sr, [
    { freq: 1620 * (1 + 0.02 * rnd()), tau: 0.006, amp: 1 },
    { freq: 3240, tau: 0.003, amp: 0.4 },
    { freq: 240, tau: 0.012, amp: 0.8 },
  ])
  const out = mix(new Float32Array(n), acc, 1)
  mix(out, wood, 0.24)
  mix(out, noiseBurst(n, sr, rnd, 0.0012, 3500, 0.9), 0.1)
  return normalized(out)
}

// Ball into the cushion: the rubber holds it for several ms, so there is no
// clack. You hear the rail and cabinet answer the push (a low modal thump),
// the quick stiffening of the cushion nose (some mid knock), and cloth slap.
export function railThump(sr, v, seed) {
  const rnd = rng(seed)
  v = clamp(v, 0.1, 8)
  const tau = 6e-3 * (v / 2) ** -0.15 * (1 + 0.05 * rnd())
  const n = Math.round(sr * 0.12)
  const force = new Float32Array(n)
  addPulse(force, sr, 0.0005, tau, 0.5, 2)
  addPulse(force, sr, 0.0005, tau * 0.18, 0.5, 1.5)
  const j = () => 1 + 0.03 * rnd()
  const out = resonate(force, sr, [
    { freq: 98 * j(), tau: 0.04, amp: 0.3 },
    { freq: 176 * j(), tau: 0.03, amp: 0.6 },
    { freq: 315 * j(), tau: 0.02, amp: 0.9 },
    { freq: 560 * j(), tau: 0.012, amp: 1 },
    { freq: 980 * j(), tau: 0.007, amp: 0.8 },
    { freq: 1700 * j(), tau: 0.004, amp: 0.45 },
  ])
  mix(out, noiseBurst(n, sr, rnd, 0.003, 1600, 1), 0.12 + 0.12 * clamp(v / 4, 0, 1))
  return normalized(out)
}

// Pocketed ball: it knocks the pocket facing as it falls (a short click
// into a deep cabinet thunk), then lands in the pocket ~0.1 s later.
export function pocketDrop(sr, seed) {
  const rnd = rng(seed)
  const n = Math.round(sr * 0.32)
  const land = 0.105 + 0.015 * Math.abs(rnd())
  const force = new Float32Array(n)
  addPulse(force, sr, 0.0005, 1.6e-3, 1, 1.5)
  addPulse(force, sr, land, 5e-3, 0.7, 2)
  const j = () => 1 + 0.03 * rnd()
  const out = resonate(force, sr, [
    { freq: 72 * j(), tau: 0.06, amp: 0.35 },
    { freq: 128 * j(), tau: 0.05, amp: 0.6 },
    { freq: 205 * j(), tau: 0.035, amp: 0.9 },
    { freq: 340 * j(), tau: 0.022, amp: 1 },
    { freq: 610 * j(), tau: 0.012, amp: 0.8 },
    { freq: 1100 * j(), tau: 0.006, amp: 0.5 },
  ])
  const knock = biquad(addPulse(new Float32Array(n), sr, 0.0002, 8e-4, 1), 'bandpass', RADIATION_HZ, Math.SQRT1_2, sr)
  mix(out, knock, 0.35)
  return normalized(out)
}

// A live pool-hall room: floor bounce under the table, ceiling and walls,
// then a diffuse tail (RT60 ≈ 0.55 s) whose highs die first, decorrelated
// between the ears. Wet only; the dry hit goes around it.
export function roomImpulse(sr, seed) {
  const rnd = rng(seed)
  const n = Math.round(sr * 0.9)
  const ch = [new Float32Array(n), new Float32Array(n)]
  const early = [[0.0048, 0.42, -0.2], [0.0071, 0.33, 0.3], [0.0112, 0.26, -0.5], [0.0149, 0.22, 0.6],
    [0.0187, 0.17, -0.1], [0.0236, 0.15, 0.4], [0.0291, 0.12, -0.6], [0.0353, 0.1, 0.2]]
  for (const [t, g, pan] of early) {
    const i = Math.round(t * sr)
    ch[0][i] += g * Math.sqrt((1 - pan) / 2)
    ch[1][i] += g * Math.sqrt((1 + pan) / 2)
  }
  const rt60 = 0.55
  for (const d of ch) {
    let lp = 0
    for (let i = Math.round(0.006 * sr); i < n; i++) {
      const t = i / sr
      const fc = 1400 + 8600 * Math.exp(-t / 0.12)
      const a = Math.exp(-2 * Math.PI * fc / sr)
      lp = a * lp + (1 - a) * rnd()
      d[i] += lp * 0.5 * Math.exp(-6.9 * t / rt60) * (1 - Math.exp(-(t - 0.006) / 0.012))
    }
    biquad(d, 'lowpass', 7000, Math.SQRT1_2, sr)
  }
  // fix the reverb energy (direct-to-reverberant ≈ +6 dB for a click)
  let e = 0
  for (const d of ch) for (let i = 0; i < n; i++) e += d[i] * d[i]
  const g = Math.sqrt(0.25 / e)
  for (const d of ch) for (let i = 0; i < n; i++) d[i] *= g
  return ch
}

let seq = 1

export class TableAudio {
  constructor(buttonParent) {
    let stored = null
    try { stored = localStorage.getItem(STORE) } catch { /* private mode */ }
    this.enabled = stored !== '0'
    this.ctx = null
    this.bus = null
    this._shot = null
    this._table = null
    this._events = []
    this._i = 0
    this._last = -1
    this._held = false
    this.btn = document.createElement('button')
    this.btn.id = 'sound'
    this.btn.type = 'button'
    this._label()
    this.btn.onclick = () => {
      this.enabled = !this.enabled
      try { localStorage.setItem(STORE, this.enabled ? '1' : '0') } catch { /* private mode */ }
      if (this.bus) this.bus.gain.value = this.enabled ? 0.9 : 0
      this._label()
      if (this.enabled) this.unlock()
    }
    buttonParent?.appendChild(this.btn)
    const arm = () => { if (this.enabled) this.unlock() }
    window.addEventListener('pointerdown', arm)
    window.addEventListener('keydown', arm)
  }

  _label() {
    this.btn.textContent = this.enabled ? 'Sound' : 'Muted'
    this.btn.classList.toggle('off', !this.enabled)
    this.btn.setAttribute('aria-pressed', String(this.enabled))
    const live = this.ctx?.state === 'running'
    this.btn.title = this.enabled ? (live ? 'Turn sound off' : 'Click anywhere to start sound') : 'Turn sound on'
  }

  unlock() {
    if (!this.enabled) return
    const fresh = !this.ctx
    if (fresh) this._boot()
    if (this.ctx.state === 'suspended') this.ctx.resume().then(() => this._label()).catch(() => {})
    if (fresh) this._held = true
    this._label()
  }

  _boot() {
    const ctx = new AudioContext()
    this.ctx = ctx
    this.sr = ctx.sampleRate
    // Slow attack so the crack itself is not flattened. Only a stacked break ducks.
    const comp = ctx.createDynamicsCompressor()
    comp.threshold.value = -6
    comp.knee.value = 6
    comp.ratio.value = 1.8
    comp.attack.value = 0.012
    comp.release.value = 0.08
    const bus = ctx.createGain()
    bus.gain.value = this.enabled ? 0.9 : 0
    const room = ctx.createConvolver()
    room.normalize = false
    const ir = roomImpulse(this.sr, 11)
    const irBuf = ctx.createBuffer(2, ir[0].length, this.sr)
    irBuf.getChannelData(0).set(ir[0])
    irBuf.getChannelData(1).set(ir[1])
    room.buffer = irBuf
    bus.connect(comp)
    bus.connect(room)
    room.connect(comp)
    comp.connect(ctx.destination)
    this.bus = bus
  }

  // playing: clock is running. jump: seek/scrub/loop — catch up silently.
  sync(shot, table, local, playing, jump, speed) {
    this._table = table
    if (shot !== this._shot) {
      this._shot = shot
      this._events = eventsFor(shot, table)
      this._i = 0
      this._last = local
      jump = true
    }
    const live = this.enabled && this.ctx?.state === 'running'
    // Waiting on the first gesture: keep the cursor so a click during the
    // aim still gets the cue strike. Muted or suspended: drop elapsed hits.
    if (!live) {
      if (this.enabled && !this.ctx) { this._last = local; return }
      while (this._i < this._events.length && this._events[this._i].t <= local) this._i++
      this._last = local
      return
    }
    if (this._held) {
      this._held = false
      const tail = local - 0.12
      while (this._i < this._events.length && this._events[this._i].t < tail) this._i++
    }
    const limit = 0.08 + 0.12 * (speed || 1)
    const rewind = jump || local < this._last - 0.02
    if (local < 0 || rewind || !playing || local - this._last > limit) {
      if (rewind) this._i = 0
      while (this._i < this._events.length && this._events[this._i].t <= local) this._i++
      this._last = local
      return
    }
    const due = []
    while (this._i < this._events.length && this._events[this._i].t <= local) due.push(this._events[this._i++])
    this._last = local
    const pockets = due.filter((e) => e.kind === 'pocket' || e.kind === 'scratch' || e.kind === 'cue' || e.kind === 'rack')
    const hits = due.filter((e) => e.kind === 'ball' || e.kind === 'rail').sort((a, b) => b.gain - a.gain).slice(0, 4)
    let n = 0
    for (const e of pockets.concat(hits)) this._play(e, this.ctx.currentTime + (n++) * 0.006)
  }

  // Broadcast view: the table's long axis runs across the screen.
  _pan(e) {
    const L = this._table?.length
    if (!L || e.y == null) return 0
    return clamp((e.y / L - 0.5) * 0.9, -0.6, 0.6)
  }

  _play(e, when) {
    const pan = this._pan(e)
    if (e.kind === 'cue') this._cue(when, e, pan)
    else if (e.kind === 'rack') this._rack(when, e, pan)
    else if (e.kind === 'ball') this._out(when, ballClack(this.sr, e.dv, seq++), e.gain, pan)
    else if (e.kind === 'rail') this._out(when, railThump(this.sr, e.dv, seq++), e.gain * 0.9, pan)
    else if (e.kind === 'pocket') this._out(when, pocketDrop(this.sr, seq++), e.gain * 0.4, pan)
    else if (e.kind === 'scratch') this._out(when, pocketDrop(this.sr, seq++), e.gain * 0.38, pan)
  }

  _out(when, data, gain, pan) {
    if (gain < 0.02) return
    const ctx = this.ctx
    const buf = ctx.createBuffer(1, data.length, this.sr)
    buf.getChannelData(0).set(data)
    const src = ctx.createBufferSource()
    src.buffer = buf
    const g = ctx.createGain()
    g.gain.setValueAtTime(gain, when)
    src.connect(g)
    if (pan && ctx.createStereoPanner) {
      const p = ctx.createStereoPanner()
      p.pan.setValueAtTime(pan, when)
      g.connect(p)
      p.connect(this.bus)
    } else g.connect(this.bus)
    src.start(when)
  }

  _cue(when, e, pan) {
    this._out(when, cueStrike(this.sr, e.dv, seq++), e.gain * (e.break ? 0.7 : 0.5), pan)
  }

  // The cue ball's hit on the head ball runs through the rack within a
  // couple of ms (one thick crack), then the pack spreads in smaller clacks.
  _rack(when, e, pan) {
    const rnd = rng(seq++)
    for (const [dt, k] of [[0, 1.1], [0.0007, 0.8], [0.0016, 0.7]]) {
      this._out(when + dt, ballClack(this.sr, e.dv * k, seq++), e.gain * Math.min(1, k), pan)
    }
    const m = clamp((e.n || 8) - 2, 3, 8)
    for (let i = 0; i < m; i++) {
      const f = i / m
      const dt = 0.004 + f ** 1.5 * 0.09 + Math.abs(rnd()) * 0.006
      const dv = e.dv * (0.6 - 0.4 * f) * (0.8 + 0.4 * Math.abs(rnd()))
      this._out(when + dt, ballClack(this.sr, dv, seq++), e.gain * (0.7 - 0.45 * f), clamp(pan + 0.2 * rnd(), -0.7, 0.7))
    }
  }
}
