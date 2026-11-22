// Render the replay viewer's table sounds (client/sound.js, unchanged) to a dry stereo
// bus + the viewer's room impulse, for an offline soundtrack.
//   node render_sound.mjs SOUND_JS EVENTS.json OUT_PREFIX
// EVENTS.json: {sr, dur, events:[{at, kind, dv, gain, pan, n, break}]} with kinds and
// gains exactly as sound.js's eventsFor() would produce them.
// Writes OUT_PREFIX.dry.f32 (interleaved L,R) and OUT_PREFIX.ir.f32 (interleaved L,R).
import fs from 'node:fs'
import { pathToFileURL } from 'node:url'

const [soundJs, evFile, prefix] = process.argv.slice(2)
const S = await import(pathToFileURL(soundJs).href)
const { sr, dur, events } = JSON.parse(fs.readFileSync(evFile, 'utf8'))
const n = Math.ceil(dur * sr) + sr
const L = new Float32Array(n), R = new Float32Array(n)
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
let seq = 1

// TableAudio._out: gain node, then StereoPanner (equal power, mono source)
function out(when, data, gain, pan) {
  if (gain < 0.02) return
  const x = (clamp(pan || 0, -1, 1) + 1) / 2
  const gl = gain * Math.cos(x * Math.PI / 2), gr = gain * Math.sin(x * Math.PI / 2)
  const i0 = Math.round(when * sr)
  for (let i = 0; i < data.length && i0 + i < n; i++) {
    if (i0 + i < 0) continue
    L[i0 + i] += data[i] * gl
    R[i0 + i] += data[i] * gr
  }
}

function rng(seed) {   // same LCG as sound.js
  let state = (seed >>> 0) || 1
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return (state / 4294967296) * 2 - 1 }
}

// TableAudio._play / _cue / _rack, verbatim logic
function play(e) {
  const when = e.at, pan = e.pan
  if (e.kind === 'cue') out(when, S.cueStrike(sr, e.dv, seq++), e.gain * (e.break ? 0.7 : 0.5), pan)
  else if (e.kind === 'rack') {
    const rnd = rng(seq++)
    for (const [dt, k] of [[0, 1.1], [0.0007, 0.8], [0.0016, 0.7]]) out(when + dt, S.ballClack(sr, e.dv * k, seq++), e.gain * Math.min(1, k), pan)
    const m = clamp((e.n || 8) - 2, 3, 8)
    for (let i = 0; i < m; i++) {
      const f = i / m
      const dt = 0.004 + f ** 1.5 * 0.09 + Math.abs(rnd()) * 0.006
      const dv = e.dv * (0.6 - 0.4 * f) * (0.8 + 0.4 * Math.abs(rnd()))
      out(when + dt, S.ballClack(sr, dv, seq++), e.gain * (0.7 - 0.45 * f), clamp(pan + 0.2 * rnd(), -0.7, 0.7))
    }
  } else if (e.kind === 'ball') out(when, S.ballClack(sr, e.dv, seq++), e.gain, pan)
  else if (e.kind === 'rail') out(when, S.railThump(sr, e.dv, seq++), e.gain * 0.9, pan)
  else if (e.kind === 'pocket') out(when, S.pocketDrop(sr, seq++), e.gain * 0.4, pan)
  else if (e.kind === 'scratch') out(when, S.pocketDrop(sr, seq++), e.gain * 0.38, pan)
}

for (const e of events) play(e)
const inter = (a, b) => { const o = new Float32Array(a.length * 2); for (let i = 0; i < a.length; i++) { o[2 * i] = a[i]; o[2 * i + 1] = b[i] } return o }
fs.writeFileSync(prefix + '.dry.f32', Buffer.from(inter(L, R).buffer))
const ir = S.roomImpulse(sr, 11)
fs.writeFileSync(prefix + '.ir.f32', Buffer.from(inter(ir[0], ir[1]).buffer))
console.log(`rendered ${events.length} events, ${dur.toFixed(1)} s`)
