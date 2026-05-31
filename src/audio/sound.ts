/**
 * Procedural WebAudio sound engine — every cue is synthesized, so the game
 * ships with full audio and zero asset files (matching the rest of the project).
 *
 * Browsers require a user gesture before audio can start, so {@link resume} must
 * be called from a click/tap (the lobby Start button). Until then everything
 * no-ops. If WebAudio is unavailable, the whole engine degrades to silence.
 *
 * Signal path: voices → reverb send + dry → master gain → compressor → out.
 */

type Wave = OscillatorType

interface ToneOpts {
  freq: number
  dur?: number
  type?: Wave
  gain?: number
  attack?: number
  release?: number
  when?: number
  glideTo?: number
  reverb?: number
}

const midi = (m: number): number => 440 * Math.pow(2, (m - 69) / 12)
const MINOR_PENTA = [0, 3, 5, 7, 10] // scale steps for the music bed

export class SoundEngine {
  private ctx: AudioContext | null = null
  private master: GainNode | null = null
  private reverb: ConvolverNode | null = null
  private enabled = true
  private musicTimer = 0
  private musicStep = 0
  private holdVoice: { osc: OscillatorNode; gain: GainNode } | null = null

  /** Lazily create the context (call from a user gesture). Safe to call often. */
  async resume(): Promise<void> {
    if (!this.ctx) this.init()
    if (this.ctx?.state === 'suspended') await this.ctx.resume()
  }

  setEnabled(on: boolean): void {
    this.enabled = on
    const m = this.master
    if (!m || !this.ctx) return
    const t = this.ctx.currentTime
    m.gain.cancelScheduledValues(t)
    m.gain.linearRampToValueAtTime(on ? 0.32 : 0, t + 0.08)
  }

  isEnabled(): boolean {
    return this.enabled
  }

  private init(): void {
    try {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctx()
      const master = ctx.createGain()
      master.gain.value = this.enabled ? 0.32 : 0
      const comp = ctx.createDynamicsCompressor()
      master.connect(comp)
      comp.connect(ctx.destination)

      // reverb send for a little space
      const reverb = ctx.createConvolver()
      reverb.buffer = impulse(ctx, 1.6, 2.4)
      const reverbGain = ctx.createGain()
      reverbGain.gain.value = 0.9
      reverb.connect(reverbGain)
      reverbGain.connect(master)

      this.ctx = ctx
      this.master = master
      this.reverb = reverb
    } catch {
      this.ctx = null // silent fallback
    }
  }

  // --- primitives -----------------------------------------------------------

  private tone(o: ToneOpts): void {
    const ctx = this.ctx
    const master = this.master
    if (!ctx || !master) return
    const when = (o.when ?? 0) + ctx.currentTime
    const dur = o.dur ?? 0.2
    const attack = o.attack ?? 0.005
    const release = o.release ?? Math.min(0.25, dur)
    const peak = o.gain ?? 0.3

    const osc = ctx.createOscillator()
    osc.type = o.type ?? 'sine'
    osc.frequency.setValueAtTime(o.freq, when)
    if (o.glideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(1, o.glideTo), when + dur)

    const g = ctx.createGain()
    g.gain.setValueAtTime(0, when)
    g.gain.linearRampToValueAtTime(peak, when + attack)
    g.gain.linearRampToValueAtTime(0.0001, when + dur)

    osc.connect(g)
    g.connect(master)
    if (o.reverb && this.reverb) {
      const send = ctx.createGain()
      send.gain.value = o.reverb
      g.connect(send)
      send.connect(this.reverb)
    }
    osc.start(when)
    osc.stop(when + dur + release)
    osc.onended = () => {
      osc.disconnect()
      g.disconnect()
    }
  }

  private noise(opts: { dur: number; gain?: number; when?: number; lowpass?: number; highpass?: number }): void {
    const ctx = this.ctx
    const master = this.master
    if (!ctx || !master) return
    const when = (opts.when ?? 0) + ctx.currentTime
    const len = Math.floor(ctx.sampleRate * opts.dur)
    const buf = ctx.createBuffer(1, len, ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len)
    const src = ctx.createBufferSource()
    src.buffer = buf
    let node: AudioNode = src
    if (opts.lowpass) {
      const f = ctx.createBiquadFilter()
      f.type = 'lowpass'
      f.frequency.value = opts.lowpass
      node.connect(f)
      node = f
    }
    if (opts.highpass) {
      const f = ctx.createBiquadFilter()
      f.type = 'highpass'
      f.frequency.value = opts.highpass
      node.connect(f)
      node = f
    }
    const g = ctx.createGain()
    g.gain.value = opts.gain ?? 0.2
    node.connect(g)
    g.connect(master)
    src.start(when)
    src.onended = () => {
      src.disconnect()
      g.disconnect()
    }
  }

  private arp(notes: number[], step: number, opts: { type?: Wave; gain?: number; dur?: number; reverb?: number } = {}): void {
    notes.forEach((m, i) =>
      this.tone({
        freq: midi(m),
        when: i * step,
        dur: opts.dur ?? step * 1.8,
        type: opts.type ?? 'triangle',
        gain: opts.gain ?? 0.26,
        reverb: opts.reverb ?? 0.25,
      }),
    )
  }

  // --- cues -----------------------------------------------------------------

  countdown(n: number): void {
    // 3 → 2 → 1 rising ticks
    this.tone({ freq: midi(64 + (3 - n) * 3), dur: 0.18, type: 'triangle', gain: 0.3 })
  }

  go(): void {
    this.arp([60, 64, 67, 72], 0.07, { type: 'triangle', gain: 0.32, reverb: 0.4 })
  }

  /** Capture started — the game is "listening". */
  attemptStart(): void {
    this.tone({ freq: midi(72), dur: 0.08, type: 'sine', gain: 0.18 })
    this.tone({ freq: midi(76), when: 0.06, dur: 0.1, type: 'sine', gain: 0.18 })
  }

  /** A quiet rising tone while the player holds the sign. */
  startHold(): void {
    const ctx = this.ctx
    const master = this.master
    if (!ctx || !master || this.holdVoice) return
    const osc = ctx.createOscillator()
    osc.type = 'sawtooth'
    const filter = ctx.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.value = 700
    const g = ctx.createGain()
    g.gain.value = 0.0
    g.gain.linearRampToValueAtTime(0.06, ctx.currentTime + 0.1)
    osc.frequency.value = midi(57)
    osc.connect(filter)
    filter.connect(g)
    g.connect(master)
    osc.start()
    this.holdVoice = { osc, gain: g }
  }

  updateHold(progress: number): void {
    const ctx = this.ctx
    if (!ctx || !this.holdVoice) return
    this.holdVoice.osc.frequency.linearRampToValueAtTime(midi(57 + progress * 12), ctx.currentTime + 0.05)
  }

  stopHold(): void {
    const ctx = this.ctx
    const v = this.holdVoice
    if (!ctx || !v) return
    v.gain.gain.linearRampToValueAtTime(0.0001, ctx.currentTime + 0.08)
    v.osc.stop(ctx.currentTime + 0.12)
    v.osc.onended = () => {
      v.osc.disconnect()
      v.gain.disconnect()
    }
    this.holdVoice = null
  }

  verdict(tier: 'perfect' | 'great' | 'good' | 'miss', combo = 0): void {
    const lift = Math.min(combo, 8) // small pitch lift as combo grows
    switch (tier) {
      case 'perfect':
        this.arp([72 + lift, 76 + lift, 79 + lift, 84 + lift], 0.06, { type: 'triangle', gain: 0.3, reverb: 0.45 })
        this.tone({ freq: midi(96 + lift), dur: 0.5, type: 'sine', gain: 0.08, reverb: 0.6 })
        break
      case 'great':
        this.arp([69 + lift, 73 + lift, 76 + lift], 0.06, { type: 'triangle', gain: 0.28, reverb: 0.35 })
        break
      case 'good':
        this.tone({ freq: midi(67 + lift), dur: 0.22, type: 'sine', gain: 0.28, reverb: 0.3 })
        this.tone({ freq: midi(79 + lift), dur: 0.18, type: 'sine', gain: 0.12, reverb: 0.3 })
        break
      case 'miss':
        this.tone({ freq: midi(48), glideTo: midi(40), dur: 0.28, type: 'sawtooth', gain: 0.22 })
        this.noise({ dur: 0.2, gain: 0.12, lowpass: 500 })
        break
    }
  }

  comboMilestone(combo: number): void {
    const base = 72 + Math.min(combo, 12)
    this.arp([base, base + 4, base + 7, base + 12, base + 16], 0.05, { type: 'square', gain: 0.14, reverb: 0.5 })
  }

  win(): void {
    this.arp([60, 64, 67, 72, 76, 79], 0.1, { type: 'triangle', gain: 0.3, reverb: 0.5, dur: 0.6 })
  }

  lose(): void {
    this.arp([67, 63, 60, 56], 0.16, { type: 'sine', gain: 0.26, reverb: 0.4, dur: 0.5 })
  }

  // --- generative music bed -------------------------------------------------

  startMusic(): void {
    if (!this.ctx || this.musicTimer) return
    this.musicStep = 0
    const stepDur = 0.2
    const root = 45 // A2
    const tick = (): void => {
      const ctx = this.ctx
      if (!ctx) return
      const step = this.musicStep++
      // bass pulse every 4 steps
      if (step % 4 === 0) {
        this.tone({ freq: midi(root - 12), dur: 0.5, type: 'triangle', gain: 0.12 })
      }
      // soft pad chord every 8 steps
      if (step % 8 === 0) {
        for (const m of [root + 12, root + 19, root + 24]) {
          this.tone({ freq: midi(m), dur: 1.8, type: 'sine', gain: 0.05, attack: 0.4, reverb: 0.5 })
        }
      }
      // sparse pentatonic arpeggio
      if (step % 2 === 0) {
        const note = root + 24 + MINOR_PENTA[(step / 2) % MINOR_PENTA.length]
        this.tone({ freq: midi(note), dur: 0.3, type: 'triangle', gain: 0.06, reverb: 0.4 })
      }
      this.musicTimer = window.setTimeout(tick, stepDur * 1000)
    }
    tick()
  }

  stopMusic(): void {
    if (this.musicTimer) clearTimeout(this.musicTimer)
    this.musicTimer = 0
  }
}

/** Generate a synthetic noise-decay impulse response for the reverb convolver. */
function impulse(ctx: AudioContext, seconds: number, decay: number): AudioBuffer {
  const rate = ctx.sampleRate
  const len = Math.floor(rate * seconds)
  const buf = ctx.createBuffer(2, len, rate)
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch)
    for (let i = 0; i < len; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay)
    }
  }
  return buf
}
