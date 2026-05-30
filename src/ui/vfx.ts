import type { SignOutcome } from '../game/types.ts'
import type { GameStore } from '../game/store.ts'

/**
 * Full-screen canvas effects layer. Reacts to `sign` events: a particle burst +
 * floating "+points" on an accepted sign, a subtle red shake on a miss. The
 * animation loop only does work while particles are alive, so it idles cheaply.
 */

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  life: number // 1 → 0
  decay: number
  size: number
  color: string
}

interface FloatingText {
  x: number
  y: number
  vy: number
  life: number
  text: string
  color: string
}

const ACCENT = '#5eead4' // teal accent (matches glass theme)
const MISS = '#f87171'

export class Vfx {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private particles: Particle[] = []
  private texts: FloatingText[] = []
  private running = false
  private flash = 0
  private flashColor = ACCENT
  private lastTs = 0

  constructor(canvas: HTMLCanvasElement, store: GameStore) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('VFX: 2D canvas context unavailable')
    this.ctx = ctx
    this.resize()
    window.addEventListener('resize', () => this.resize())

    store.on('sign', (o) => this.onSign(o))
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1
    this.canvas.width = window.innerWidth * dpr
    this.canvas.height = window.innerHeight * dpr
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  private onSign(o: SignOutcome): void {
    // Only the local player's signs produce on-screen feedback.
    if (o.result.player !== 'me') return
    const cx = window.innerWidth / 2
    const cy = window.innerHeight * 0.62

    if (o.accepted) {
      this.burst(cx, cy, ACCENT, 28)
      this.texts.push({ x: cx, y: cy - 40, vy: -0.06, life: 1, text: `+${o.points}`, color: ACCENT })
      if (o.combo >= 2) {
        this.texts.push({ x: cx, y: cy - 80, vy: -0.05, life: 1, text: `${o.combo} COMBO`, color: '#fde68a' })
      }
      this.flash = 0.35
      this.flashColor = ACCENT
    } else {
      this.flash = 0.4
      this.flashColor = MISS
      this.texts.push({ x: cx, y: cy - 30, vy: -0.04, life: 1, text: 'MISS', color: MISS })
    }
    this.ensureRunning()
  }

  private burst(x: number, y: number, color: string, count: number): void {
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.4
      const speed = 0.15 + Math.random() * 0.35
      this.particles.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        decay: 0.0012 + Math.random() * 0.0015,
        size: 2 + Math.random() * 4,
        color,
      })
    }
  }

  private ensureRunning(): void {
    if (this.running) return
    this.running = true
    this.lastTs = performance.now()
    requestAnimationFrame((ts) => this.frame(ts))
  }

  private frame(ts: number): void {
    const dt = Math.min(48, ts - this.lastTs)
    this.lastTs = ts
    const { ctx } = this
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)

    if (this.flash > 0) {
      ctx.save()
      ctx.globalAlpha = this.flash * 0.25
      ctx.fillStyle = this.flashColor
      ctx.fillRect(0, 0, window.innerWidth, window.innerHeight)
      ctx.restore()
      this.flash = Math.max(0, this.flash - dt / 400)
    }

    for (const p of this.particles) {
      p.x += p.vx * dt
      p.y += p.vy * dt
      p.vy += 0.0006 * dt // gravity
      p.life -= p.decay * dt
      if (p.life <= 0) continue
      ctx.save()
      ctx.globalAlpha = Math.max(0, p.life)
      ctx.fillStyle = p.color
      ctx.shadowColor = p.color
      ctx.shadowBlur = 8
      ctx.beginPath()
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
    }
    this.particles = this.particles.filter((p) => p.life > 0)

    ctx.save()
    ctx.textAlign = 'center'
    ctx.font = '700 28px system-ui, sans-serif'
    for (const t of this.texts) {
      t.y += t.vy * dt
      t.life -= dt / 1100
      if (t.life <= 0) continue
      ctx.globalAlpha = Math.max(0, t.life)
      ctx.fillStyle = t.color
      ctx.shadowColor = t.color
      ctx.shadowBlur = 12
      ctx.fillText(t.text, t.x, t.y)
    }
    ctx.restore()
    this.texts = this.texts.filter((t) => t.life > 0)

    if (this.particles.length || this.texts.length || this.flash > 0) {
      requestAnimationFrame((next) => this.frame(next))
    } else {
      this.running = false
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)
    }
  }
}
