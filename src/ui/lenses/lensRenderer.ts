import type { LandmarkProvider } from '../../game/types.ts'
import type { ColorFilter } from '../filters/colorFilter.ts'
import { coverParams, type ScreenPoint } from '../coverFit.ts'
import { buildFaceRig, triggerActive, type FaceAnchor, type FaceRig, type Trigger } from './faceRig.ts'
import { getSticker } from './stickers.ts'

/**
 * Draws the active face lens on a transparent canvas above the camera.
 *
 * A {@link Lens} is a stack of {@link LensLayer}s: each is either a cached
 * {@link getSticker} accessory or a live particle {@link Effect}, hung on a
 * {@link FaceAnchor} from the per-frame {@link FaceRig} and optionally gated by
 * a facial-expression {@link Trigger} (open your mouth, smile, …). Sticker sizes
 * are multiples of the face width so they track distance automatically.
 */

export type Effect = 'rainbow' | 'fire' | 'hearts' | 'sparkle' | 'tears'

export interface LensLayer {
  /** Cached sticker id (see stickers.ts). */
  sticker?: string
  /** Live particle effect. */
  effect?: Effect
  /** Anchor for stickers/effects. */
  anchor?: FaceAnchor
  /** Size as a multiple of face width. */
  scale?: number
  /** Nudge down(+)/up(−) as a fraction of face width, in head-local space. */
  offsetY?: number
  /** Nudge right(+)/left(−) as a fraction of face width, in head-local space. */
  offsetX?: number
  /** Only show / emit while this expression is held. Default: always. */
  trigger?: Trigger
  /** Rotate the sticker with head roll. Default true. */
  rotate?: boolean
}

export interface Lens {
  id: string
  layers: LensLayer[]
  /** Optional color grade paired with the lens. */
  filter?: ColorFilter
}

interface Particle {
  kind: Effect
  x: number
  y: number
  vx: number
  vy: number
  life: number
  decay: number
  size: number
  rot: number
  vrot: number
  hue: number
}

export class LensRenderer {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private video: HTMLVideoElement
  private provider: LandmarkProvider
  private lens: Lens | null = null
  private particles: Particle[] = []
  private rafId = 0
  private lastTs = 0

  constructor(canvas: HTMLCanvasElement, video: HTMLVideoElement, provider: LandmarkProvider) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('LensRenderer: 2D canvas context unavailable')
    this.ctx = ctx
    this.video = video
    this.provider = provider
    this.resize()
    window.addEventListener('resize', () => this.resize())
  }

  setLens(lens: Lens | null): void {
    this.lens = lens
    this.particles = []
    // Warm the sticker cache so the first frame draws instantly.
    for (const layer of lens?.layers ?? []) if (layer.sticker) getSticker(layer.sticker)
  }

  start(): void {
    if (this.rafId) return
    this.lastTs = performance.now()
    const loop = (ts: number): void => {
      this.rafId = requestAnimationFrame(loop)
      this.draw(ts)
    }
    this.rafId = requestAnimationFrame(loop)
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1
    this.canvas.width = window.innerWidth * dpr
    this.canvas.height = window.innerHeight * dpr
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  private draw(ts: number): void {
    const dt = Math.min(48, ts - this.lastTs)
    this.lastTs = ts
    const ctx = this.ctx
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)

    const lens = this.lens
    const frame = this.provider.latest()
    if (!lens || !frame?.face) {
      this.particles = []
      return
    }

    const cover = coverParams(this.video)
    const face = frame.face
    const rig = buildFaceRig(face, cover)
    const blend = frame.blendshapes

    // local axes from head roll
    const right: ScreenPoint = { x: Math.cos(rig.angle), y: Math.sin(rig.angle) }
    const down: ScreenPoint = { x: -Math.sin(rig.angle), y: Math.cos(rig.angle) }
    const place = (layer: LensLayer): ScreenPoint => {
      const a = rig.anchor(layer.anchor ?? 'face')
      const dx = (layer.offsetX ?? 0) * rig.width
      const dy = (layer.offsetY ?? 0) * rig.width
      return {
        x: a.x + right.x * dx + down.x * dy,
        y: a.y + right.y * dx + down.y * dy,
      }
    }

    for (const layer of lens.layers) {
      const active = triggerActive(layer.trigger ?? 'always', blend)
      if (!active) continue
      if (layer.sticker) {
        this.drawSticker(layer, place(layer), rig)
      } else if (layer.effect) {
        this.emit(layer.effect, place(layer), rig, dt)
      }
    }

    this.updateParticles(dt)
  }

  private drawSticker(layer: LensLayer, pos: ScreenPoint, rig: FaceRig): void {
    const sprite = getSticker(layer.sticker!)
    if (!sprite) return
    const w = rig.width * (layer.scale ?? 0.5)
    const h = (w * sprite.height) / sprite.width
    const ctx = this.ctx
    ctx.save()
    ctx.translate(pos.x, pos.y)
    if (layer.rotate !== false) ctx.rotate(rig.angle)
    ctx.drawImage(sprite, -w / 2, -h / 2, w, h)
    ctx.restore()
  }

  // --- particle effects -----------------------------------------------------

  private emit(kind: Effect, pos: ScreenPoint, rig: FaceRig, dt: number): void {
    const n = Math.round((dt / 16) * RATE[kind])
    const ref = rig.width
    for (let i = 0; i < n; i++) {
      const spread = (Math.random() - 0.5) * ref * 0.6
      switch (kind) {
        case 'rainbow':
          this.particles.push(p(kind, pos.x + spread, pos.y, spread * 0.01, 0.25 + Math.random() * 0.25, 0.0016, ref * 0.16, ((pos.x + spread) % 360)))
          break
        case 'fire':
          this.particles.push(p(kind, pos.x + spread * 0.6, pos.y, spread * 0.006, 0.2 + Math.random() * 0.2, 0.0026, ref * (0.22 + Math.random() * 0.18), 40 * Math.random()))
          break
        case 'hearts':
          this.particles.push(p(kind, pos.x + spread, pos.y, spread * 0.004, -(0.12 + Math.random() * 0.1), 0.0012, ref * (0.12 + Math.random() * 0.1), 0))
          break
        case 'sparkle':
          this.particles.push(p(kind, pos.x + (Math.random() - 0.5) * ref, pos.y + (Math.random() - 0.5) * ref * 0.6, 0, 0, 0.004, ref * (0.06 + Math.random() * 0.08), 0))
          break
        case 'tears':
          this.particles.push(p(kind, pos.x, pos.y, 0, 0.08 + Math.random() * 0.06, 0.0016, ref * 0.07, 0))
          break
      }
    }
  }

  private updateParticles(dt: number): void {
    const ctx = this.ctx
    const heart = getSticker('heart')
    const star = getSticker('star')
    for (const pt of this.particles) {
      pt.life -= pt.decay * dt
      if (pt.life <= 0) continue
      pt.x += pt.vx * dt
      pt.y += pt.vy * dt
      pt.rot += pt.vrot * dt

      ctx.save()
      ctx.globalAlpha = Math.max(0, Math.min(1, pt.life))
      switch (pt.kind) {
        case 'rainbow': {
          pt.vy += 0.0009 * dt
          ctx.globalCompositeOperation = 'lighter'
          ctx.fillStyle = `hsl(${pt.hue}, 95%, 60%)`
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, pt.size, 0, Math.PI * 2)
          ctx.fill()
          break
        }
        case 'fire': {
          pt.vy += 0.0011 * dt
          pt.size *= 0.99
          ctx.globalCompositeOperation = 'lighter'
          const g = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, pt.size)
          g.addColorStop(0, 'rgba(255,240,160,0.9)')
          g.addColorStop(0.4, 'rgba(255,140,40,0.8)')
          g.addColorStop(1, 'rgba(200,20,20,0)')
          ctx.fillStyle = g
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, pt.size, 0, Math.PI * 2)
          ctx.fill()
          break
        }
        case 'hearts': {
          pt.x += Math.sin(pt.life * 8) * 0.4
          if (heart) ctx.drawImage(heart, pt.x - pt.size / 2, pt.y - pt.size / 2, pt.size, pt.size)
          break
        }
        case 'sparkle': {
          const sc = 0.6 + 0.4 * Math.sin(pt.life * Math.PI)
          if (star) ctx.drawImage(star, pt.x - (pt.size * sc) / 2, pt.y - (pt.size * sc) / 2, pt.size * sc, pt.size * sc)
          break
        }
        case 'tears': {
          pt.vy += 0.0006 * dt
          ctx.fillStyle = 'rgba(120,200,255,0.85)'
          ctx.beginPath()
          ctx.ellipse(pt.x, pt.y, pt.size * 0.6, pt.size, 0, 0, Math.PI * 2)
          ctx.fill()
          break
        }
      }
      ctx.restore()
    }
    this.particles = this.particles.filter((pt) => pt.life > 0 && pt.y < window.innerHeight + 80)
  }
}

const RATE: Record<Effect, number> = { rainbow: 5, fire: 6, hearts: 1.2, sparkle: 2, tears: 0.5 }

function p(
  kind: Effect, x: number, y: number, vx: number, vy: number, decay: number, size: number, hue: number,
): Particle {
  return { kind, x, y, vx, vy, life: 1, decay, size, rot: 0, vrot: (Math.random() - 0.5) * 0.01, hue }
}
