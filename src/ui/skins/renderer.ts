import type { Landmark, LandmarkProvider, SkinDefinition } from '../../game/types.ts'
import { keyCheckerboard } from './sprite.ts'

/**
 * Draws the active costume — pose-anchored sprite layers — on a transparent
 * canvas above the video.
 *
 * Pose landmarks are normalized (0..1) to the *source* video frame; the video
 * is shown with `object-fit: cover`, so we replicate that cover transform to
 * place sprites correctly. The provider already mirrors x, so this canvas draws
 * in plain (un-mirrored) screen coordinates.
 *
 * Sprite images can ship with a baked-in transparency checkerboard (the image
 * generator sometimes draws the pattern instead of real alpha), so on load we
 * key out near-white / light-gray pixels into a transparent offscreen canvas.
 *
 * Only `costume` skins draw here; `css-filter` / `none` clear the canvas.
 */

// MediaPipe Pose landmark indices we anchor to.
const POSE = {
  earL: 7,
  earR: 8,
  shoulderL: 11,
  shoulderR: 12,
} as const

interface ScreenPoint {
  x: number
  y: number
}

export class SkinRenderer {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private video: HTMLVideoElement
  private provider: LandmarkProvider
  private skin: SkinDefinition | null = null
  private rafId = 0
  /** Decoded + checkerboard-keyed sprites, keyed by src. */
  private sprites = new Map<string, HTMLCanvasElement>()
  private loading = new Set<string>()

  constructor(canvas: HTMLCanvasElement, video: HTMLVideoElement, provider: LandmarkProvider) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('SkinRenderer: 2D canvas context unavailable')
    this.ctx = ctx
    this.video = video
    this.provider = provider
    this.resize()
    window.addEventListener('resize', () => this.resize())
  }

  setSkin(skin: SkinDefinition | null): void {
    this.skin = skin && skin.kind === 'costume' ? skin : null
    for (const layer of this.skin?.costume ?? []) this.load(layer.src)
  }

  start(): void {
    if (this.rafId) return
    const loop = (): void => {
      this.rafId = requestAnimationFrame(loop)
      this.draw()
    }
    loop()
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  /** Returns the keyed sprite canvas once decoded, else undefined while loading. */
  private load(src: string): HTMLCanvasElement | undefined {
    const ready = this.sprites.get(src)
    if (ready) return ready
    if (!this.loading.has(src)) {
      this.loading.add(src)
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        this.sprites.set(src, keyCheckerboard(img))
        this.loading.delete(src)
      }
      img.onerror = () => this.loading.delete(src)
      img.src = src
    }
    return undefined
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1
    this.canvas.width = window.innerWidth * dpr
    this.canvas.height = window.innerHeight * dpr
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  private draw(): void {
    const { ctx } = this
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)
    const skin = this.skin
    const frame = this.provider.latest()
    if (!skin?.costume || !frame?.pose) return
    const pose = frame.pose

    const earL = this.toScreen(pose[POSE.earL])
    const earR = this.toScreen(pose[POSE.earR])
    const shoulderL = this.toScreen(pose[POSE.shoulderL])
    const shoulderR = this.toScreen(pose[POSE.shoulderR])

    const headCenter = mid(earL, earR)
    const headWidth = dist(earL, earR)
    const headAngle = Math.atan2(earR.y - earL.y, earR.x - earL.x)

    const shoulderCenter = mid(shoulderL, shoulderR)
    const shoulderWidth = dist(shoulderL, shoulderR)
    const shoulderAngle = Math.atan2(shoulderR.y - shoulderL.y, shoulderR.x - shoulderL.x)

    for (const layer of skin.costume) {
      const sprite = this.load(layer.src)
      if (!sprite) continue
      if (layer.anchor === 'head') {
        const w = headWidth * layer.scale
        const cx = headCenter.x
        const cy = headCenter.y + headWidth * (layer.offsetY ?? 0)
        // Hat: its bottom (brim) sits at the anchor point.
        this.drawSprite(sprite, cx, cy, w, headAngle, 'bottom')
      } else {
        const w = shoulderWidth * layer.scale
        const cx = shoulderCenter.x
        const cy = shoulderCenter.y + shoulderWidth * (layer.offsetY ?? 0)
        // Cape/robe: its top sits at the shoulder line and drapes down.
        this.drawSprite(sprite, cx, cy, w, shoulderAngle, 'top')
      }
    }
  }

  /** Source-normalized landmark → screen px, replicating object-fit: cover. */
  private toScreen(p: Landmark): ScreenPoint {
    const W = window.innerWidth
    const H = window.innerHeight
    const vw = this.video.videoWidth || W
    const vh = this.video.videoHeight || H
    const s = Math.max(W / vw, H / vh)
    const ox = (W - vw * s) / 2
    const oy = (H - vh * s) / 2
    return { x: ox + p.x * vw * s, y: oy + p.y * vh * s }
  }

  private drawSprite(
    sprite: HTMLCanvasElement,
    cx: number,
    cy: number,
    w: number,
    angle: number,
    vAnchor: 'top' | 'center' | 'bottom',
  ): void {
    const h = (w * sprite.height) / sprite.width
    const y = vAnchor === 'top' ? 0 : vAnchor === 'center' ? -h / 2 : -h
    const { ctx } = this
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    ctx.drawImage(sprite, -w / 2, y, w, h)
    ctx.restore()
  }
}

function dist(a: ScreenPoint, b: ScreenPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function mid(a: ScreenPoint, b: ScreenPoint): ScreenPoint {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}
