/**
 * Code-drawn sticker library. Every lens accessory is rendered procedurally to
 * an offscreen canvas once, then cached and stamped onto the face each frame —
 * so the whole lens system ships with zero external image assets and works
 * fully offline.
 *
 * Each generator draws into an `s`×`s` box, centred, "upright" (the renderer
 * rotates it to the head roll). Left/right variants are separate ids so paired
 * accessories (ears, hearts) can be anchored independently.
 */

type Draw = (ctx: CanvasRenderingContext2D, s: number) => void

const SIZE = 256
const cache = new Map<string, HTMLCanvasElement>()

/** Render (and cache) a sticker by id, or null if unknown. */
export function getSticker(id: string): HTMLCanvasElement | null {
  const cached = cache.get(id)
  if (cached) return cached
  const draw = STICKERS[id]
  if (!draw) return null
  const c = document.createElement('canvas')
  c.width = SIZE
  c.height = SIZE
  const ctx = c.getContext('2d')
  if (!ctx) return null
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  draw(ctx, SIZE)
  cache.set(id, c)
  return c
}

// --- drawing helpers --------------------------------------------------------

function outline(ctx: CanvasRenderingContext2D, s: number, w = 0.03): void {
  ctx.strokeStyle = 'rgba(20,18,30,0.92)'
  ctx.lineWidth = s * w
  ctx.stroke()
}

/** Draw with center at box centre, optionally mirrored horizontally. */
function centered(
  ctx: CanvasRenderingContext2D,
  s: number,
  side: 1 | -1,
  fn: () => void,
): void {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.scale(side, 1)
  fn()
  ctx.restore()
}

// --- ears -------------------------------------------------------------------

function floppyEar(ctx: CanvasRenderingContext2D, s: number, side: 1 | -1, outer: string, dark: string): void {
  centered(ctx, s, side, () => {
    const g = ctx.createLinearGradient(0, -s * 0.42, 0, s * 0.42)
    g.addColorStop(0, outer)
    g.addColorStop(1, dark)
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.moveTo(s * 0.04, -s * 0.42)
    ctx.bezierCurveTo(s * 0.34, -s * 0.46, s * 0.46, -s * 0.1, s * 0.34, s * 0.28)
    ctx.bezierCurveTo(s * 0.26, s * 0.46, -s * 0.02, s * 0.46, -s * 0.08, s * 0.24)
    ctx.bezierCurveTo(-s * 0.12, -s * 0.06, -s * 0.1, -s * 0.3, s * 0.04, -s * 0.42)
    ctx.closePath()
    ctx.fill()
    outline(ctx, s)
  })
}

function triEar(ctx: CanvasRenderingContext2D, s: number, side: 1 | -1, outer: string, inner: string): void {
  centered(ctx, s, side, () => {
    ctx.fillStyle = outer
    ctx.beginPath()
    ctx.moveTo(-s * 0.3, s * 0.38)
    ctx.lineTo(s * 0.02, -s * 0.44)
    ctx.lineTo(s * 0.32, s * 0.34)
    ctx.closePath()
    ctx.fill()
    outline(ctx, s)
    ctx.fillStyle = inner
    ctx.beginPath()
    ctx.moveTo(-s * 0.14, s * 0.28)
    ctx.lineTo(s * 0.04, -s * 0.24)
    ctx.lineTo(s * 0.2, s * 0.26)
    ctx.closePath()
    ctx.fill()
  })
}

function longEar(ctx: CanvasRenderingContext2D, s: number, side: 1 | -1): void {
  centered(ctx, s, side, () => {
    ctx.fillStyle = '#fdfdff'
    ctx.beginPath()
    ctx.ellipse(s * 0.04, 0, s * 0.16, s * 0.46, side * 0.12, 0, Math.PI * 2)
    ctx.fill()
    outline(ctx, s, 0.025)
    ctx.fillStyle = '#ff9ec4'
    ctx.beginPath()
    ctx.ellipse(s * 0.04, s * 0.02, s * 0.08, s * 0.34, side * 0.12, 0, Math.PI * 2)
    ctx.fill()
  })
}

function roundEar(ctx: CanvasRenderingContext2D, s: number, side: 1 | -1): void {
  centered(ctx, s, side, () => {
    ctx.fillStyle = '#9b6b43'
    ctx.beginPath()
    ctx.arc(0, 0, s * 0.34, 0, Math.PI * 2)
    ctx.fill()
    outline(ctx, s)
    ctx.fillStyle = '#caa07a'
    ctx.beginPath()
    ctx.arc(0, s * 0.04, s * 0.18, 0, Math.PI * 2)
    ctx.fill()
  })
}

function devilHorn(ctx: CanvasRenderingContext2D, s: number, side: 1 | -1): void {
  centered(ctx, s, side, () => {
    const g = ctx.createLinearGradient(0, s * 0.4, 0, -s * 0.44)
    g.addColorStop(0, '#7a0d12')
    g.addColorStop(1, '#ff3b3b')
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.moveTo(-s * 0.16, s * 0.4)
    ctx.quadraticCurveTo(-s * 0.26, -s * 0.1, s * 0.12, -s * 0.46)
    ctx.quadraticCurveTo(-s * 0.02, -s * 0.04, s * 0.16, s * 0.4)
    ctx.closePath()
    ctx.fill()
    outline(ctx, s)
  })
}

// --- noses ------------------------------------------------------------------

const dogNose: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  // snout pad
  ctx.fillStyle = 'rgba(40,30,30,0.18)'
  ctx.beginPath()
  ctx.ellipse(0, s * 0.04, s * 0.26, s * 0.2, 0, 0, Math.PI * 2)
  ctx.fill()
  const g = ctx.createRadialGradient(-s * 0.06, -s * 0.08, s * 0.02, 0, 0, s * 0.22)
  g.addColorStop(0, '#4a4a52')
  g.addColorStop(1, '#15151a')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.moveTo(0, s * 0.16)
  ctx.bezierCurveTo(-s * 0.24, s * 0.14, -s * 0.24, -s * 0.14, 0, -s * 0.12)
  ctx.bezierCurveTo(s * 0.24, -s * 0.14, s * 0.24, s * 0.14, 0, s * 0.16)
  ctx.closePath()
  ctx.fill()
  outline(ctx, s, 0.02)
  ctx.fillStyle = 'rgba(255,255,255,0.45)'
  ctx.beginPath()
  ctx.ellipse(-s * 0.06, -s * 0.05, s * 0.05, s * 0.03, -0.5, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

const catNose: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  // whiskers
  ctx.strokeStyle = 'rgba(30,30,40,0.7)'
  ctx.lineWidth = s * 0.012
  for (const side of [-1, 1]) {
    for (const dy of [-0.04, 0.03, 0.1]) {
      ctx.beginPath()
      ctx.moveTo(side * s * 0.12, s * 0.04)
      ctx.quadraticCurveTo(side * s * 0.34, s * (dy - 0.02), side * s * 0.46, s * dy)
      ctx.stroke()
    }
  }
  ctx.fillStyle = '#ff7da6'
  ctx.beginPath()
  ctx.moveTo(-s * 0.1, -s * 0.04)
  ctx.lineTo(s * 0.1, -s * 0.04)
  ctx.lineTo(0, s * 0.08)
  ctx.closePath()
  ctx.fill()
  outline(ctx, s, 0.018)
  ctx.restore()
}

const pigNose: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.fillStyle = '#ff9bb3'
  ctx.beginPath()
  ctx.ellipse(0, 0, s * 0.26, s * 0.2, 0, 0, Math.PI * 2)
  ctx.fill()
  outline(ctx, s, 0.022)
  ctx.fillStyle = '#c95f7e'
  for (const x of [-0.1, 0.1]) {
    ctx.beginPath()
    ctx.ellipse(s * x, 0, s * 0.05, s * 0.09, 0, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}

const clownNose: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  const g = ctx.createRadialGradient(-s * 0.08, -s * 0.08, s * 0.03, 0, 0, s * 0.26)
  g.addColorStop(0, '#ff6b6b')
  g.addColorStop(1, '#d11')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.arc(0, 0, s * 0.24, 0, Math.PI * 2)
  ctx.fill()
  outline(ctx, s, 0.02)
  ctx.fillStyle = 'rgba(255,255,255,0.6)'
  ctx.beginPath()
  ctx.arc(-s * 0.08, -s * 0.08, s * 0.06, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

// --- eyewear ----------------------------------------------------------------

const glassesRound: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.strokeStyle = '#1a1a22'
  ctx.lineWidth = s * 0.03
  const r = s * 0.16
  const cx = s * 0.22
  for (const sign of [-1, 1]) {
    const grd = ctx.createLinearGradient(sign * cx - r, -r, sign * cx + r, r)
    grd.addColorStop(0, 'rgba(60,200,220,0.55)')
    grd.addColorStop(1, 'rgba(20,40,90,0.7)')
    ctx.fillStyle = grd
    ctx.beginPath()
    ctx.arc(sign * cx, 0, r, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.strokeStyle = 'rgba(255,255,255,0.5)'
    ctx.lineWidth = s * 0.012
    ctx.beginPath()
    ctx.arc(sign * cx, 0, r * 0.7, Math.PI * 1.1, Math.PI * 1.5)
    ctx.stroke()
    ctx.strokeStyle = '#1a1a22'
    ctx.lineWidth = s * 0.03
  }
  ctx.beginPath()
  ctx.moveTo(-cx + r, 0)
  ctx.quadraticCurveTo(0, -s * 0.06, cx - r, 0)
  ctx.stroke()
  ctx.restore()
}

const glassesPixel: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.fillStyle = '#0a0a0a'
  // chunky pixel sunglasses
  const px = s * 0.07
  const lens = (ox: number): void => {
    for (let i = 0; i < 4; i++) {
      const w = i === 0 || i === 3 ? 2 : 3
      ctx.fillRect(ox + i * px - px * 1.5, -px * 1.0 + (i === 0 || i === 3 ? px * 0.5 : 0), px, w * px * 0.5)
    }
    ctx.fillRect(ox - px * 1.5, -px, px * 3.5, px * 1.8)
  }
  lens(-s * 0.2)
  lens(s * 0.2)
  ctx.fillRect(-px, -px, px * 2, px * 0.7)
  ctx.restore()
}

const monocle: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.strokeStyle = '#e0b84a'
  ctx.lineWidth = s * 0.035
  ctx.fillStyle = 'rgba(200,230,255,0.35)'
  ctx.beginPath()
  ctx.arc(0, 0, s * 0.22, 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  ctx.lineWidth = s * 0.02
  ctx.beginPath()
  ctx.moveTo(s * 0.12, s * 0.18)
  ctx.quadraticCurveTo(s * 0.22, s * 0.4, s * 0.06, s * 0.46)
  ctx.stroke()
  ctx.restore()
}

// --- headwear ---------------------------------------------------------------

const crown: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  const g = ctx.createLinearGradient(0, -s * 0.3, 0, s * 0.3)
  g.addColorStop(0, '#ffe488')
  g.addColorStop(1, '#e0a92e')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.moveTo(-s * 0.36, s * 0.22)
  ctx.lineTo(-s * 0.36, -s * 0.08)
  ctx.lineTo(-s * 0.18, s * 0.06)
  ctx.lineTo(0, -s * 0.26)
  ctx.lineTo(s * 0.18, s * 0.06)
  ctx.lineTo(s * 0.36, -s * 0.08)
  ctx.lineTo(s * 0.36, s * 0.22)
  ctx.closePath()
  ctx.fill()
  outline(ctx, s, 0.022)
  const jewels = ['#ff5e7a', '#5ec8ff', '#7dff9e']
  ctx.fillStyle = '#c0392b'
  for (let i = 0; i < 3; i++) {
    ctx.fillStyle = jewels[i]
    ctx.beginPath()
    ctx.arc((i - 1) * s * 0.2, s * 0.1, s * 0.045, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}

const halo: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.shadowColor = '#fff6c0'
  ctx.shadowBlur = s * 0.12
  ctx.strokeStyle = '#ffe066'
  ctx.lineWidth = s * 0.06
  ctx.beginPath()
  ctx.ellipse(0, 0, s * 0.34, s * 0.13, 0, 0, Math.PI * 2)
  ctx.stroke()
  ctx.restore()
}

const flowerCrown: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  const cols = ['#ff8fab', '#ffd23f', '#a78bfa', '#7dd3fc', '#ff8fab', '#86efac']
  const drawFlower = (x: number, y: number, r: number, c: string): void => {
    ctx.fillStyle = c
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2
      ctx.beginPath()
      ctx.ellipse(x + Math.cos(a) * r * 0.7, y + Math.sin(a) * r * 0.7, r * 0.55, r * 0.4, a, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.fillStyle = '#fff3b0'
    ctx.beginPath()
    ctx.arc(x, y, r * 0.45, 0, Math.PI * 2)
    ctx.fill()
  }
  for (let i = 0; i < 6; i++) {
    const t = i / 5 - 0.5
    drawFlower(t * s * 0.74, Math.abs(t) * s * 0.18 - s * 0.04, s * (0.11 - Math.abs(t) * 0.04), cols[i])
  }
  ctx.restore()
}

const partyHat: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  const g = ctx.createLinearGradient(-s * 0.2, 0, s * 0.2, 0)
  g.addColorStop(0, '#ff5e7a')
  g.addColorStop(0.5, '#ffd23f')
  g.addColorStop(1, '#5ec8ff')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.moveTo(-s * 0.22, s * 0.34)
  ctx.lineTo(0, -s * 0.44)
  ctx.lineTo(s * 0.22, s * 0.34)
  ctx.closePath()
  ctx.fill()
  outline(ctx, s, 0.022)
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.arc(0, -s * 0.44, s * 0.06, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

// --- face accents -----------------------------------------------------------

const heart: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2 + s * 0.06)
  ctx.shadowColor = 'rgba(255,60,110,0.6)'
  ctx.shadowBlur = s * 0.06
  const g = ctx.createLinearGradient(0, -s * 0.3, 0, s * 0.3)
  g.addColorStop(0, '#ff7aa8')
  g.addColorStop(1, '#ff2d6b')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.moveTo(0, s * 0.3)
  ctx.bezierCurveTo(-s * 0.42, -s * 0.02, -s * 0.18, -s * 0.36, 0, -s * 0.12)
  ctx.bezierCurveTo(s * 0.18, -s * 0.36, s * 0.42, -s * 0.02, 0, s * 0.3)
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

const star: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.shadowColor = 'rgba(255,210,70,0.7)'
  ctx.shadowBlur = s * 0.07
  ctx.fillStyle = '#ffd23f'
  ctx.beginPath()
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? s * 0.34 : s * 0.15
    const a = (i / 10) * Math.PI * 2 - Math.PI / 2
    ctx[i === 0 ? 'moveTo' : 'lineTo'](Math.cos(a) * r, Math.sin(a) * r)
  }
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

const blush: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  const g = ctx.createRadialGradient(0, 0, s * 0.02, 0, 0, s * 0.3)
  g.addColorStop(0, 'rgba(255,120,160,0.65)')
  g.addColorStop(1, 'rgba(255,120,160,0)')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.ellipse(0, 0, s * 0.3, s * 0.22, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

const mustache: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  ctx.fillStyle = '#2b2118'
  ctx.beginPath()
  ctx.moveTo(0, -s * 0.04)
  ctx.bezierCurveTo(-s * 0.12, -s * 0.16, -s * 0.34, -s * 0.18, -s * 0.46, -s * 0.02)
  ctx.bezierCurveTo(-s * 0.4, s * 0.04, -s * 0.3, s * 0.0, -s * 0.2, s * 0.06)
  ctx.bezierCurveTo(-s * 0.1, s * 0.12, -s * 0.04, s * 0.08, 0, s * 0.04)
  ctx.bezierCurveTo(s * 0.04, s * 0.08, s * 0.1, s * 0.12, s * 0.2, s * 0.06)
  ctx.bezierCurveTo(s * 0.3, s * 0.0, s * 0.4, s * 0.04, s * 0.46, -s * 0.02)
  ctx.bezierCurveTo(s * 0.34, -s * 0.18, s * 0.12, -s * 0.16, 0, -s * 0.04)
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

const tongue: Draw = (ctx, s) => {
  ctx.save()
  ctx.translate(s / 2, s / 2)
  const g = ctx.createLinearGradient(0, -s * 0.3, 0, s * 0.4)
  g.addColorStop(0, '#ff7a9c')
  g.addColorStop(1, '#e84d72')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.moveTo(-s * 0.2, -s * 0.3)
  ctx.lineTo(s * 0.2, -s * 0.3)
  ctx.quadraticCurveTo(s * 0.26, s * 0.36, 0, s * 0.4)
  ctx.quadraticCurveTo(-s * 0.26, s * 0.36, -s * 0.2, -s * 0.3)
  ctx.closePath()
  ctx.fill()
  outline(ctx, s, 0.02)
  ctx.strokeStyle = 'rgba(180,40,70,0.6)'
  ctx.lineWidth = s * 0.02
  ctx.beginPath()
  ctx.moveTo(0, -s * 0.2)
  ctx.lineTo(0, s * 0.2)
  ctx.stroke()
  ctx.restore()
}

export const STICKERS: Record<string, Draw> = {
  'dog-ear-l': (c, s) => floppyEar(c, s, -1, '#8a5a3a', '#5c3a24'),
  'dog-ear-r': (c, s) => floppyEar(c, s, 1, '#8a5a3a', '#5c3a24'),
  'cat-ear-l': (c, s) => triEar(c, s, -1, '#3a3a44', '#ff9ec4'),
  'cat-ear-r': (c, s) => triEar(c, s, 1, '#3a3a44', '#ff9ec4'),
  'bunny-ear-l': (c, s) => longEar(c, s, -1),
  'bunny-ear-r': (c, s) => longEar(c, s, 1),
  'bear-ear-l': (c, s) => roundEar(c, s, -1),
  'bear-ear-r': (c, s) => roundEar(c, s, 1),
  'devil-horn-l': (c, s) => devilHorn(c, s, -1),
  'devil-horn-r': (c, s) => devilHorn(c, s, 1),
  'dog-nose': dogNose,
  'cat-nose': catNose,
  'pig-nose': pigNose,
  'clown-nose': clownNose,
  'glasses-round': glassesRound,
  'glasses-pixel': glassesPixel,
  monocle,
  crown,
  halo,
  'flower-crown': flowerCrown,
  'party-hat': partyHat,
  heart,
  star,
  blush,
  mustache,
  tongue,
}
