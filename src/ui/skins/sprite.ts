/**
 * Sprite loading shared by the costume renderer and the lobby thumbnails.
 *
 * The image generator sometimes bakes the transparency checkerboard into the
 * pixels instead of using a real alpha channel, so we key out near-white /
 * light-gray (low-saturation, high-lightness) pixels. The colorful,
 * black-outlined art is saturated or dark, so it survives.
 */

/** Copy an image to a canvas and make its checkerboard background transparent. */
export function keyCheckerboard(img: HTMLImageElement): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = img.naturalWidth
  c.height = img.naturalHeight
  const x = c.getContext('2d')
  if (!x) return c
  x.drawImage(img, 0, 0)
  try {
    const data = x.getImageData(0, 0, c.width, c.height)
    const d = data.data
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i]
      const g = d[i + 1]
      const b = d[i + 2]
      const max = Math.max(r, g, b)
      const min = Math.min(r, g, b)
      if (max - min < 26 && min > 168) d[i + 3] = 0
    }
    x.putImageData(data, 0, 0)
  } catch {
    // getImageData throws on a tainted canvas; fall back to the raw draw.
  }
  return c
}

/**
 * Returns a canvas that fills in with the keyed sprite once `src` loads.
 * Safe to insert into the DOM or draw immediately (blank until decoded).
 */
export function keyedSprite(src: string): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.onload = () => {
    const keyed = keyCheckerboard(img)
    canvas.width = keyed.width
    canvas.height = keyed.height
    ctx?.drawImage(keyed, 0, 0)
  }
  img.src = src
  return canvas
}
