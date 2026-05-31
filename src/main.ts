import { startCamera } from './camera.ts'
import { createDetector, drawLandmarks } from './landmarks.ts'
import { createConnection, type GameState, type LandmarkMessage } from './net.ts'
import { createOverlay } from './overlay.ts'

const video = document.querySelector<HTMLVideoElement>('#feed')!
const message = document.querySelector<HTMLDivElement>('#message')!
const landmarks = document.querySelector<HTMLCanvasElement>('#landmarks')!
const debug = document.querySelector<HTMLPreElement>('#debug')!
const clip = document.querySelector<HTMLVideoElement>('#clip')!
const clipLandmarks = document.querySelector<HTMLCanvasElement>('#clip-landmarks')!

function showMessage(text: string): void {
  message.textContent = text
  message.classList.remove('hidden')
}

// Warm the HTTP cache for upcoming reference clips so prompt changes don't stall.
const preloaded = new Set<string>()
function preload(glosses: string[]): void {
  for (const g of glosses) {
    if (preloaded.has(g)) continue
    preloaded.add(g)
    fetch(`/clips/${g}.mp4`).catch(() => preloaded.delete(g))
  }
}

const bar = (v: number) => '█'.repeat(Math.round(v * 10)) + '░'.repeat(10 - Math.round(v * 10))
const yn = (v: unknown) => (v ? '✓' : '✗')

function renderDebug(
  state: GameState | null,
  msg: LandmarkMessage | null,
  fps: number,
  rx: number,
): void {
  const s = state
  const str = s?.strength ?? 0
  debug.textContent = [
    `target  : ${s?.current ?? '—'}`,
    `dist    : ${s?.distance != null ? s.distance.toFixed(2) : '—'}`,
    `top     : ${(s?.topk ?? []).map((e) => `${e.gloss} ${e.distance.toFixed(2)}`).join('  ') || '—'}`,
    `strength: ${str.toFixed(4)} ${bar(str)}`,
    `score   : ${s?.score ?? 0}`,
    `event   : ${s?.event ?? '—'}`,
    `hands   : L${yn(msg?.handLeft)} R${yn(msg?.handRight)}`,
    `rx/fps  : ${rx} / ${fps.toFixed(0)}`,
  ].join('\n')
}

async function main(): Promise<void> {
  try {
    video.srcObject = await startCamera()
    await video.play()
  } catch (error) {
    const name = error instanceof Error ? error.name : ''
    showMessage(
      name === 'NotAllowedError' || name === 'SecurityError'
        ? 'Camera access denied. Please allow camera permission and reload.'
        : name === 'NotFoundError'
          ? 'No camera found.'
          : `Could not start camera${name ? ` (${name})` : ''}.`,
    )
    return
  }

  const render = createOverlay({
    prompts: document.querySelector<HTMLElement>('#prompts')!,
    clip,
    ropeMarker: document.querySelector<HTMLElement>('#rope-marker')!,
    score: document.querySelector<HTMLElement>('#score')!,
  })

  let latest: GameState | null = null
  let rx = 0 // count of state messages from the server (0 = server dropping frames)
  const conn = createConnection({
    onState: (s) => {
      latest = s
      rx++
      render(s)
      preload(s.queue) // queue = upcoming signs; fetch their clips ahead of time
    },
  })

  // Two independent detectors: one for the live feed, one for the demo clip
  // (separate instances because VIDEO-mode tracking is per-stream/timestamp).
  const detect = await createDetector()
  const detectClip = await createDetector()

  const sizeCanvas = () => {
    landmarks.width = window.innerWidth
    landmarks.height = window.innerHeight
  }
  sizeCanvas()
  window.addEventListener('resize', sizeCanvas)

  let fps = 0
  let prev = performance.now()
  let frameN = 0
  const CLIP_EVERY = 4 // run the demo-clip detector only every Nth frame (perf)

  const loop = () => {
    const now = performance.now()
    const dt = (now - prev) / 1000
    prev = now
    if (dt > 0) fps = fps * 0.9 + (1 / dt) * 0.1
    frameN++

    let msg: LandmarkMessage | null = null
    if (video.readyState >= 2) {
      msg = detect(video, now / 1000)
      conn.send(msg)
      drawLandmarks(landmarks, video, msg) // live vectors over the feed
    }

    // Expected vectors from the demo clip — throttled; it's only a reference.
    if (frameN % CLIP_EVERY === 0 && clip.readyState >= 2 && clip.videoWidth) {
      if (clipLandmarks.width !== clip.clientWidth || clipLandmarks.height !== clip.clientHeight) {
        clipLandmarks.width = clip.clientWidth
        clipLandmarks.height = clip.clientHeight
      }
      const expected = detectClip(clip, now / 1000)
      drawLandmarks(clipLandmarks, clip, expected, { fit: 'fill', mirror: false })
    }

    renderDebug(latest, msg, fps, rx)
    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}

main()
