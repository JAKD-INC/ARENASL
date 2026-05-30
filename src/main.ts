import { startCamera } from './camera.ts'
import { createDetector, drawLandmarks } from './landmarks.ts'
import { createConnection } from './net.ts'
import { createOverlay } from './overlay.ts'

const video = document.querySelector<HTMLVideoElement>('#feed')!
const message = document.querySelector<HTMLDivElement>('#message')!
const landmarks = document.querySelector<HTMLCanvasElement>('#landmarks')!

function showMessage(text: string): void {
  message.textContent = text
  message.classList.remove('hidden')
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
    clip: document.querySelector<HTMLVideoElement>('#clip')!,
    ropeMarker: document.querySelector<HTMLElement>('#rope-marker')!,
    score: document.querySelector<HTMLElement>('#score')!,
  })
  const conn = createConnection({ onState: render })
  const detect = await createDetector()

  // Keep the landmark canvas matched to the viewport (device-pixel sized).
  const sizeCanvas = () => {
    landmarks.width = window.innerWidth
    landmarks.height = window.innerHeight
  }
  sizeCanvas()
  window.addEventListener('resize', sizeCanvas)

  // Pump landmarks to the server each animation frame, and draw them.
  const loop = () => {
    if (video.readyState >= 2) {
      const msg = detect(video, performance.now() / 1000)
      conn.send(msg)
      drawLandmarks(landmarks, video, msg)
    }
    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}

main()
