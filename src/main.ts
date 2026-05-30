import { startCamera } from './camera.ts'
import { GameStore } from './game/store.ts'
import { MockDriver } from './game/mockDriver.ts'
import { Hud } from './ui/hud.ts'
import { Vfx } from './ui/vfx.ts'
import { ResultsScreen } from './ui/results.ts'
import { MediaPipeLandmarkProvider } from './ui/lenses/landmarkProvider.mediapipe.ts'
import { createColorFilterRenderer } from './ui/filters/colorFilter.ts'
import { LensRenderer } from './ui/lenses/lensRenderer.ts'
import { LookController } from './ui/looks/controller.ts'
import { LookPicker } from './ui/looks/picker.ts'

const video = document.querySelector<HTMLVideoElement>('#feed')!
const message = document.querySelector<HTMLDivElement>('#message')!
const filterCanvas = document.querySelector<HTMLCanvasElement>('#filter')!
const lensCanvas = document.querySelector<HTMLCanvasElement>('#lens')!
const vfxCanvas = document.querySelector<HTMLCanvasElement>('#vfx')!
const hudRoot = document.querySelector<HTMLDivElement>('#hud')!
const pickerRoot = document.querySelector<HTMLDivElement>('#picker')!
const resultsRoot = document.querySelector<HTMLDivElement>('#results')!

const MATCH_ID = 'dev-match'

function showMessage(text: string): void {
  message.textContent = text
  message.classList.remove('hidden')
}

async function main(): Promise<void> {
  let stream: MediaStream
  try {
    stream = await startCamera()
    video.srcObject = stream
  } catch (error) {
    const name = error instanceof Error ? error.name : ''
    switch (name) {
      case 'NotAllowedError':
      case 'SecurityError':
        showMessage('Camera access denied. Please allow camera permission and reload.')
        break
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        showMessage('No camera found.')
        break
      default:
        showMessage(`Could not start camera${name ? ` (${name})` : ''}.`)
    }
    return
  }

  // --- game + overlay layers ---
  const store = new GameStore({ seed: 1, myName: 'You', oppName: 'Rival' })
  new Hud(hudRoot, store)
  new Vfx(vfxCanvas, store)
  const results = new ResultsScreen(resultsRoot)

  // --- match flow ---
  // The store ends the match when HP hits 0 and emits 'finished'; then we stop
  // the driver and reveal the results.
  const driver = new MockDriver(store)
  store.on('finished', () => {
    driver.stop()
    void results.show(store, MATCH_ID)
  })

  const startMatch = async (): Promise<void> => {
    pickerRoot.classList.add('hidden')
    await store.runCountdown(3)
    driver.start()
  }

  // --- looks layer: face lenses + color filters (WebGL) ---
  // The face-mesh landmark provider feeds the lens renderer. The color pass is
  // pure WebGL over the video and needs no model.
  const landmarks = new MediaPipeLandmarkProvider(video)

  const colorRenderer = createColorFilterRenderer(filterCanvas, video)
  colorRenderer?.startLoop()

  const lensRenderer = new LensRenderer(lensCanvas, video, landmarks)
  lensRenderer.start()

  const looks = new LookController(video, colorRenderer, lensRenderer)
  new LookPicker(pickerRoot, looks, () => void startMatch())

  // MediaPipe model loads from CDN; if it fails (offline), lenses simply won't
  // anchor — color filters and the rest of the overlay still work.
  landmarks.start().catch((err) => console.warn('Landmark provider unavailable:', err))
}

void main()
