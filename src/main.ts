import { startCamera } from './camera.ts'
import { GameStore } from './game/store.ts'
import { MockDriver } from './game/mockDriver.ts'
import { Hud } from './ui/hud.ts'
import { Vfx } from './ui/vfx.ts'
import { ResultsScreen } from './ui/results.ts'
import { MediaPipePoseProvider } from './ui/skins/landmarkProvider.mediapipe.ts'
import { SkinRenderer } from './ui/skins/renderer.ts'
import { SkinPicker } from './ui/skins/picker.ts'

const video = document.querySelector<HTMLVideoElement>('#feed')!
const message = document.querySelector<HTMLDivElement>('#message')!
const skinsCanvas = document.querySelector<HTMLCanvasElement>('#skins')!
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

  // --- skins layer (face-anchored via standalone MediaPipe) + lobby ---
  const landmarks = new MediaPipePoseProvider(video)
  const renderer = new SkinRenderer(skinsCanvas, video, landmarks)
  renderer.start()
  new SkinPicker(pickerRoot, video, renderer, () => void startMatch())
  // MediaPipe model loads from CDN; if it fails (offline), skins simply won't
  // anchor — the rest of the overlay is unaffected.
  landmarks.start().catch((err) => console.warn('Landmark provider unavailable:', err))
}

void main()
