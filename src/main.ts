import { startCamera } from './camera.ts'
import { GameStore } from './game/store.ts'
import { MockDriver } from './game/mockDriver.ts'
import { SignCapture } from './game/signCapture.ts'
import { Hud } from './ui/hud.ts'
import { Vfx } from './ui/vfx.ts'
import { CaptureUI } from './ui/capture.ts'
import { ResultsScreen } from './ui/results.ts'
import { CoachOverlay, type CoachStep } from './ui/coach.ts'
import { PracticeBar } from './ui/practice.ts'
import { SoundEngine } from './audio/sound.ts'
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
const captureRoot = document.querySelector<HTMLDivElement>('#capture')!
const hudRoot = document.querySelector<HTMLDivElement>('#hud')!
const pickerRoot = document.querySelector<HTMLDivElement>('#picker')!
const practiceRoot = document.querySelector<HTMLDivElement>('#practicebar')!
const coachRoot = document.querySelector<HTMLDivElement>('#coach')!
const resultsRoot = document.querySelector<HTMLDivElement>('#results')!

const MATCH_ID = 'dev-match'
const COACH_KEY = 'arenasl.coachSeen'

/** First-run tutorial steps for Practice mode. */
const COACH_STEPS: CoachStep[] = [
  { title: 'Welcome to ArenaSL 👋', body: 'Learn ASL by signing — let’s warm up. No opponent, no pressure here.' },
  { target: '.hud-word', title: 'Copy this sign', body: 'Watch the looping demo, then make the same sign with your hands.' },
  { target: '.cap-stage', title: 'Raise your hands', body: 'Lift your hands into view and hold the sign until the ring fills.' },
  { title: 'Score big', body: 'A clean sign scores PERFECT and builds your combo. Miss? Just try again — nothing to lose.' },
  { target: '[data-looks]', title: 'Make it yours', body: 'Tap Looks any time to add face lenses and color filters to your camera.' },
  { title: 'Ready to battle?', body: 'Hit “Start match” to race a rival. Your HP stays hidden until the big reveal at the end!' },
]

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
  const sound = new SoundEngine()
  new Hud(hudRoot, store)
  new Vfx(vfxCanvas, store)
  const results = new ResultsScreen(resultsRoot)
  mountSoundToggle(sound)

  // --- camera looks: face lenses + color filters (need landmarks) ---
  const landmarks = new MediaPipeLandmarkProvider(video)
  const colorRenderer = createColorFilterRenderer(filterCanvas, video)
  colorRenderer?.startLoop()
  const lensRenderer = new LensRenderer(lensCanvas, video, landmarks)
  lensRenderer.start()
  const looks = new LookController(video, colorRenderer, lensRenderer)

  // --- the live loop: the player signs for real, the opponent is mocked ---
  const captureUI = new CaptureUI(captureRoot, sound)
  const capture = new SignCapture(store, landmarks, captureUI)
  capture.start() // idles until the race actually begins
  captureRoot.classList.add('hidden')

  const driver = new MockDriver(store)

  // --- solo Practice room + first-run tutorial ---
  const coach = new CoachOverlay(coachRoot)
  const runCoach = (): void => coach.start(COACH_STEPS, () => localStorage.setItem(COACH_KEY, '1'))

  const startPractice = async (): Promise<void> => {
    await sound.resume()
    picker.hide()
    store.startPractice()
    if (!localStorage.getItem(COACH_KEY)) runCoach()
  }
  const toLobby = (): void => {
    store.endPractice()
    picker.show()
  }
  const practiceBar = new PracticeBar(practiceRoot, {
    onDone: toLobby,
    onLooks: () => picker.toggle(),
    onHelp: runCoach,
  })

  // Match-level audio + capture/practice visibility follow the phase machine.
  let prevCountdown = -1
  store.on('change', (s) => {
    if (s.phase === 'countdown' && s.countdown !== prevCountdown && s.countdown > 0) {
      sound.countdown(s.countdown)
    }
    prevCountdown = s.countdown
  })
  store.on('phase', (phase) => {
    // The capture loop + feedback run in both a match and Practice.
    captureRoot.classList.toggle('hidden', phase !== 'racing' && phase !== 'practice')
    practiceBar.setVisible(phase === 'practice')
    if (phase === 'racing') {
      sound.go()
      sound.startMusic()
    }
  })
  store.on('finished', () => {
    driver.stop()
    sound.stopHold() // belt-and-suspenders: never let the hold tone leak
    sound.stopMusic()
    store.getState().winner === 'me' ? sound.win() : sound.lose()
    void results.show(store, MATCH_ID)
  })

  const startMatch = async (): Promise<void> => {
    await sound.resume() // the Start click is our audio-unlock gesture
    picker.hide()
    store.beginMatch() // fresh session so practice never leaks into the match
    await store.runCountdown(3)
    driver.start()
  }

  const picker = new LookPicker(pickerRoot, looks, {
    onStart: () => void startMatch(),
    onPractice: () => void startPractice(),
  })

  // MediaPipe models load from CDN; if they fail (offline), lenses won't anchor
  // and hand capture won't fire — color filters and the rest still work.
  landmarks.start().catch((err) => console.warn('Landmark provider unavailable:', err))
}

/** Small persistent sound on/off toggle in the corner. */
function mountSoundToggle(sound: SoundEngine): void {
  const btn = document.createElement('button')
  btn.className = 'sound-toggle'
  btn.type = 'button'
  btn.setAttribute('aria-label', 'Toggle sound')
  const render = (): void => {
    btn.textContent = sound.isEnabled() ? '🔊' : '🔇'
  }
  render()
  btn.addEventListener('click', () => {
    const on = !sound.isEnabled()
    sound.setEnabled(on)
    if (on) void sound.resume()
    render()
  })
  document.body.append(btn)
}

void main()
