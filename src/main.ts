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
import { Router } from './app/router.ts'
import { createNetClient } from './net/wsClient.ts'
import { MockNetClient } from './net/mockClient.ts'
import { ensureAuth } from './net/auth.ts'
import type { OpponentView } from './net/protocol.ts'
import { TitleScreen } from './ui/screens/title.ts'
import { NameEntryScreen } from './ui/screens/nameEntry.ts'
import { ModeSelectScreen } from './ui/screens/modeSelect.ts'
import { JoinCodeScreen } from './ui/screens/joinCode.ts'
import { LobbyRoomScreen } from './ui/screens/lobbyRoom.ts'
import { FindRivalScreen } from './ui/screens/findRival.ts'
import { WarmupScreen } from './ui/screens/warmup.ts'

const video = document.querySelector<HTMLVideoElement>('#feed')!
const message = document.querySelector<HTMLDivElement>('#message')!
const filterCanvas = document.querySelector<HTMLCanvasElement>('#filter')!
const lensCanvas = document.querySelector<HTMLCanvasElement>('#lens')!
const vfxCanvas = document.querySelector<HTMLCanvasElement>('#vfx')!
const captureRoot = document.querySelector<HTMLDivElement>('#capture')!
const hudRoot = document.querySelector<HTMLDivElement>('#hud')!
const screensRoot = document.querySelector<HTMLDivElement>('#screens')!
const practiceRoot = document.querySelector<HTMLDivElement>('#practicebar')!
const coachRoot = document.querySelector<HTMLDivElement>('#coach')!
const resultsRoot = document.querySelector<HTMLDivElement>('#results')!

const MATCH_ID = 'dev-match'
const NAME_KEY = 'arenasl.name'

const COACH_STEPS: CoachStep[] = [
  { title: 'Welcome to ArenaSL 👋', body: 'Learn ASL by signing — let’s warm up. No opponent, no pressure here.' },
  { target: '.hud-word', title: 'Copy this sign', body: 'Watch the looping demo, then make the same sign with your hands.' },
  { target: '.cap-stage', title: 'Raise your hands', body: 'Lift your hands into view and hold the sign until the ring fills.' },
  { title: 'Score big', body: 'A clean sign scores PERFECT and builds your combo. Miss? Just try again — nothing to lose.' },
  { title: 'Ready to battle?', body: 'Leave Practice and hit Play to face a rival. Your HP stays hidden until the big reveal!' },
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
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      showMessage('Camera access denied. Please allow camera permission and reload.')
    } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      showMessage('No camera found.')
    } else {
      showMessage(`Could not start camera${name ? ` (${name})` : ''}.`)
    }
    return
  }

  let displayName = localStorage.getItem(NAME_KEY) ?? ''

  // --- persistent services (live across every screen) ---
  const store = new GameStore({ seed: 1, myName: displayName || 'You', oppName: 'Rival' })
  const sound = new SoundEngine()
  new Hud(hudRoot, store)
  new Vfx(vfxCanvas, store)
  mountSoundToggle(sound)

  const landmarks = new MediaPipeLandmarkProvider(video)
  const colorRenderer = createColorFilterRenderer(filterCanvas, video)
  colorRenderer?.startLoop()
  const lensRenderer = new LensRenderer(lensCanvas, video, landmarks)
  lensRenderer.start()
  const looks = new LookController(video, colorRenderer, lensRenderer)

  const captureUI = new CaptureUI(captureRoot, sound)
  const capture = new SignCapture(store, landmarks, captureUI)
  capture.start() // idles until a match/practice begins
  captureRoot.classList.add('hidden')

  const driver = new MockDriver(store)
  let net = createNetClient()
  const router = new Router(screensRoot)
  const coach = new CoachOverlay(coachRoot)

  const practiceBar = new PracticeBar(practiceRoot, {
    onDone: () => exitPractice(),
    onHelp: () => runCoach(),
  })
  const results = new ResultsScreen(resultsRoot, { onHome: () => goTitle(), onRematch: () => goMode() })

  // --- navigation (hoisted; reference the services above) ---
  function goTitle(): void {
    router.show(new TitleScreen(displayName || 'You', { onPlay: goMode, onPractice: () => void enterPractice(), onChangeName: goName }))
  }
  function goName(): void {
    router.show(
      new NameEntryScreen(displayName, {
        onSubmit: (n) => {
          displayName = n
          localStorage.setItem(NAME_KEY, n)
          store.setMyName(n)
          goTitle()
        },
      }),
    )
  }
  function goMode(): void {
    // Intended leaves are explicit (lobby Leave / find Cancel); don't disconnect here.
    router.show(new ModeSelectScreen({ onCreate, onJoin: goJoin, onFind, onBack: goTitle }))
  }
  function goJoin(): void {
    router.show(new JoinCodeScreen(net, { onJoined: goLobby, onBack: goMode }))
  }
  function onCreate(): void {
    net.createLobby()
    goLobby()
  }
  function goLobby(): void {
    router.show(new LobbyRoomScreen(net, looks, { onLeave: () => { net.leaveLobby(); goMode() } }))
  }
  function onFind(): void {
    net.joinQueue()
    router.show(new FindRivalScreen(net, { onPaired: goLobby, onCancel: () => { net.leaveQueue(); goMode() } }))
  }
  function goWarmup(seed: number, opp: OpponentView): void {
    router.show(
      new WarmupScreen(
        { name: opp.displayName, elo: opp.elo },
        { name: displayName || 'You', elo: net.elo ?? undefined },
        { onDone: () => void startNetMatch(seed, opp) },
      ),
    )
  }
  async function startNetMatch(seed: number, opp: OpponentView): Promise<void> {
    router.show(null)
    await sound.resume()
    store.beginMatch(seed, opp.displayName)
    await store.runCountdown(3)
    driver.start()
  }
  async function enterPractice(): Promise<void> {
    router.show(null)
    await sound.resume()
    store.startPractice()
    runCoach() // the tutorial is the practice — always walk through it
  }
  function exitPractice(): void {
    store.endPractice()
    goTitle()
  }
  function runCoach(): void {
    coach.start(COACH_STEPS)
  }

  // --- match start handshake (mock or real, same events) ---
  // Re-attachable because `net` can fall back to the mock client if the backend
  // is unreachable at boot.
  let pendingOpponent: OpponentView | null = null
  let unsubMatch: (() => void) | null = null
  function subscribeMatchEvents(): void {
    unsubMatch?.()
    unsubMatch = net.on((e) => {
      if (e.type === 'matchFound') pendingOpponent = e.opponent
      else if (e.type === 'warmupStart') {
        goWarmup(e.wordSeed, pendingOpponent ?? { playerId: 2, displayName: 'Rival', elo: 1000 })
      }
    })
  }

  /**
   * Authenticate (per-device account → JWT) and connect to the backend. If the
   * backend is unreachable, fall back to the offline mock so Practice and a
   * standalone demo still work.
   */
  async function connectNet(name: string): Promise<void> {
    if (!(net instanceof MockNetClient)) {
      try {
        const id = await ensureAuth(name)
        store.setMyName(id.displayName)
        await net.connect(id.displayName, id.token)
        subscribeMatchEvents()
        return
      } catch (err) {
        console.warn('Backend unavailable — running offline (mock):', err)
        net = new MockNetClient()
      }
    }
    await net.connect(name)
    subscribeMatchEvents()
  }

  // --- audio + capture/practice visibility follow the phase machine ---
  let prevCountdown = -1
  store.on('change', (s) => {
    if (s.phase === 'countdown' && s.countdown !== prevCountdown && s.countdown > 0) sound.countdown(s.countdown)
    prevCountdown = s.countdown
  })
  store.on('phase', (phase) => {
    captureRoot.classList.toggle('hidden', phase !== 'racing' && phase !== 'practice')
    practiceBar.setVisible(phase === 'practice')
    if (phase === 'racing') {
      sound.go()
      sound.startMusic()
    }
  })
  store.on('finished', () => {
    driver.stop()
    sound.stopHold()
    sound.stopMusic()
    store.getState().winner === 'me' ? sound.win() : sound.lose()
    void results.show(store, MATCH_ID)
  })

  // --- boot: name (first run) → auth + connect → title ---
  landmarks.start().catch((err) => console.warn('Landmark provider unavailable:', err))
  if (!displayName) {
    router.show(
      new NameEntryScreen('', {
        onSubmit: async (n) => {
          displayName = n
          localStorage.setItem(NAME_KEY, n)
          await connectNet(n)
          goTitle()
        },
      }),
    )
  } else {
    await connectNet(displayName)
    goTitle()
  }
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
