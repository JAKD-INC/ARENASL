import type { GameStore } from './store.ts'
import type { VerdictTier } from './signCapture.ts'
import type { CaptureUI } from '../ui/capture.ts'
import type { NetClient, NetEvent } from '../net/protocol.ts'
import type { RecognitionSource } from './netMatch.ts'

/**
 * Drives **server-backed Practice** — the solo sibling of {@link NetMatchDriver}.
 * It uses the real server recognizer over a DEDICATED practice stream (NOT
 * matchmaking): {@link NetClient.startPractice} opens it, landmark frames stream
 * up at ~15fps, and the server replies with the SAME `recognition.update`
 * messages a match/warmup uses.
 *
 *  - `recognition.update {word, strength}` → the current word ({@link GameStore.setNetWord},
 *    which the HUD turns into the per-gloss example clip) + the hold ring.
 *  - a rising `word_index` → the server confirmed a sign → verdict + score
 *    ({@link GameStore.netConfirm}).
 *
 * Unlike the match driver there is NO HP, NO match.over, and no elapsed clock —
 * Practice is stakes-free. {@link stop} (via {@link NetClient.stopPractice})
 * tears the stream down.
 */

const SEND_INTERVAL_MS = 66 // ~15fps, matching the server's per-player budget

export class PracticeDriver {
  private off: Array<() => void> = []
  private rafId = 0
  private lastSendMs = 0
  private lastWordIndex = -1
  private peakStrength = 0
  /** Gate: only stream landmarks once the server has acked practice.start. */
  private ready = false

  constructor(
    private store: GameStore,
    private net: NetClient,
    private source: RecognitionSource,
    private captureUI: CaptureUI,
    /** Optional debug sink for the on-screen readout. */
    private onDebug?: (line: string) => void,
  ) {}

  start(): void {
    this.lastWordIndex = -1
    this.peakStrength = 0
    this.ready = false
    this.off.push(this.net.on((e) => this.onEvent(e)))
    this.net.startPractice()
    const loop = (ts: number): void => {
      this.rafId = requestAnimationFrame(loop)
      this.stream(ts)
    }
    this.rafId = requestAnimationFrame(loop)
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.rafId = 0
    this.ready = false
    this.net.stopPractice()
    for (const off of this.off) off()
    this.off = []
  }

  private stream(ts: number): void {
    // Per the practice protocol the server sets status='practice' (and replies
    // practice.start) before it will process landmarks; hold frames until then so
    // none arrive early and get dropped.
    if (!this.ready) return
    if (ts - this.lastSendMs < SEND_INTERVAL_MS) return
    const frame = this.source.latestRecognition()
    if (!frame) return
    this.lastSendMs = ts
    this.net.sendLandmark(frame)
  }

  private onEvent(e: NetEvent): void {
    if (e.type === 'practiceStart') {
      this.ready = true
      return
    }
    if (e.type !== 'recognitionUpdate') return
    // The server advances its recognizer by one word per confirmed sign, so a
    // rising index = a confirmation. Treat ANY increase (not strictly +1) as a
    // confirmation so a dropped/throttled frame can't silently swallow one.
    const confirmed = this.lastWordIndex >= 0 && e.wordIndex > this.lastWordIndex
    if (confirmed) {
      const outcome = this.store.netConfirm(this.peakStrength)
      this.captureUI.onResolved(outcome, tierFor(this.peakStrength))
      this.peakStrength = 0
    }
    this.lastWordIndex = e.wordIndex
    this.peakStrength = Math.max(this.peakStrength, e.strength)
    this.store.setNetWord(e.wordIndex, e.word, e.difficulty)
    if (!confirmed) this.captureUI.online(e.strength)
    this.onDebug?.(`practice w#${e.wordIndex} "${e.word}" d=${e.difficulty} strength=${e.strength.toFixed(2)}`)
  }
}

function tierFor(strength: number): VerdictTier {
  if (strength >= 0.9) return 'perfect'
  if (strength >= 0.78) return 'great'
  return 'good'
}
