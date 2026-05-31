import type { LandmarkProvider, SignOutcome } from './types.ts'
import type { GameStore } from './store.ts'

/**
 * Turns the player's actual hand activity into {@link SignOutcome}s — this is
 * what makes the camera the controller instead of wallpaper.
 *
 * It is a deliberately honest *heuristic* stand-in for real ASL recognition
 * (Alex's workstream), driving the exact same {@link GameStore.submitSign} seam
 * his recognizer will: raise your hands into the signing space, perform while a
 * hold window fills, and the attempt resolves to an accuracy the store scores.
 * Accuracy rewards what a sign actually looks like — hands kept in frame and
 * moving — so doing nothing misses and signing energetically lands. Swap this
 * out for the real recognizer and the whole UI/feedback loop keeps working.
 *
 * Lifecycle callbacks let the capture UI draw the prompt, the hold ring, and the
 * verdict, and let the sound engine react.
 */

export type VerdictTier = 'perfect' | 'great' | 'good' | 'miss'

export interface CaptureCallbacks {
  /** Racing began or a previous attempt finished — ready for the next sign. */
  onWaiting?: () => void
  /** Hands raised; a capture has begun. */
  onAttemptStart?: () => void
  /** Hold progress 0..1 while the player performs. */
  onProgress?: (p: number) => void
  /** Attempt resolved into an outcome + a display tier. */
  onResolved?: (outcome: SignOutcome, tier: VerdictTier) => void
}

type State = 'waiting' | 'holding' | 'verdict'

interface Pt {
  x: number
  y: number
}

// Tuning.
const HOLD_MS = 1100 // how long a sign must be performed
const VERDICT_MS = 750 // how long the verdict shows before the next sign
const RAISED_Y = 0.78 // wrist must be above this (upper frame) to count as "signing"
const START_FRAMES = 3 // consecutive raised frames to begin an attempt
const DROP_FRAMES = 9 // consecutive lowered frames that abort an attempt

export class SignCapture {
  private store: GameStore
  private provider: LandmarkProvider
  private cb: CaptureCallbacks

  private state: State = 'waiting'
  private rafId = 0
  private raisedStreak = 0
  private droppedStreak = 0

  private holdStartMs = 0
  private verdictStartMs = 0
  private motionEnergy = 0
  private visibleFrames = 0
  private totalFrames = 0
  private lastCentroid: Pt | null = null

  private currentWordId: string | null = null
  private wordShownMs = 0

  constructor(store: GameStore, provider: LandmarkProvider, cb: CaptureCallbacks = {}) {
    this.store = store
    this.provider = provider
    this.cb = cb
  }

  start(): void {
    if (this.rafId) return
    const loop = (): void => {
      this.rafId = requestAnimationFrame(loop)
      this.tick()
    }
    loop()
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  private tick(): void {
    const now = performance.now()
    const s = this.store.getState()
    // The same capture loop powers both a real match and Practice.
    if (s.phase !== 'racing' && s.phase !== 'practice') {
      // Releasing mid-sign (e.g. the match just ended) must fire onWaiting so
      // the UI/sound stop the hold tone — otherwise it drones into the results.
      if (this.state === 'holding') this.enterWaiting()
      this.state = 'waiting'
      this.currentWordId = null
      return
    }

    // Track when the current word first appeared (for the speed bonus).
    if (s.currentWord.id !== this.currentWordId) {
      this.currentWordId = s.currentWord.id
      this.wordShownMs = now
      if (this.state !== 'verdict') this.enterWaiting()
    }

    const hands = this.provider.latest()?.hands ?? []
    const signing = isSigning(hands)

    switch (this.state) {
      case 'waiting': {
        this.raisedStreak = signing ? this.raisedStreak + 1 : 0
        if (this.raisedStreak >= START_FRAMES) this.beginHold(now)
        break
      }
      case 'holding': {
        this.totalFrames++
        if (signing) {
          this.visibleFrames++
          this.droppedStreak = 0
          const c = centroid(hands)
          if (c && this.lastCentroid) this.motionEnergy += dist(c, this.lastCentroid)
          this.lastCentroid = c
        } else {
          this.droppedStreak++
          if (this.droppedStreak >= DROP_FRAMES) {
            // Player pulled their hands down — abort with no penalty.
            this.enterWaiting()
            break
          }
        }
        const progress = Math.min(1, (now - this.holdStartMs) / HOLD_MS)
        this.cb.onProgress?.(progress)
        if (progress >= 1) this.resolve(now)
        break
      }
      case 'verdict': {
        if (now - this.verdictStartMs < VERDICT_MS) break
        // Hands still up → straight into the next hold (skip the prompt flash).
        if (signing) this.beginHold(now)
        else this.enterWaiting()
        break
      }
    }
  }

  private enterWaiting(): void {
    this.state = 'waiting'
    this.raisedStreak = 0
    this.cb.onWaiting?.()
  }

  private beginHold(now: number): void {
    this.state = 'holding'
    this.holdStartMs = now
    this.motionEnergy = 0
    this.visibleFrames = 0
    this.totalFrames = 0
    this.droppedStreak = 0
    this.lastCentroid = null
    this.cb.onAttemptStart?.()
  }

  private resolve(now: number): void {
    const visibility = this.visibleFrames / Math.max(1, this.totalFrames)
    const motionAvg = this.motionEnergy / Math.max(1, this.totalFrames)

    // Base for keeping hands up through the window, bonus for real movement.
    let accuracy = 0.6 + Math.min(0.34, motionAvg * 9)
    // Hands drifting out of frame tank the score (teaches "stay in frame").
    if (visibility < 0.6) accuracy *= visibility / 0.6
    accuracy += (Math.random() - 0.5) * 0.06 // a little life
    accuracy = Math.max(0, Math.min(0.99, accuracy))

    const timeMs = now - this.wordShownMs
    const outcome = this.store.submitSign({
      player: 'me',
      wordId: this.store.getState().currentWord.id,
      accuracy,
      timeMs,
    })

    this.state = 'verdict'
    this.verdictStartMs = now
    this.cb.onResolved?.(outcome, tierFor(outcome))
  }
}

function tierFor(o: SignOutcome): VerdictTier {
  if (!o.accepted) return 'miss'
  const a = o.result.accuracy
  if (a >= 0.9) return 'perfect'
  if (a >= 0.78) return 'great'
  return 'good'
}

/** At least one hand raised into the signing space (upper part of frame). */
function isSigning(hands: Pt[][]): boolean {
  for (const hand of hands) {
    const wrist = hand[0]
    if (wrist && wrist.y < RAISED_Y) return true
  }
  return false
}

function centroid(hands: Pt[][]): Pt | null {
  let x = 0
  let y = 0
  let n = 0
  for (const hand of hands) {
    for (const p of hand) {
      x += p.x
      y += p.y
      n++
    }
  }
  return n ? { x: x / n, y: y / n } : null
}

function dist(a: Pt, b: Pt): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}
