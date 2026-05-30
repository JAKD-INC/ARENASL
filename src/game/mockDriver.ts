import type { GameStore } from './store.ts'
import { setMockReplay } from './replay.ts'

/**
 * Stand-in for the recognition layer (Alex) + opponent feed (David), so the
 * whole overlay runs on one machine with no backend.
 *
 * Both players sign **live** during the race: synthetic signs are fed to the
 * store for `me` and `opponent` on independent cadences, so the tug-of-war
 * actually moves in real time and the match ends when someone's HP hits 0 (the
 * store owns that condition). No word-count cutoff.
 */

export interface MockDriverOptions {
  /** Mean delay between the local player's sign attempts, ms. */
  myAttemptMs?: number
  /** Mean delay between the opponent's sign attempts, ms. */
  oppAttemptMs?: number
}

export class MockDriver {
  private store: GameStore
  private myAttemptMs: number
  private oppAttemptMs: number
  private myTimer = 0
  private oppTimer = 0
  private rafId = 0
  private stopped = false

  constructor(store: GameStore, opts: MockDriverOptions = {}) {
    this.store = store
    this.myAttemptMs = opts.myAttemptMs ?? 1100
    this.oppAttemptMs = opts.oppAttemptMs ?? 1250
  }

  start(): void {
    this.stopped = false
    // No real recording in dev; the replay video is unavailable (server-owned).
    setMockReplay({ videoUrl: null, timeline: [] })
    this.tickClock()
    this.scheduleMine()
    this.scheduleOpp()
  }

  stop(): void {
    this.stopped = true
    clearTimeout(this.myTimer)
    clearTimeout(this.oppTimer)
    cancelAnimationFrame(this.rafId)
  }

  /** Drive the elapsed-time clock for the HUD. */
  private tickClock = (): void => {
    if (this.stopped) return
    this.store.tick()
    this.rafId = requestAnimationFrame(this.tickClock)
  }

  private scheduleMine(): void {
    this.myTimer = window.setTimeout(() => this.attempt('me'), jitter(this.myAttemptMs))
  }

  private scheduleOpp(): void {
    this.oppTimer = window.setTimeout(() => this.attempt('opponent'), jitter(this.oppAttemptMs))
  }

  private attempt(player: 'me' | 'opponent'): void {
    if (this.stopped) return
    const state = this.store.getState()
    if (state.phase !== 'racing') {
      // Pause attempts until the race actually starts; the match may also be
      // over, in which case the store will have flipped phase and stopped us.
      if (state.phase === 'countdown') {
        player === 'me' ? this.scheduleMine() : this.scheduleOpp()
      }
      return
    }

    // ~80% land cleanly, ~20% are misses.
    const miss = Math.random() < 0.2
    const accuracy = miss ? 0.2 + Math.random() * 0.35 : 0.62 + Math.random() * 0.38
    const timeMs = 900 + Math.random() * 2600

    this.store.submitSign({
      player,
      wordId: player === 'me' ? state.currentWord.id : '',
      accuracy,
      timeMs,
    })

    if (this.stopped || this.store.getState().phase !== 'racing') return
    player === 'me' ? this.scheduleMine() : this.scheduleOpp()
  }
}

/** Randomize a delay around its mean so attempts don't lock-step. */
function jitter(meanMs: number): number {
  return meanMs * (0.6 + Math.random() * 0.8)
}
