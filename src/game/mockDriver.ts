import type { GameStore } from './store.ts'
import { setMockReplay } from './replay.ts'

/**
 * Drives the **opponent** only — the local player now signs for real through
 * {@link SignCapture}. This stands in for the remote peer (David's workstream):
 * synthetic opponent signs land on a jittered cadence so the tug-of-war has
 * live pressure and the match can end when someone's HP hits 0 (the store owns
 * that condition). Also ticks the elapsed-time clock for the HUD.
 *
 * The opponent is tuned a touch slower than a focused human so an engaged player
 * can win — tweak {@link MockDriverOptions.oppAttemptMs} to rebalance.
 */

export interface MockDriverOptions {
  /** Mean delay between the opponent's sign attempts, ms. */
  oppAttemptMs?: number
}

export class MockDriver {
  private store: GameStore
  private oppAttemptMs: number
  private oppTimer = 0
  private rafId = 0
  private stopped = false

  constructor(store: GameStore, opts: MockDriverOptions = {}) {
    this.store = store
    this.oppAttemptMs = opts.oppAttemptMs ?? 1700
  }

  start(): void {
    this.stopped = false
    // No real recording in dev; the replay video is unavailable (server-owned).
    setMockReplay({ videoUrl: null, timeline: [] })
    this.tickClock()
    this.scheduleOpp()
  }

  stop(): void {
    this.stopped = true
    clearTimeout(this.oppTimer)
    cancelAnimationFrame(this.rafId)
  }

  /** Drive the elapsed-time clock for the HUD. */
  private tickClock = (): void => {
    if (this.stopped) return
    this.store.tick()
    this.rafId = requestAnimationFrame(this.tickClock)
  }

  private scheduleOpp(): void {
    this.oppTimer = window.setTimeout(() => this.attempt(), jitter(this.oppAttemptMs))
  }

  private attempt(): void {
    if (this.stopped) return
    const state = this.store.getState()
    if (state.phase !== 'racing') {
      if (state.phase === 'countdown') this.scheduleOpp()
      return
    }

    // ~80% land cleanly, ~20% are misses.
    const miss = Math.random() < 0.2
    const accuracy = miss ? 0.2 + Math.random() * 0.35 : 0.62 + Math.random() * 0.38
    const timeMs = 900 + Math.random() * 2600

    this.store.submitSign({ player: 'opponent', wordId: '', accuracy, timeMs })

    if (this.stopped || this.store.getState().phase !== 'racing') return
    this.scheduleOpp()
  }
}

/** Randomize a delay around its mean so attempts don't lock-step. */
function jitter(meanMs: number): number {
  return meanMs * (0.6 + Math.random() * 0.8)
}
