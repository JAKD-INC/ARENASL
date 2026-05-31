import type { GameStore } from './store.ts'
import type { VerdictTier } from './signCapture.ts'
import type { CaptureUI } from '../ui/capture.ts'
import type { LandmarkPayload, NetClient, NetEvent } from '../net/protocol.ts'

/**
 * Drives an **online** duel: streams the raw landmark frames up to the server
 * (which owns recognition), and turns the server's responses back into the same
 * UI the offline loop uses.
 *
 *  - `recognition.update {word, strength}` → the current word + the hold ring.
 *  - a rising `word_index` → the server confirmed a sign → verdict + score.
 *  - `match.state` → server-authoritative HP (revealed on the results screen).
 *  - `match.over` → winner + result.
 *
 * Replaces both the local heuristic {@link SignCapture} and the {@link MockDriver}
 * for networked matches; the store becomes a view of server state.
 */
export interface RecognitionSource {
  latestRecognition(): LandmarkPayload | null
}

const SEND_INTERVAL_MS = 66 // ~15fps, matching the server's per-player budget

export class NetMatchDriver {
  private off: Array<() => void> = []
  private rafId = 0
  private lastSendMs = 0
  private lastWordIndex = -1
  private peakStrength = 0
  private myId = -1

  constructor(
    private store: GameStore,
    private net: NetClient,
    private source: RecognitionSource,
    private captureUI: CaptureUI,
    /** Optional debug sink for the on-screen readout. */
    private onDebug?: (line: string) => void,
  ) {}

  start(myId: number): void {
    this.myId = myId
    this.lastWordIndex = -1
    this.peakStrength = 0
    this.off.push(this.net.on((e) => this.onEvent(e)))
    const loop = (ts: number): void => {
      this.rafId = requestAnimationFrame(loop)
      this.store.tick() // drive the elapsed-time clock (MockDriver does this offline)
      this.stream(ts)
    }
    this.rafId = requestAnimationFrame(loop)
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.rafId = 0
    for (const off of this.off) off()
    this.off = []
  }

  private stream(ts: number): void {
    if (ts - this.lastSendMs < SEND_INTERVAL_MS) return
    const frame = this.source.latestRecognition()
    if (!frame) return
    this.lastSendMs = ts
    this.net.sendLandmark(frame)
  }

  private onEvent(e: NetEvent): void {
    switch (e.type) {
      case 'recognitionUpdate': {
        const confirmed = this.lastWordIndex >= 0 && e.wordIndex > this.lastWordIndex
        if (confirmed) {
          const outcome = this.store.netConfirm(this.peakStrength)
          this.captureUI.onResolved(outcome, tierFor(this.peakStrength))
          this.peakStrength = 0
        }
        this.lastWordIndex = e.wordIndex
        this.peakStrength = Math.max(this.peakStrength, e.strength)
        this.store.setNetWord(e.wordIndex, e.word)
        if (!confirmed) this.captureUI.online(e.strength)
        this.onDebug?.(`reco w#${e.wordIndex} "${e.word}" strength=${e.strength.toFixed(2)}`)
        break
      }
      case 'matchState': {
        const me = e.players.find((p) => p.playerId === this.myId)
        const opp = e.players.find((p) => p.playerId !== this.myId)
        if (me && opp) {
          this.store.setNetHp(me.hp, opp.hp)
          this.onDebug?.(`state me.hp=${me.hp.toFixed(0)} opp.hp=${opp.hp.toFixed(0)}`)
        }
        break
      }
      case 'matchOver': {
        this.store.netFinish(e.winnerId === this.myId ? 'me' : 'opponent')
        this.onDebug?.(`over winner=${e.winnerId} elo${e.eloDelta != null ? ` Δ${e.eloDelta}` : ''}`)
        break
      }
      case 'opponentStatus':
        this.onDebug?.(`opponent ${e.connected ? 'connected' : 'disconnected'}`)
        break
    }
  }
}

function tierFor(strength: number): VerdictTier {
  if (strength >= 0.9) return 'perfect'
  if (strength >= 0.78) return 'great'
  return 'good'
}
