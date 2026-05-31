import type { Lobby, NetClient, NetEvent, NetListener, OpponentView } from './protocol.ts'

/**
 * Local simulation of the backend so the whole lobby/matchmaking flow is
 * demoable with no server. It fakes a single opponent ("Rival") who joins
 * private lobbies, readies up, and gets matched in the queue. Timing is tuned to
 * feel live. The in-match opponent is still driven by {@link MockDriver}; this
 * only covers the pre-match protocol.
 *
 * Swap for {@link WebSocketNetClient} (same interface) to talk to the real
 * server — nothing in the UI changes.
 */

const CODE_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
const RIVAL: OpponentView = { playerId: 2, displayName: 'Rival', elo: 994 }
/** Single-token glosses (== clip slugs) for the offline practice stub. */
const PRACTICE_GLOSSES = ['hello', 'thanks', 'please', 'yes', 'no', 'name', 'help', 'good']

export class MockNetClient implements NetClient {
  playerId: number | null = null
  elo: number | null = null
  displayName = 'You'

  private listeners = new Set<NetListener>()
  private lobby: Lobby | null = null
  private queuePosition = 0
  private timers = new Set<number>()
  private starting = false
  private practiceWordIndex = 0

  async connect(displayName: string): Promise<void> {
    this.displayName = displayName || 'You'
    this.playerId = 1
    this.elo = 1000
    this.after(0, () => this.emit({ type: 'authOk', playerId: 1, elo: 1000 }))
  }

  disconnect(): void {
    this.clearTimers()
    this.lobby = null
    this.queuePosition = 0
  }

  // --- matchmaking ----------------------------------------------------------

  joinQueue(): void {
    this.reset()
    this.queuePosition = 1
    this.emit({ type: 'queueStatus', position: 1 })
    // A rival is "found" after a short search.
    this.after(2600, () => {
      this.queuePosition = 0
      this.lobby = this.fullLobby(this.me(), { ...this.member(RIVAL), ready: true })
      this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
    })
  }

  leaveQueue(): void {
    this.clearTimers()
    this.queuePosition = 0
  }

  // --- lobbies --------------------------------------------------------------

  createLobby(): void {
    this.reset()
    this.lobby = { code: this.code(), state: 'waiting', members: [this.me()] }
    this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
    // Rival joins, then readies up.
    this.after(1900, () => {
      this.lobby = this.fullLobby(this.me(), this.member(RIVAL))
      this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
      this.after(1300, () => this.setRivalReady())
    })
  }

  joinLobby(code: string): void {
    this.reset()
    // The rival is already the waiting host; we join to fill it.
    this.lobby = this.fullLobby(this.member(RIVAL), this.me(), code.toUpperCase())
    this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
    this.after(1000, () => this.setRivalReady())
  }

  setReady(ready: boolean): void {
    if (!this.lobby) return
    const me = this.lobby.members.find((m) => m.playerId === this.playerId)
    if (me) me.ready = ready
    this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
    this.maybeStart()
  }

  leaveLobby(): void {
    this.clearTimers()
    this.lobby = null
    this.starting = false
  }

  sendLandmark(): void {
    /* mock: the local SignCapture + MockDriver own the match offline */
  }

  // --- practice (offline stub) ----------------------------------------------
  // No real recognizer offline, but we mimic the server's practice stream so the
  // same PracticeDriver path type-checks and runs: emit practice.start, then walk
  // a small set of single-token glosses via the existing recognitionUpdate event
  // (rising wordIndex = a confirmed sign), advancing on a timer.

  startPractice(): void {
    this.clearTimers()
    this.practiceWordIndex = 0
    this.after(0, () => this.emit({ type: 'practiceStart', wordSeed: Math.floor(Math.random() * 1e9), datasetVersion: 'mock' }))
    this.after(200, () => this.tickPractice())
  }

  stopPractice(): void {
    this.clearTimers()
  }

  private tickPractice(): void {
    const word = PRACTICE_GLOSSES[this.practiceWordIndex % PRACTICE_GLOSSES.length]
    // Ramp strength up to ~0.9 then advance to the next word (rising index).
    let strength = 0
    const ramp = (): void => {
      strength = Math.min(0.95, strength + 0.18)
      this.emit({ type: 'recognitionUpdate', wordIndex: this.practiceWordIndex, word, strength, difficulty: 1 })
      if (strength >= 0.9) {
        this.after(700, () => {
          this.practiceWordIndex += 1
          this.tickPractice()
        })
      } else {
        this.after(130, ramp)
      }
    }
    ramp()
  }

  // --- subscription ---------------------------------------------------------

  on(listener: NetListener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getLobby(): Lobby | null {
    return this.lobby
  }
  getQueuePosition(): number {
    return this.queuePosition
  }

  // --- internals ------------------------------------------------------------

  private emit(event: NetEvent): void {
    for (const l of [...this.listeners]) l(event)
  }

  private setRivalReady(): void {
    if (!this.lobby) return
    const rival = this.lobby.members.find((m) => m.playerId === RIVAL.playerId)
    if (rival) rival.ready = true
    this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
    this.maybeStart()
  }

  private maybeStart(): void {
    const lobby = this.lobby
    if (!lobby || this.starting) return
    if (lobby.state !== 'full' || !lobby.members.every((m) => m.ready)) return
    this.starting = true
    const matchId = `mock-${this.code()}`
    const wordSeed = Math.floor(Math.random() * 1e9)
    this.after(500, () => {
      this.emit({ type: 'matchFound', matchId, role: 'offerer', opponent: RIVAL })
      this.after(650, () => {
        this.emit({ type: 'warmupStart', wordSeed, datasetVersion: 'mock' })
        this.after(1600, () => this.emit({ type: 'matchStart', matchId, wordSeed, recordStartMs: 0 }))
      })
    })
  }

  private reset(): void {
    this.clearTimers()
    this.lobby = null
    this.queuePosition = 0
    this.starting = false
  }

  private me(): Lobby['members'][number] {
    return { playerId: this.playerId ?? 1, displayName: this.displayName, ready: false, connected: true }
  }
  private member(v: OpponentView): Lobby['members'][number] {
    return { playerId: v.playerId, displayName: v.displayName, ready: false, connected: true }
  }
  private fullLobby(a: Lobby['members'][number], b: Lobby['members'][number], code = this.code()): Lobby {
    return { code, state: 'full', members: [a, b] }
  }

  private code(): string {
    let c = ''
    for (let i = 0; i < 6; i++) c += CODE_ALPHABET[Math.floor(Math.random() * CODE_ALPHABET.length)]
    return c
  }

  private after(ms: number, fn: () => void): void {
    const id = window.setTimeout(() => {
      this.timers.delete(id)
      fn()
    }, ms)
    this.timers.add(id)
  }
  private clearTimers(): void {
    for (const id of this.timers) clearTimeout(id)
    this.timers.clear()
  }
}
