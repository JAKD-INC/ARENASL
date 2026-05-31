import type {
  GamePhase,
  GameState,
  PlayerId,
  PlayerState,
  SignOutcome,
  SignResult,
  TugStep,
  Word,
} from './types.ts'
import { WordStream } from './words.ts'
import { ACCEPT_THRESHOLD, MAX_HP, damage, damageRamp, points } from './scoring.ts'

/**
 * Central game store: holds {@link GameState}, runs the match finite-state
 * machine, and turns {@link SignResult}s (from the recognition layer or the mock
 * driver) into state changes the UI subscribes to.
 *
 * Tug-of-war (live): every accepted sign deals `complexity × accuracy` damage to
 * the other player and heals the signer the same amount, clamped to 0..MAX_HP.
 * That base hit is scaled by {@link damageRamp}, which grows exponentially with
 * match time, so hits get bigger the longer the match runs and HP is guaranteed
 * to reach 0 — the match ends deterministically with no fixed time cap. The
 * match ends the instant either player's HP hits 0 — HP is tracked here in real
 * time but, per "solo focus", is never shown during the race; it (and the
 * winner) are only revealed on the results screen.
 *
 * Both players race the *same* infinite word list from independent streams
 * seeded identically (David syncs the seed), so each advances at their own pace
 * through the same sequence.
 *
 * Events:
 *  - `change`   (GameState)   — any state mutation; HUD re-renders.
 *  - `phase`    (GamePhase)   — phase transition; layers mount/unmount.
 *  - `sign`     (SignOutcome) — a sign was applied; VFX reacts.
 *  - `finished` (void)        — HP hit 0; results screen mounts.
 */

type EventMap = {
  change: GameState
  phase: GamePhase
  sign: SignOutcome
  finished: void
}

type Listener<T> = (payload: T) => void

function newPlayer(id: PlayerId, name: string): PlayerState {
  return { id, name, hp: MAX_HP, score: 0, combo: 0 }
}

export class GameStore {
  private state: GameState
  /** Local player's word stream. */
  private stream: WordStream
  /** Opponent's word stream — same seed, advanced independently. */
  private oppStream: WordStream
  private oppCurrentWord: Word
  private oppCleared = 0
  /** Stakes-free word stream for Practice mode (biased to easy words). */
  private practiceStream: WordStream | null = null
  private seed: number
  private myName: string
  private oppName: string
  private listeners: { [K in keyof EventMap]: Set<Listener<EventMap[K]>> } = {
    change: new Set(),
    phase: new Set(),
    sign: new Set(),
    finished: new Set(),
  }

  /** Live tug-of-war history (one step per accepted sign) for the results recap. */
  readonly tugHistory: TugStep[] = []
  private startedAt = 0

  constructor(opts: { seed?: number; myName?: string; oppName?: string } = {}) {
    this.seed = opts.seed ?? 1
    this.myName = opts.myName ?? 'You'
    this.oppName = opts.oppName ?? 'Opponent'
    this.stream = new WordStream(this.seed)
    this.oppStream = new WordStream(this.seed)
    this.oppCurrentWord = this.oppStream.next()
    this.state = {
      phase: 'idle',
      players: {
        me: newPlayer('me', this.myName),
        opponent: newPlayer('opponent', this.oppName),
      },
      currentWord: this.stream.next(),
      cleared: 0,
      countdown: 3,
      elapsedMs: 0,
      winner: null,
    }
  }

  // --- session control ------------------------------------------------------

  /**
   * Reset to a fresh, stakes-free **Practice** session: easy words, no opponent,
   * no HP/win-loss. The capture loop and feedback run exactly as in a match, so
   * the player learns the controls safely. Both players race nothing here.
   */
  startPractice(maxDifficulty = 2): void {
    this.practiceStream = new WordStream(freshSeed(), maxDifficulty)
    this.state.players.me = newPlayer('me', this.myName)
    this.state.cleared = 0
    this.state.elapsedMs = 0
    this.state.winner = null
    this.state.currentWord = this.practiceStream.next()
    this.setPhase('practice')
  }

  /** Leave Practice and return to the idle lobby. */
  endPractice(): void {
    this.practiceStream = null
    this.setPhase('idle')
  }

  /**
   * Reset to a fresh **match** session so mode-switching (lobby ⇄ practice ⇄
   * match) never carries words/score over. Call before {@link runCountdown}.
   */
  beginMatch(seed: number = this.seed, opponentName: string = this.oppName): void {
    this.practiceStream = null
    this.oppName = opponentName
    this.stream = new WordStream(seed)
    this.oppStream = new WordStream(seed)
    this.oppCurrentWord = this.oppStream.next()
    this.oppCleared = 0
    this.tugHistory.length = 0
    this.startedAt = 0
    this.state.players.me = newPlayer('me', this.myName)
    this.state.players.opponent = newPlayer('opponent', this.oppName)
    this.state.currentWord = this.stream.next()
    this.state.cleared = 0
    this.state.elapsedMs = 0
    this.state.countdown = 3
    this.state.winner = null
    this.setPhase('idle')
  }

  // --- subscription ---------------------------------------------------------

  on<K extends keyof EventMap>(event: K, fn: Listener<EventMap[K]>): () => void {
    this.listeners[event].add(fn as Listener<EventMap[keyof EventMap]>)
    return () => this.listeners[event].delete(fn as Listener<EventMap[keyof EventMap]>)
  }

  private emit<K extends keyof EventMap>(event: K, payload: EventMap[K]): void {
    for (const fn of this.listeners[event]) fn(payload)
  }

  getState(): GameState {
    return this.state
  }

  /** Update the local player's display name (reflected on the results screen). */
  setMyName(name: string): void {
    this.myName = name
    this.state.players.me.name = name
    this.emit('change', this.state)
  }

  /** Words the opponent cleared — for the results summary. */
  opponentCleared(): number {
    return this.oppCleared
  }

  // --- phase machine --------------------------------------------------------

  private setPhase(phase: GamePhase): void {
    this.state.phase = phase
    this.emit('phase', phase)
    this.emit('change', this.state)
  }

  /** idle → countdown. Returns once the countdown reaches 0 (racing begins). */
  async runCountdown(seconds = 3, tickMs = 1000): Promise<void> {
    this.state.countdown = seconds
    this.setPhase('countdown')
    while (this.state.countdown > 0) {
      await delay(tickMs)
      this.state.countdown -= 1
      this.emit('change', this.state)
    }
    this.startRacing()
  }

  private startRacing(): void {
    this.startedAt = nowMs()
    this.state.elapsedMs = 0
    this.setPhase('racing')
  }

  /**
   * Advance the elapsed clock (called by the driver's frame loop). There is no
   * time cap — the damage ramp ({@link damageRamp}) guarantees the match ends.
   */
  tick(): void {
    if (this.state.phase !== 'racing') return
    this.state.elapsedMs = nowMs() - this.startedAt
    this.emit('change', this.state)
  }

  // --- gameplay -------------------------------------------------------------

  /**
   * Apply a recognized sign. This is the seam the recognition layer (Alex)
   * calls; the mock driver calls it for both players. Each accepted sign moves
   * the tug-of-war live and can end the match.
   */
  submitSign(result: SignResult): SignOutcome {
    const isMe = result.player === 'me'
    const word = isMe ? this.state.currentWord : this.oppCurrentWord
    const player = this.state.players[result.player]
    const practice = this.state.phase === 'practice'
    const live = this.state.phase === 'racing' || practice

    // Stale/early signs don't count. The local player's word id must match
    // (guards recognition races); opponent signs are validated server-side.
    const accepted =
      live && result.accuracy >= ACCEPT_THRESHOLD && (!isMe || result.wordId === word.id)

    let dmg = 0
    let pts = 0
    if (accepted) {
      player.combo += 1
      pts = points(result.accuracy, result.timeMs, player.combo)
      player.score += pts
      const base = damage(word.difficulty, result.accuracy)
      if (practice) {
        // Practice has no opponent and no HP — show the raw hit, nothing to lose.
        dmg = base
      } else {
        // Scale the base hit by the time-based ramp so the race always converges.
        dmg = Math.round(base * damageRamp(this.state.elapsedMs))
        this.applyTug(result.player, dmg, word)
      }

      if (isMe) {
        this.state.cleared += 1
        this.state.currentWord = (practice && this.practiceStream ? this.practiceStream : this.stream).next()
      } else {
        this.oppCleared += 1
        this.oppCurrentWord = this.oppStream.next()
      }
    } else {
      player.combo = 0
    }

    const outcome: SignOutcome = {
      result,
      word,
      accepted,
      damage: dmg,
      points: pts,
      combo: player.combo,
    }
    this.emit('sign', outcome)
    this.emit('change', this.state)

    // Resolve the end condition after notifying listeners of the final hit.
    if (accepted && !practice) this.checkDeath()
    return outcome
  }

  /** Move HP from the loser to the signer, clamped, and record the step. */
  private applyTug(signer: PlayerId, dmg: number, word: Word): void {
    const me = this.state.players.me
    const opp = this.state.players.opponent
    if (signer === 'me') {
      opp.hp = Math.max(0, opp.hp - dmg)
      me.hp = Math.min(MAX_HP, me.hp + dmg)
    } else {
      me.hp = Math.max(0, me.hp - dmg)
      opp.hp = Math.min(MAX_HP, opp.hp + dmg)
    }
    this.tugHistory.push({
      word,
      delta: signer === 'me' ? dmg : -dmg,
      myHp: me.hp,
      oppHp: opp.hp,
      winner: signer,
    })
  }

  private checkDeath(): void {
    if (this.state.phase !== 'racing') return
    const { me, opponent } = this.state.players
    if (me.hp <= 0) this.endMatch('opponent')
    else if (opponent.hp <= 0) this.endMatch('me')
  }

  private endMatch(winner: PlayerId): void {
    if (this.state.phase === 'finished') return
    this.state.winner = winner
    this.setPhase('finished')
    this.emit('finished', undefined)
  }

  // --- online (server-authoritative) -----------------------------------------
  // In a networked duel the server owns recognition, HP, and the word sequence.
  // These setters make the store a view of that state (the local damage/death
  // path in submitSign is not used online).

  /** The word the server wants the local player to sign now. */
  setNetWord(index: number, text: string, difficulty = 1): void {
    if (this.state.currentWord.id === String(index) && this.state.currentWord.text === text) return
    this.state.currentWord = { id: String(index), text, difficulty }
    this.state.cleared = Math.max(this.state.cleared, index)
    this.emit('change', this.state)
  }

  /** Server-authoritative HP for both players (revealed on the results screen). */
  setNetHp(meHp: number, oppHp: number): void {
    this.state.players.me.hp = meHp
    this.state.players.opponent.hp = oppHp
    this.emit('change', this.state)
  }

  /** The server confirmed a sign — local feedback (combo/score/VFX). */
  netConfirm(strength: number): SignOutcome {
    const me = this.state.players.me
    me.combo += 1
    const accuracy = Math.max(0.6, Math.min(0.99, strength || 0.85))
    const pts = points(accuracy, 1500, me.combo)
    me.score += pts
    this.state.cleared += 1
    const outcome: SignOutcome = {
      result: { player: 'me', wordId: this.state.currentWord.id, accuracy, timeMs: 0 },
      word: this.state.currentWord,
      accepted: true,
      damage: 0,
      points: pts,
      combo: me.combo,
    }
    this.emit('sign', outcome)
    this.emit('change', this.state)
    return outcome
  }

  /** The server ended the match. */
  netFinish(winner: PlayerId): void {
    this.endMatch(winner)
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function nowMs(): number {
  return performance.now()
}

/** A non-deterministic seed so each Practice session varies. */
function freshSeed(): number {
  return Math.floor(Math.random() * 1e9)
}
