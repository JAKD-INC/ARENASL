/**
 * Shared game-state contract for ArenaSL.
 *
 * This is the integration boundary between the three workstreams:
 *  - UI overlay (this layer) reads {@link GameState} and renders the HUD/VFX.
 *  - Sign recognition (Alex) emits {@link SignResult} events into the store and
 *    publishes {@link LandmarkFrame}s through a {@link LandmarkProvider}.
 *  - Call connection (David) syncs the word sequence and serves the opponent
 *    replay that {@link ReplayData} describes.
 *
 * Nothing here imports the DOM, MediaPipe, or any transport, so each workstream
 * can move independently as long as it speaks this vocabulary.
 */

export type PlayerId = 'me' | 'opponent'

export interface PlayerState {
  id: PlayerId
  name: string
  /** 0..MAX_HP. Tug-of-war: a landed sign drains the opponent and refills you. */
  hp: number
  /** Accumulated points from accuracy + speed across the match. */
  score: number
  /** Consecutive accepted signs; resets on a miss. Drives the score multiplier. */
  combo: number
}

export interface Word {
  /** Stable id (index in the shared infinite list). */
  id: string
  text: string
  /** 1 (easy) .. 5 (hard). Scales base damage and points. */
  difficulty: number
}

export type GamePhase = 'idle' | 'countdown' | 'racing' | 'finished'

export interface GameState {
  phase: GamePhase
  players: Record<PlayerId, PlayerState>
  /** The word the local player is currently signing. */
  currentWord: Word
  /** How many words the local player has cleared so far. */
  cleared: number
  /** Seconds remaining in the countdown (only meaningful during 'countdown'). */
  countdown: number
  /** Elapsed match time in ms (only meaningful during 'racing'). */
  elapsedMs: number
  winner: PlayerId | null
}

/**
 * Emitted by the recognition layer when a player performs a sign.
 * The store turns this into a {@link SignOutcome} and mutates state.
 */
export interface SignResult {
  player: PlayerId
  /** Which word the sign was attempted against (guards against race conditions). */
  wordId: string
  /** Recognition confidence, 0..1. */
  accuracy: number
  /** How long the player took to perform the sign, in ms (drives speed bonus). */
  timeMs: number
}

/** Result of applying a {@link SignResult} — what the VFX layer reacts to. */
export interface SignOutcome {
  result: SignResult
  word: Word
  /** Whether accuracy cleared the acceptance threshold. */
  accepted: boolean
  /** HP transferred from opponent to the signing player. */
  damage: number
  /** Points awarded to the signing player. */
  points: number
  /** Combo count after this outcome (for the player who signed). */
  combo: number
}

/** One recorded attempt in a player's run — the timeline used to resolve a match. */
export interface SignEvent {
  wordId: string
  accuracy: number
  timeMs: number
  /** ms from match start when the sign landed. */
  atMs: number
}

/** Final, post-match summary shown on the results screen. */
export interface MatchResult {
  winner: PlayerId
  players: Record<PlayerId, { name: string; hp: number; score: number; cleared: number }>
  /** Per-word resolution, in order, for the breakdown list. */
  breakdown: TugStep[]
}

/** One step of the tug-of-war resolution (one shared word). */
export interface TugStep {
  word: Word
  /** Net HP moved this step. Positive = toward opponent (me winning the word). */
  delta: number
  /** Running HP for the local player after this step. */
  myHp: number
  /** Running HP for the opponent after this step. */
  oppHp: number
  winner: PlayerId | null
}

/**
 * Opponent replay, fetched from the server at results time.
 * The overlay never records video — see src/game/replay.ts.
 */
export interface ReplayData {
  /** Server-hosted video URL, or null while it is still unavailable. */
  videoUrl: string | null
  /** Opponent's sign timeline, used to drive the synced breakdown overlay. */
  timeline: SignEvent[]
}

// --- Skins / landmarks ------------------------------------------------------

/** A normalized 2D point (0..1 of the video frame), x already un-mirrored. */
export interface Landmark {
  x: number
  y: number
}

/** A single frame of detected landmarks for the local player. */
export interface LandmarkFrame {
  /** Face mesh points, if a face was detected this frame. */
  face: Landmark[] | null
  /** Hand points per detected hand. */
  hands: Landmark[][]
  /** Body pose points (MediaPipe Pose: 33 keypoints), if a body was detected. */
  pose: Landmark[] | null
  /** Frame timestamp in ms. */
  atMs: number
}

/**
 * Source of {@link LandmarkFrame}s. The overlay ships a standalone MediaPipe
 * implementation; in production Alex's recognizer can implement this instead so
 * MediaPipe only runs once.
 */
export interface LandmarkProvider {
  start(): Promise<void>
  stop(): void
  /** Latest frame, or null if nothing detected yet. */
  latest(): LandmarkFrame | null
}

export type SkinKind = 'none' | 'css-filter' | 'costume'

/** Where a costume sprite is anchored on the detected body. */
export type CostumeAnchor = 'head' | 'torso'

export interface CostumeLayer {
  /** Sprite image URL (transparent PNG/WebP). */
  src: string
  anchor: CostumeAnchor
  /**
   * Sprite width as a multiple of the reference width — head width for `head`
   * anchors, shoulder width for `torso` anchors.
   */
  scale: number
  /** Vertical nudge as a fraction of the reference width (negative = up). */
  offsetY?: number
}

export interface SkinDefinition {
  id: string
  name: string
  kind: SkinKind
  /** Accent color used by the picker swatch fallback. */
  accent: string
  /** For css-filter skins: a CSS `filter` value applied to the video element. */
  cssFilter?: string
  /** For costume skins: pose-anchored sprite layers, drawn in array order. */
  costume?: CostumeLayer[]
  /** Optional picker thumbnail image (e.g. the costume's hat sprite). */
  thumb?: string
}
