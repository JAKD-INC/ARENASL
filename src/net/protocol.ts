/**
 * Frontend mirror of the backend WebSocket protocol (server/app/messages.py on
 * `feat/backend-server`). The UI is built entirely against {@link NetClient} so
 * the {@link MockNetClient} (local simulation, demoable now) and the real
 * {@link WebSocketNetClient} are interchangeable.
 *
 * Field names are camelCase here; the WS client maps to/from the backend's
 * snake_case on the wire.
 */

export type LobbyState = 'waiting' | 'full'

export interface LobbyMember {
  playerId: number
  displayName: string
  ready: boolean
  connected: boolean
}

export interface Lobby {
  code: string
  state: LobbyState
  members: LobbyMember[]
}

export interface OpponentView {
  playerId: number
  displayName: string
  elo: number
}

/** Authoritative per-player state the server streams during a duel. */
export interface PlayerProgress {
  playerId: number
  hp: number
  wordIndex: number
}

/** One frame of MediaPipe landmarks streamed up during an active match. */
export interface LandmarkPayload {
  /** Client clock (ms) — the server downsamples on this. */
  t: number
  pose: number[][] | null
  handLeft: number[][] | null
  handRight: number[][] | null
}

/** Server → client events the UI reacts to. */
export type NetEvent =
  | { type: 'authOk'; playerId: number; elo: number }
  | { type: 'error'; code: string; message: string }
  | { type: 'queueStatus'; position: number }
  | { type: 'lobbyUpdate'; lobby: Lobby }
  | { type: 'matchFound'; matchId: string; role: 'offerer' | 'answerer'; opponent: OpponentView }
  | { type: 'warmupStart'; wordSeed: number; datasetVersion: string }
  | { type: 'matchStart'; matchId: string; wordSeed: number; recordStartMs: number }
  | { type: 'matchState'; matchId: string; players: PlayerProgress[] }
  | { type: 'recognitionUpdate'; wordIndex: number; word: string; strength: number; difficulty: number }
  | { type: 'practiceStart'; wordSeed: number; datasetVersion: string }
  | { type: 'matchOver'; matchId: string; winnerId: number | null; elo: number | null; eloDelta: number | null }
  | { type: 'opponentStatus'; playerId: number; connected: boolean }

export type NetListener = (event: NetEvent) => void

/**
 * The single networking seam. The current lobby/queue are exposed as getters so
 * a freshly mounted screen can render immediately, then keep up via {@link on}.
 */
export interface NetClient {
  /** Open the connection and authenticate; resolves once authed. */
  connect(displayName: string, token?: string): Promise<void>
  disconnect(): void

  // matchmaking
  joinQueue(): void
  leaveQueue(): void

  // lobbies
  createLobby(): void
  joinLobby(code: string): void
  setReady(ready: boolean): void
  leaveLobby(): void

  // in-match: stream landmarks up; the server recognizes and streams results back
  sendLandmark(frame: LandmarkPayload): void

  // practice: a SOLO server recognizer (no matchmaking). Start/stop bookend a
  // dedicated stream; landmarks flow over the same sendLandmark seam and results
  // come back as the existing `recognitionUpdate` events.
  startPractice(): void
  stopPractice(): void

  on(listener: NetListener): () => void

  readonly playerId: number | null
  readonly elo: number | null
  readonly displayName: string
  getLobby(): Lobby | null
  getQueuePosition(): number
}
