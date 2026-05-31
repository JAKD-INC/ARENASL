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

/** Live opponent state the server streams during a duel (authoritative). */
export interface OpponentProgress {
  hp: number
  wordIndex: number
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
  | { type: 'matchState'; matchId: string; opponent: OpponentProgress | null }
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

  // in-match
  sendSignAttempt(wordIndex: number, accuracy: number): void

  on(listener: NetListener): () => void

  readonly playerId: number | null
  readonly elo: number | null
  readonly displayName: string
  getLobby(): Lobby | null
  getQueuePosition(): number
}
