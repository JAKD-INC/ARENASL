import type {
  Lobby,
  LandmarkPayload,
  NetClient,
  NetEvent,
  NetListener,
  OpponentView,
  PlayerProgress,
} from './protocol.ts'
import { MockNetClient } from './mockClient.ts'

/**
 * Real WebSocket client speaking the backend protocol (server/app/messages.py).
 * Written to spec but NOT yet verified against the live server — the app uses
 * {@link MockNetClient} by default (see {@link createNetClient}). It maps the
 * backend's snake_case wire format to/from the camelCase {@link NetEvent}s.
 *
 * Phase-5 TODO (needs David's running server): WebRTC peer link. On `match.found`
 * we get a `role`; the real client must build an RTCPeerConnection, exchange SDP/
 * ICE via `{type:'signal'}` messages, and feed the opponent's live progress into
 * the store. Until then, the in-match opponent is the local MockDriver.
 */
export class WebSocketNetClient implements NetClient {
  playerId: number | null = null
  elo: number | null = null
  displayName = 'You'

  private url: string
  private ws: WebSocket | null = null
  private listeners = new Set<NetListener>()
  private lobby: Lobby | null = null
  private queuePosition = 0

  constructor(url: string) {
    this.url = url
  }

  connect(displayName: string, token?: string): Promise<void> {
    this.displayName = displayName || 'You'
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url)
      this.ws = ws
      ws.addEventListener('open', () => this.send({ type: 'auth', token: token ?? '' }))
      ws.addEventListener('message', (ev) => this.handle(JSON.parse(String(ev.data)), resolve))
      ws.addEventListener('error', () => reject(new Error('WebSocket connection failed')))
      ws.addEventListener('close', () => this.emit({ type: 'opponentStatus', playerId: -1, connected: false }))
    })
  }

  disconnect(): void {
    this.ws?.close()
    this.ws = null
  }

  joinQueue(): void {
    this.send({ type: 'queue.join' })
  }
  leaveQueue(): void {
    this.send({ type: 'queue.leave' })
  }
  createLobby(): void {
    this.send({ type: 'lobby.create', private: true })
  }
  joinLobby(code: string): void {
    this.send({ type: 'lobby.join', code: code.toUpperCase() })
  }
  setReady(ready: boolean): void {
    this.send({ type: 'lobby.ready', ready })
  }
  leaveLobby(): void {
    // No explicit lobby-leave message in the protocol; the server drops you on
    // disconnect or when you create/join elsewhere. Reconnect to return to idle.
    this.disconnect()
  }
  sendLandmark(frame: LandmarkPayload): void {
    // Server accepts camelCase handLeft/handRight (populate_by_name alias).
    this.send({ type: 'landmark', t: frame.t, pose: frame.pose, handLeft: frame.handLeft, handRight: frame.handRight })
  }
  startPractice(): void {
    this.send({ type: 'practice.start' })
  }
  stopPractice(): void {
    this.send({ type: 'practice.stop' })
  }

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

  // --- wire mapping ---------------------------------------------------------

  private send(obj: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj))
  }

  private emit(event: NetEvent): void {
    for (const l of [...this.listeners]) l(event)
  }

  private handle(msg: Record<string, unknown>, resolveAuth: () => void): void {
    switch (msg.type) {
      case 'auth.ok':
        this.playerId = msg.player_id as number
        this.elo = msg.elo as number
        resolveAuth()
        this.emit({ type: 'authOk', playerId: this.playerId, elo: this.elo })
        break
      case 'error':
        this.emit({ type: 'error', code: msg.code as string, message: msg.message as string })
        break
      case 'queue.status':
        this.queuePosition = msg.position as number
        this.emit({ type: 'queueStatus', position: this.queuePosition })
        break
      case 'lobby.update':
        this.lobby = mapLobby(msg)
        this.emit({ type: 'lobbyUpdate', lobby: this.lobby })
        break
      case 'match.found':
        this.emit({
          type: 'matchFound',
          matchId: msg.match_id as string,
          role: msg.role as 'offerer' | 'answerer',
          opponent: mapOpponent(msg.opponent as Record<string, unknown>),
        })
        break
      case 'warmup.start':
        this.emit({ type: 'warmupStart', wordSeed: msg.word_seed as number, datasetVersion: msg.dataset_version as string })
        break
      case 'practice.start':
        this.emit({ type: 'practiceStart', wordSeed: msg.word_seed as number, datasetVersion: msg.dataset_version as string })
        break
      case 'match.start':
        this.emit({
          type: 'matchStart',
          matchId: msg.match_id as string,
          wordSeed: msg.word_seed as number,
          recordStartMs: (msg.record_start_ms as number) ?? 0,
        })
        break
      case 'recognition.update':
        this.emit({
          type: 'recognitionUpdate',
          wordIndex: msg.word_index as number,
          word: msg.word as string,
          strength: msg.strength as number,
          // The server may include per-word difficulty (word_at().difficulty);
          // default to 1 when absent so the HUD/scoring stay correct either way.
          difficulty: (msg.difficulty as number | undefined) ?? 1,
        })
        break
      case 'match.state':
        this.emit({ type: 'matchState', matchId: msg.match_id as string, players: mapPlayers(msg) })
        break
      case 'match.over':
        this.emit({
          type: 'matchOver',
          matchId: msg.match_id as string,
          winnerId: (msg.winner_id as number | null) ?? null,
          elo: (msg.elo as number | null) ?? null,
          eloDelta: (msg.elo_delta as number | null) ?? null,
        })
        break
      case 'opponent.status':
        this.emit({ type: 'opponentStatus', playerId: msg.player_id as number, connected: Boolean(msg.connected) })
        break
    }
  }

}

function mapPlayers(msg: Record<string, unknown>): PlayerProgress[] {
  const players = (msg.players as Array<Record<string, unknown>>) ?? []
  return players.map((p) => ({
    playerId: p.player_id as number,
    hp: p.hp as number,
    wordIndex: p.word_index as number,
  }))
}

function mapLobby(msg: Record<string, unknown>): Lobby {
  const members = (msg.members as Array<Record<string, unknown>>) ?? []
  return {
    code: msg.code as string,
    state: msg.state as Lobby['state'],
    members: members.map((m) => ({
      playerId: m.player_id as number,
      displayName: m.display_name as string,
      ready: Boolean(m.ready),
      connected: Boolean(m.connected),
    })),
  }
}

function mapOpponent(o: Record<string, unknown>): OpponentView {
  return { playerId: o.player_id as number, displayName: o.display_name as string, elo: o.elo as number }
}

/**
 * Pick the networking implementation. Defaults to the local mock so the flow is
 * demoable with no server. Set `localStorage.arenasl.server = "wss://host/ws"`
 * (or `?server=` in the URL) to use the real backend.
 */
export function createNetClient(): NetClient {
  const params = new URLSearchParams(location.search)
  if (params.get('mock') === '1') return new MockNetClient()
  const server = params.get('server') ?? localStorage.getItem('arenasl.server') ?? '/ws'
  return new WebSocketNetClient(resolveWsUrl(server))
}

function resolveWsUrl(s: string): string {
  if (/^wss?:/.test(s)) return s
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}${s.startsWith('/') ? s : `/${s}`}`
}
