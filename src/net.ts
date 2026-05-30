export interface LandmarkMessage {
  t: number
  pose: number[][] | null
  handLeft: number[][] | null
  handRight: number[][] | null
}

export interface GameState {
  current: string
  queue: string[]
  strength: number
  score: number
  event: string | null
  confirmed: string | null
}

interface Options {
  onState: (state: GameState) => void
  url?: string
  factory?: () => WebSocket
}

export function createConnection(opts: Options) {
  // Compute the default URL lazily inside the factory so `location` is only
  // read in the browser, never in the node test env (which injects a factory).
  const make = opts.factory ?? (() => new WebSocket(opts.url ?? `ws://${location.host}/ws`))
  let socket = connect()

  function connect(): WebSocket {
    const s = make()
    s.onmessage = (e: MessageEvent) => opts.onState(JSON.parse(e.data))
    s.onclose = () => setTimeout(() => { socket = connect() }, 1000)
    return s
  }

  return {
    send(msg: LandmarkMessage) {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg))
    },
  }
}
