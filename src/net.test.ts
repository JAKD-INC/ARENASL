import { expect, test, vi } from 'vitest'
import { createConnection, type LandmarkMessage } from './net.ts'

class FakeSocket {
  static OPEN = 1
  readyState = FakeSocket.OPEN
  sent: string[] = []
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  send(data: string) { this.sent.push(data) }
  close() {}
}

const msg: LandmarkMessage = { t: 0, pose: null, handLeft: null, handRight: null }

test('send serializes the message to the socket', () => {
  const sock = new FakeSocket()
  const conn = createConnection({ factory: () => sock as unknown as WebSocket, onState: () => {} })
  conn.send(msg)
  expect(JSON.parse(sock.sent[0])).toEqual(msg)
})

test('incoming data is parsed and dispatched to onState', () => {
  const sock = new FakeSocket()
  const onState = vi.fn()
  createConnection({ factory: () => sock as unknown as WebSocket, onState })
  sock.onmessage!({ data: JSON.stringify({ current: 'book', score: 20 }) })
  expect(onState).toHaveBeenCalledWith({ current: 'book', score: 20 })
})
