import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { createOverlay, ropePosition } from './overlay.ts'
import type { GameState } from './net.ts'

test('zero score is neutral centre', () => {
  expect(ropePosition(0, 200)).toBeCloseTo(0.5)
})

test('positive score pulls toward 1, negative toward 0', () => {
  expect(ropePosition(100, 200)).toBeGreaterThan(0.5)
  expect(ropePosition(-100, 200)).toBeLessThan(0.5)
})

test('clamped to [0,1] beyond the range', () => {
  expect(ropePosition(99999, 200)).toBe(1)
  expect(ropePosition(-99999, 200)).toBe(0)
})

// --- createOverlay (DOM render) -------------------------------------------
// The node test env has no DOM, so we stub the minimum the renderer touches:
// document.createElement (for prompt spans) and a set of fake root elements.

type FakeSpan = { textContent: string; className: string }

function fakeRoot() {
  const appended: FakeSpan[] = []
  const root = {
    prompts: {
      innerHTML: 'STALE',
      appended,
      appendChild(el: FakeSpan) {
        appended.push(el)
      },
    } as unknown as HTMLElement & { appended: FakeSpan[] },
    clip: {
      src: '',
      playCount: 0,
      play() {
        ;(this as { playCount: number }).playCount++
        return Promise.resolve()
      },
    } as unknown as HTMLVideoElement & { playCount: number },
    ropeMarker: { style: { left: '' } } as unknown as HTMLElement,
    score: { textContent: '' } as unknown as HTMLElement,
  }
  return root
}

const state = (over: Partial<GameState> = {}): GameState => ({
  current: 'book',
  queue: ['drink', 'help'],
  strength: 0.5,
  score: 0,
  event: null,
  confirmed: null,
  ...over,
})

beforeEach(() => {
  vi.stubGlobal('document', {
    createElement: () => ({ textContent: '', className: '' }) as FakeSpan,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

test('renders current sign first then the lookahead queue', () => {
  const root = fakeRoot()
  const render = createOverlay(root)
  render(state())
  const spans = (root.prompts as unknown as { appended: FakeSpan[] }).appended
  expect(spans.map((s) => s.textContent)).toEqual(['book', 'drink', 'help'])
  expect(spans[0].className).toBe('prompt prompt--current')
  expect(spans[1].className).toBe('prompt')
  expect(spans[2].className).toBe('prompt')
  // innerHTML is cleared before re-rendering the prompt row.
  expect((root.prompts as unknown as { innerHTML: string }).innerHTML).toBe('')
})

test('sets the reference clip src for the current sign and plays it', () => {
  const root = fakeRoot()
  const clip = root.clip as unknown as { src: string; playCount: number }
  const render = createOverlay(root)
  render(state({ current: 'book' }))
  expect(clip.src).toBe('/clips/book.mp4')
  expect(clip.playCount).toBe(1)
})

test('only swaps the clip src when the current sign changes', () => {
  const root = fakeRoot()
  const clip = root.clip as unknown as { src: string; playCount: number }
  const render = createOverlay(root)
  render(state({ current: 'book' }))
  render(state({ current: 'book', score: 20 })) // same sign, new score
  expect(clip.playCount).toBe(1) // not replayed
  render(state({ current: 'drink' }))
  expect(clip.src).toBe('/clips/drink.mp4')
  expect(clip.playCount).toBe(2)
})

test('renders score text and maps score to the rope marker position', () => {
  const root = fakeRoot()
  const score = root.score as unknown as { textContent: string }
  const marker = root.ropeMarker as unknown as { style: { left: string } }
  const render = createOverlay(root)
  render(state({ score: 0 }))
  expect(score.textContent).toBe('0')
  expect(marker.style.left).toBe('50%') // ropePosition(0, 200) = 0.5
  render(state({ current: 'book', score: 100 }))
  expect(score.textContent).toBe('100')
  expect(marker.style.left).toBe('75%') // 0.5 + 100/400 = 0.75
})
