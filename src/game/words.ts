import type { Word } from './types.ts'

/**
 * The shared infinite word list.
 *
 * There is no deck or hand — {@link WordStream.next} just keeps producing the
 * next word forever. Both peers seed the stream with the same value (David syncs
 * the seed) so they race the identical sequence until one player wins.
 */

interface WordSpec {
  text: string
  difficulty: number
}

/**
 * Source pool of ASL word-signs. Each word is a single sign (handshape +
 * motion) — not fingerspelled. The HUD teaches it with a visual demo clip
 * (`/signs/<slug>.gif`); difficulty reflects how involved the sign is.
 */
const POOL: WordSpec[] = [
  { text: 'HELLO', difficulty: 1 },
  { text: 'YES', difficulty: 1 },
  { text: 'NO', difficulty: 1 },
  { text: 'THANK YOU', difficulty: 2 },
  { text: 'PLEASE', difficulty: 2 },
  { text: 'NAME', difficulty: 2 },
  { text: 'FRIEND', difficulty: 3 },
  { text: 'WATER', difficulty: 3 },
  { text: 'LEARN', difficulty: 3 },
  { text: 'FAMILY', difficulty: 4 },
  { text: 'UNDERSTAND', difficulty: 4 },
  { text: 'PRACTICE', difficulty: 5 },
  { text: 'LANGUAGE', difficulty: 5 },
]

/** Deterministic PRNG (mulberry32) so a seed reproduces the exact sequence. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export class WordStream {
  private rand: () => number
  private index = 0
  /** Avoid repeating the same word back-to-back. */
  private lastPick = -1

  constructor(seed = 1) {
    this.rand = mulberry32(seed)
  }

  next(): Word {
    let pick = Math.floor(this.rand() * POOL.length)
    if (pick === this.lastPick) pick = (pick + 1) % POOL.length
    this.lastPick = pick
    const spec = POOL[pick]
    const word: Word = {
      id: String(this.index),
      text: spec.text,
      difficulty: spec.difficulty,
    }
    this.index += 1
    return word
  }
}
