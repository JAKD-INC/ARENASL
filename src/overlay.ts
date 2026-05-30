import type { GameState } from './net.ts'

/** Map a running score to a rope marker position in [0,1]. 0.5 is neutral;
 *  `range` is the score magnitude that reaches an end. */
export function ropePosition(score: number, range: number): number {
  const pos = 0.5 + score / (2 * range)
  return Math.max(0, Math.min(1, pos))
}

const ROPE_RANGE = 200  // score magnitude that reaches a rope end

export function createOverlay(root: {
  prompts: HTMLElement
  clip: HTMLVideoElement
  ropeMarker: HTMLElement
  score: HTMLElement
}) {
  let currentClip = ''
  return function render(state: GameState) {
    // Prompt queue: current sign first (highlighted), then the lookahead.
    root.prompts.innerHTML = ''
    ;[state.current, ...state.queue].forEach((gloss, i) => {
      const el = document.createElement('span')
      el.textContent = gloss
      el.className = i === 0 ? 'prompt prompt--current' : 'prompt'
      root.prompts.appendChild(el)
    })
    // Reference clip for the current sign (swap src only when it changes).
    if (state.current !== currentClip) {
      currentClip = state.current
      root.clip.src = `/clips/${state.current}.mp4`
      root.clip.play().catch(() => {})
    }
    // Score + rope.
    root.score.textContent = String(state.score)
    root.ropeMarker.style.left = `${ropePosition(state.score, ROPE_RANGE) * 100}%`
  }
}
