import type { GameState, Word } from '../game/types.ts'
import type { GameStore } from '../game/store.ts'
import { SIGN_GIFS } from '../game/signAssets.ts'

/**
 * The live, solo-focus HUD: a glassmorphism overlay showing only the local
 * player's match. The current word is taught with a **visual** demo of the sign
 * (a looping gif/image) — players prop their phone up and watch, they don't
 * read — plus your timer, score, and combo. No opponent info and no HP bar;
 * those are revealed on the results screen.
 *
 * Demo art is loaded from `/signs/<slug>.gif` (e.g. THANK YOU → thank-you.gif).
 * If it is missing the panel shows a neutral placeholder, so the HUD works
 * before any demo clips are added.
 *
 * The word panel + stats only appear during the countdown/race, never on the
 * idle pre-match screen.
 */
export class Hud {
  private root: HTMLElement
  private el: {
    countdown: HTMLDivElement
    panel: HTMLDivElement
    demo: HTMLDivElement
    word: HTMLDivElement
    pips: HTMLDivElement
    timer: HTMLDivElement
    score: HTMLDivElement
    combo: HTMLDivElement
  }
  private renderedWordId: string | null = null

  constructor(root: HTMLElement, store: GameStore) {
    this.root = root
    this.root.classList.add('hud')
    this.root.innerHTML = `
      <div class="hud-stats">
        <div class="stat score"><span class="stat-label">Score</span><span class="stat-value" data-score>0</span></div>
        <div class="stat timer"><span class="stat-label">Time</span><span class="stat-value" data-timer>0.0</span></div>
        <div class="stat combo"><span class="stat-label">Combo</span><span class="stat-value" data-combo>x0</span></div>
      </div>
      <div class="hud-word" data-panel>
        <div class="hud-word-inner">
          <div class="hud-word-label">Sign this</div>
          <div class="hud-demo" data-demo></div>
          <div class="hud-word-text" data-word></div>
          <div class="hud-pips" data-pips></div>
        </div>
      </div>
      <div class="hud-countdown" data-countdown></div>
    `
    this.el = {
      countdown: this.q('[data-countdown]'),
      panel: this.q('[data-panel]'),
      demo: this.q('[data-demo]'),
      word: this.q('[data-word]'),
      pips: this.q('[data-pips]'),
      timer: this.q('[data-timer]'),
      score: this.q('[data-score]'),
      combo: this.q('[data-combo]'),
    }

    store.on('change', (s) => this.render(s))
    store.on('sign', (o) => {
      if (o.result.player === 'me' && o.accepted) this.pulse()
    })
    this.render(store.getState())
  }

  private q<T extends HTMLElement>(sel: string): T {
    const node = this.root.querySelector<T>(sel)
    if (!node) throw new Error(`HUD missing element: ${sel}`)
    return node
  }

  private render(s: GameState): void {
    // CSS uses data-phase to show the panel/stats only during play.
    this.root.dataset.phase = s.phase

    if (s.phase === 'countdown') {
      this.el.countdown.textContent = s.countdown > 0 ? String(s.countdown) : 'GO'
      this.el.countdown.classList.add('show')
    } else {
      this.el.countdown.classList.remove('show')
    }

    if (this.renderedWordId !== s.currentWord.id) {
      this.renderWord(s.currentWord)
      this.renderedWordId = s.currentWord.id
    }

    this.el.timer.textContent = (s.elapsedMs / 1000).toFixed(1)
    this.el.score.textContent = String(s.players.me.score)
    this.el.combo.textContent = `x${s.players.me.combo}`
    this.el.combo.parentElement?.classList.toggle('active', s.players.me.combo >= 2)
  }

  /** Rebuild the visual teaching demo for a new word (only when it changes). */
  private renderWord(word: Word): void {
    this.el.word.textContent = word.text

    // Visual cue: a looping gif/image of the sign. No text instructions.
    this.el.demo.innerHTML = ''
    this.el.demo.classList.remove('no-demo')
    const key = slug(word.text)
    const img = document.createElement('img')
    img.className = 'hud-demo-img'
    img.alt = `Sign for ${word.text}`
    // Use the temporary placeholder GIF if one is mapped; otherwise look for a
    // real demo clip in /public/signs (Alex's dataset). Clearing SIGN_GIFS once
    // the real assets land makes the local files take over automatically.
    img.src = SIGN_GIFS[key] ?? `/signs/${key}.gif`
    img.addEventListener('error', () => this.el.demo.classList.add('no-demo'), { once: true })
    const placeholder = document.createElement('div')
    placeholder.className = 'hud-demo-placeholder'
    placeholder.textContent = '🤟'
    this.el.demo.append(img, placeholder)

    this.el.pips.innerHTML = ''
    for (let i = 0; i < 5; i++) {
      const pip = document.createElement('span')
      pip.className = i < word.difficulty ? 'pip on' : 'pip'
      this.el.pips.append(pip)
    }
  }

  /** Brief feedback flash on the panel when a sign lands. */
  private pulse(): void {
    this.el.panel.classList.remove('hit')
    // Force reflow so re-adding the class restarts the animation.
    void this.el.panel.offsetWidth
    this.el.panel.classList.add('hit')
  }
}

/** "THANK YOU" → "thank-you" for the demo asset filename. */
function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}
