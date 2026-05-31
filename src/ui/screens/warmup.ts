import type { Screen } from '../../app/router.ts'
import { esc } from './dom.ts'

/**
 * "VS" intro splash shown once a match is locked in, before handing off to the
 * in-HUD countdown. Pure theatre — the hackathon "moment" that frames the duel.
 */
export class WarmupScreen implements Screen {
  private timer = 0

  constructor(
    private opponent: { name: string; elo?: number },
    private me: { name: string; elo?: number },
    private cb: { onDone: () => void },
  ) {}

  mount(host: HTMLElement): void {
    const elo = (v?: number): string => (v != null ? `<span class="vs-elo">${v}</span>` : '')
    host.innerHTML = `
      <div class="screen screen-center screen-vs">
        <div class="vs-row">
          <div class="vs-side me">
            <div class="vs-avatar">${esc(this.me.name.slice(0, 1).toUpperCase())}</div>
            <div class="vs-name">${esc(this.me.name)}</div>${elo(this.me.elo)}
          </div>
          <div class="vs-big">VS</div>
          <div class="vs-side opp">
            <div class="vs-avatar">${esc(this.opponent.name.slice(0, 1).toUpperCase())}</div>
            <div class="vs-name">${esc(this.opponent.name)}</div>${elo(this.opponent.elo)}
          </div>
        </div>
        <div class="vs-go">Get ready to sign!</div>
      </div>
    `
    this.timer = window.setTimeout(() => this.cb.onDone(), 2000)
  }

  destroy(): void {
    clearTimeout(this.timer)
  }
}
