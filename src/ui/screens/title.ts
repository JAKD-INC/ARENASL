import type { Screen } from '../../app/router.ts'
import { esc } from './dom.ts'

export interface TitleCallbacks {
  onPlay: () => void
  onPractice: () => void
  onChangeName: () => void
}

/** Landing screen: the hook + the two ways in (Play vs Practice). */
export class TitleScreen implements Screen {
  constructor(
    private name: string,
    private cb: TitleCallbacks,
  ) {}

  mount(host: HTMLElement): void {
    host.innerHTML = `
      <div class="screen screen-title">
        <div class="title-top">
          <div class="title-brand">ARENA<span class="accent">SL</span></div>
          <div class="title-tag">Learn ASL by battling.<br />Out-sign your rival.</div>
        </div>
        <div class="title-actions">
          <button class="btn-tactile btn-coral title-play" type="button" data-play>Play ▶</button>
          <button class="btn-tactile btn-light title-practice" type="button" data-practice>Practice</button>
        </div>
        <button class="title-name" type="button" data-name>
          Signing in as <b>${esc(this.name)}</b> · change
        </button>
      </div>
    `
    host.querySelector('[data-play]')!.addEventListener('click', () => this.cb.onPlay())
    host.querySelector('[data-practice]')!.addEventListener('click', () => this.cb.onPractice())
    host.querySelector('[data-name]')!.addEventListener('click', () => this.cb.onChangeName())
  }

  destroy(): void {}
}
