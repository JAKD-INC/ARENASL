import type { Screen } from '../../app/router.ts'

export interface ModeSelectCallbacks {
  onCreate: () => void
  onJoin: () => void
  onFind: () => void
  onBack: () => void
}

/** Choose how to play: private lobby (create/join) or public "Find a Rival". */
export class ModeSelectScreen implements Screen {
  constructor(private cb: ModeSelectCallbacks) {}

  mount(host: HTMLElement): void {
    host.innerHTML = `
      <div class="screen screen-center">
        <button class="screen-back" type="button" data-back>← Back</button>
        <div class="mode-head">
          <div class="card-title">Play a match</div>
          <div class="card-sub">Battle a friend privately, or get matched by skill.</div>
        </div>
        <div class="mode-cards">
          <button class="mode-card primary" type="button" data-find>
            <span class="mode-emoji">⚔️</span>
            <span class="mode-name">Find a Rival</span>
            <span class="mode-desc">Get matched with someone your level</span>
          </button>
          <button class="mode-card" type="button" data-create>
            <span class="mode-emoji">🔒</span>
            <span class="mode-name">Create private lobby</span>
            <span class="mode-desc">Get a code to share with a friend</span>
          </button>
          <button class="mode-card" type="button" data-join>
            <span class="mode-emoji">🔑</span>
            <span class="mode-name">Join with a code</span>
            <span class="mode-desc">Enter a friend’s lobby code</span>
          </button>
        </div>
      </div>
    `
    host.querySelector('[data-find]')!.addEventListener('click', () => this.cb.onFind())
    host.querySelector('[data-create]')!.addEventListener('click', () => this.cb.onCreate())
    host.querySelector('[data-join]')!.addEventListener('click', () => this.cb.onJoin())
    host.querySelector('[data-back]')!.addEventListener('click', () => this.cb.onBack())
  }

  destroy(): void {}
}
