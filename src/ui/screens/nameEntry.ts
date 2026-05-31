import type { Screen } from '../../app/router.ts'
import { esc } from './dom.ts'

/** First-run (and "change name") identity step — the display name peers see. */
export class NameEntryScreen implements Screen {
  constructor(
    private initial: string,
    private cb: { onSubmit: (name: string) => void },
  ) {}

  mount(host: HTMLElement): void {
    host.innerHTML = `
      <div class="screen screen-center">
        <div class="card">
          <div class="card-title">What should we call you?</div>
          <div class="card-sub">This is the name your rival sees.</div>
          <input class="text-input" data-input type="text" maxlength="16"
            placeholder="Your name" value="${esc(this.initial)}" autocomplete="off" />
          <button class="btn-tactile btn-coral block" type="button" data-go>Continue</button>
        </div>
      </div>
    `
    const input = host.querySelector<HTMLInputElement>('[data-input]')!
    const go = host.querySelector<HTMLButtonElement>('[data-go]')!
    const submit = (): void => {
      const name = input.value.trim()
      if (name) this.cb.onSubmit(name)
    }
    go.addEventListener('click', submit)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submit()
    })
    setTimeout(() => input.focus(), 50)
  }

  destroy(): void {}
}
