import type { Screen } from '../../app/router.ts'
import type { NetClient } from '../../net/protocol.ts'

/** Enter a 6-character lobby code. Success → onJoined; server errors show inline. */
export class JoinCodeScreen implements Screen {
  private off: Array<() => void> = []

  constructor(
    private net: NetClient,
    private cb: { onJoined: () => void; onBack: () => void },
  ) {}

  mount(host: HTMLElement): void {
    host.innerHTML = `
      <div class="screen screen-center">
        <button class="screen-back" type="button" data-back>← Back</button>
        <div class="card">
          <div class="card-title">Join a lobby</div>
          <div class="card-sub">Enter the 6-character code your friend shared.</div>
          <input class="text-input code-input" data-input type="text" inputmode="latin"
            maxlength="6" placeholder="ABC123" autocomplete="off" autocapitalize="characters" />
          <div class="form-error hidden" data-error></div>
          <button class="btn-tactile btn-coral block" type="button" data-go disabled>Join</button>
        </div>
      </div>
    `
    const input = host.querySelector<HTMLInputElement>('[data-input]')!
    const go = host.querySelector<HTMLButtonElement>('[data-go]')!
    const error = host.querySelector<HTMLDivElement>('[data-error]')!

    const clean = (v: string): string => v.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6)
    input.addEventListener('input', () => {
      input.value = clean(input.value)
      go.disabled = input.value.length !== 6
      error.classList.add('hidden')
    })
    const submit = (): void => {
      if (input.value.length !== 6) return
      go.disabled = true
      go.textContent = 'Joining…'
      error.classList.add('hidden')
      this.net.joinLobby(input.value)
    }
    go.addEventListener('click', submit)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submit()
    })

    this.off.push(
      this.net.on((e) => {
        if (e.type === 'lobbyUpdate') this.cb.onJoined()
        else if (e.type === 'error') {
          error.textContent = e.message
          error.classList.remove('hidden')
          go.disabled = input.value.length !== 6
          go.textContent = 'Join'
        }
      }),
    )
    host.querySelector('[data-back]')!.addEventListener('click', () => this.cb.onBack())
    setTimeout(() => input.focus(), 50)
  }

  destroy(): void {
    for (const off of this.off) off()
    this.off = []
  }
}
