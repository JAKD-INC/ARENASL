import type { Screen } from '../../app/router.ts'
import type { NetClient } from '../../net/protocol.ts'

/** Public matchmaking ("Find a Rival"): searching state until the queue pairs. */
export class FindRivalScreen implements Screen {
  private off: Array<() => void> = []
  private timer = 0
  private startMs = 0

  constructor(
    private net: NetClient,
    private cb: { onPaired: () => void; onCancel: () => void },
  ) {}

  mount(host: HTMLElement): void {
    host.innerHTML = `
      <div class="screen screen-center">
        <div class="find-stage">
          <div class="radar"><span></span><span></span><span></span><div class="radar-core">⚔️</div></div>
          <div class="card-title">Finding a rival…</div>
          <div class="find-meta"><span data-pos>Searching</span> · <span data-time>0s</span></div>
        </div>
        <button class="btn-tactile btn-light find-cancel" type="button" data-cancel>Cancel</button>
      </div>
    `
    const posEl = host.querySelector<HTMLSpanElement>('[data-pos]')!
    const timeEl = host.querySelector<HTMLSpanElement>('[data-time]')!
    this.startMs = performance.now()
    this.timer = window.setInterval(() => {
      timeEl.textContent = `${Math.floor((performance.now() - this.startMs) / 1000)}s`
    }, 250)

    const pos = this.net.getQueuePosition()
    if (pos > 0) posEl.textContent = `#${pos} in queue`

    this.off.push(
      this.net.on((e) => {
        if (e.type === 'queueStatus') posEl.textContent = `#${e.position} in queue`
        else if (e.type === 'lobbyUpdate' && e.lobby.state === 'full') this.cb.onPaired()
      }),
    )
    host.querySelector('[data-cancel]')!.addEventListener('click', () => this.cb.onCancel())
  }

  destroy(): void {
    clearInterval(this.timer)
    for (const off of this.off) off()
    this.off = []
  }
}
