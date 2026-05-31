import type { Screen } from '../../app/router.ts'
import type { Lobby, NetClient } from '../../net/protocol.ts'
import type { LookController } from '../looks/controller.ts'
import { LooksPanel } from '../looks/panel.ts'
import { esc } from './dom.ts'

/**
 * The shared waiting room for both private and matched lobbies. Shows the code
 * to share, both player slots with live ready state, your looks picker (try
 * lenses while you wait), and a Ready toggle. The match begins when both players
 * are ready — the controller listens for `warmupStart` and navigates on; this
 * screen only reflects lobby state.
 */
export class LobbyRoomScreen implements Screen {
  private off: Array<() => void> = []
  private codeEl!: HTMLDivElement
  private playersEl!: HTMLDivElement
  private readyBtn!: HTMLButtonElement

  constructor(
    private net: NetClient,
    private looks: LookController,
    private cb: { onLeave: () => void },
  ) {}

  mount(host: HTMLElement): void {
    host.innerHTML = `
      <div class="screen screen-lobby">
        <button class="screen-back" type="button" data-leave>← Leave</button>
        <div class="lobby-room">
          <div class="room-code" data-code></div>
          <div class="room-players" data-players></div>
          <div class="room-looks">
            <div class="room-looks-title">Your look</div>
            <div data-looks></div>
          </div>
          <button class="btn-tactile btn-mint block ready-btn" type="button" data-ready disabled>Ready up</button>
        </div>
      </div>
    `
    this.codeEl = host.querySelector('[data-code]')!
    this.playersEl = host.querySelector('[data-players]')!
    this.readyBtn = host.querySelector('[data-ready]')!
    new LooksPanel(host.querySelector<HTMLDivElement>('[data-looks]')!, this.looks)

    this.readyBtn.addEventListener('click', () => {
      const lobby = this.net.getLobby()
      this.net.setReady(!this.isMeReady(lobby))
    })
    host.querySelector('[data-leave]')!.addEventListener('click', () => this.cb.onLeave())

    this.off.push(this.net.on((e) => e.type === 'lobbyUpdate' && this.render(e.lobby)))
    this.render(this.net.getLobby())
  }

  destroy(): void {
    for (const off of this.off) off()
    this.off = []
  }

  private isMeReady(lobby: Lobby | null): boolean {
    return lobby?.members.find((m) => m.playerId === this.net.playerId)?.ready ?? false
  }

  private render(lobby: Lobby | null): void {
    const waiting = !lobby || lobby.state !== 'full'
    this.codeEl.innerHTML = lobby
      ? `<div class="code-label">Lobby code</div>
         <button class="code-value" type="button" data-copy title="Copy">${esc(lobby.code)}<span class="code-copy">⧉</span></button>`
      : `<div class="code-label">Creating lobby…</div>`
    this.codeEl.querySelector('[data-copy]')?.addEventListener('click', (ev) => {
      void navigator.clipboard?.writeText(lobby!.code)
      const b = ev.currentTarget as HTMLElement
      b.classList.add('copied')
      setTimeout(() => b.classList.remove('copied'), 1200)
    })

    const slots: string[] = []
    const members = lobby?.members ?? []
    for (let i = 0; i < 2; i++) {
      const m = members[i]
      if (m) {
        const me = m.playerId === this.net.playerId
        slots.push(`
          <div class="player-slot ${m.ready ? 'is-ready' : ''}">
            <div class="player-avatar">${esc(m.displayName.slice(0, 1).toUpperCase())}</div>
            <div class="player-name">${esc(m.displayName)}${me ? ' <span class="you-tag">you</span>' : ''}</div>
            <div class="player-status">${m.ready ? 'Ready ✓' : 'Not ready'}</div>
          </div>`)
      } else {
        slots.push(`
          <div class="player-slot empty">
            <div class="player-avatar"><span class="dot-spin"></span></div>
            <div class="player-name">Waiting…</div>
            <div class="player-status">share the code</div>
          </div>`)
      }
    }
    this.playersEl.innerHTML = `${slots[0]}<div class="vs-chip">VS</div>${slots[1]}`

    const meReady = this.isMeReady(lobby)
    this.readyBtn.disabled = waiting
    this.readyBtn.classList.toggle('btn-mint', !meReady)
    this.readyBtn.classList.toggle('btn-light', meReady)
    this.readyBtn.textContent = waiting
      ? 'Waiting for opponent…'
      : meReady
        ? 'Ready ✓ · tap to cancel'
        : 'Ready up'
  }
}
