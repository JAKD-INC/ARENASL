/**
 * Minimal screen router for app-level navigation (title → mode → lobby → …),
 * which is separate from the in-match {@link GameStore} phase machine. Screens
 * are plain components mounted into a single host element over the persistent
 * camera/looks background; showing `null` clears the host so the game (HUD,
 * capture) takes over during a match.
 */
export interface Screen {
  mount(host: HTMLElement): void
  /** Tear down listeners/timers. Always called before the next screen mounts. */
  destroy(): void
}

export class Router {
  private host: HTMLElement
  private current: Screen | null = null

  constructor(host: HTMLElement) {
    this.host = host
  }

  show(screen: Screen | null): void {
    this.current?.destroy()
    this.host.replaceChildren()
    this.current = screen
    if (screen) {
      this.host.classList.remove('hidden')
      screen.mount(this.host)
    } else {
      this.host.classList.add('hidden')
    }
  }
}
