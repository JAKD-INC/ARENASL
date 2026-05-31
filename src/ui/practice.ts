/**
 * Top bar shown only in Practice mode: leave (Done) and replay the tutorial (?).
 * Practice is purely about learning to sign — looks/filters live in the lobby.
 */
export interface PracticeBarCallbacks {
  onDone: () => void
  onHelp: () => void
}

export class PracticeBar {
  private root: HTMLElement

  constructor(root: HTMLElement, cb: PracticeBarCallbacks) {
    this.root = root
    root.classList.add('practice-bar', 'hidden')
    root.innerHTML = `
      <button class="pb-btn pb-done" type="button" data-done>← Done</button>
      <div class="pb-title">Practice</div>
      <button class="pb-btn pb-icon" type="button" data-help aria-label="How to play">?</button>
    `
    root.querySelector('[data-done]')!.addEventListener('click', () => cb.onDone())
    root.querySelector('[data-help]')!.addEventListener('click', () => cb.onHelp())
  }

  setVisible(visible: boolean): void {
    this.root.classList.toggle('hidden', !visible)
  }
}
