/**
 * Top bar shown only in Practice mode: leave (Done), replay the tutorial (?),
 * and re-open the look picker (Looks) so the player can try lenses/filters
 * without leaving the room. Purely a thin control strip — the callbacks wire it
 * to the store/picker in main.
 */
export interface PracticeBarCallbacks {
  onDone: () => void
  onLooks: () => void
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
      <div class="pb-right">
        <button class="pb-btn pb-icon" type="button" data-help aria-label="How to play">?</button>
        <button class="pb-btn pb-looks" type="button" data-looks>🎭 Looks</button>
      </div>
    `
    root.querySelector('[data-done]')!.addEventListener('click', () => cb.onDone())
    root.querySelector('[data-help]')!.addEventListener('click', () => cb.onHelp())
    root.querySelector('[data-looks]')!.addEventListener('click', () => cb.onLooks())
  }

  setVisible(visible: boolean): void {
    this.root.classList.toggle('hidden', !visible)
  }
}
