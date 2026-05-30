import { LOOK_GROUPS, type LookCategoryGroup, type LookItem } from './catalogue.ts'
import type { LookController } from './controller.ts'

/**
 * Pre-match lobby: a Snapchat-style look picker. Category tabs (Lenses /
 * Filters) swap a horizontally-scrolling carousel of swatches; the leading
 * "None" chip clears everything. Picking a swatch applies the look to the live
 * feed immediately through the {@link LookController}; only one look is active at
 * a time, and the selection persists across tabs.
 */
export class LookPicker {
  private root: HTMLElement
  private controller: LookController
  private onStart: () => void
  private swatchRow!: HTMLDivElement
  private tabRow!: HTMLDivElement
  private activeGroup: LookCategoryGroup = LOOK_GROUPS[0]
  private selectedId: string | null = null
  private buttons = new Map<string, HTMLButtonElement>()

  constructor(root: HTMLElement, controller: LookController, onStart: () => void) {
    this.root = root
    this.controller = controller
    this.onStart = onStart
    this.build()
    this.showGroup(this.activeGroup)
  }

  private build(): void {
    this.root.innerHTML = `
      <div class="lobby-sheet">
        <div class="lobby-notch">ARENA<span class="accent">SL</span></div>
        <div class="lobby-head">
          <div class="lobby-title">Pick your vibe</div>
          <div class="lobby-sub">Lenses &amp; filters — try one, then go.</div>
        </div>
        <div class="look-tabs" data-tabs></div>
        <div class="skin-swatches" data-swatches></div>
        <button class="btn-tactile btn-coral lobby-start" type="button" data-start>Start match</button>
      </div>
    `
    this.tabRow = this.root.querySelector<HTMLDivElement>('[data-tabs]')!
    this.swatchRow = this.root.querySelector<HTMLDivElement>('[data-swatches]')!

    for (const group of LOOK_GROUPS) {
      const tab = document.createElement('button')
      tab.className = 'look-tab'
      tab.type = 'button'
      tab.textContent = group.label
      tab.dataset.cat = group.category
      tab.addEventListener('click', () => this.showGroup(group))
      this.tabRow.append(tab)
    }

    this.root.querySelector<HTMLButtonElement>('[data-start]')!
      .addEventListener('click', () => this.onStart())
  }

  private showGroup(group: LookCategoryGroup): void {
    this.activeGroup = group
    for (const tab of this.tabRow.querySelectorAll<HTMLButtonElement>('.look-tab')) {
      tab.classList.toggle('active', tab.dataset.cat === group.category)
    }

    this.swatchRow.innerHTML = ''
    this.buttons.clear()
    this.swatchRow.append(this.noneSwatch())
    for (const item of group.items) this.swatchRow.append(this.swatch(item))
    this.syncActive()
  }

  private noneSwatch(): HTMLButtonElement {
    const btn = document.createElement('button')
    btn.className = 'skin-swatch'
    btn.type = 'button'
    btn.innerHTML = `
      <span class="skin-thumb look-none" style="--accent:#aeb6bd">
        <span class="look-glyph">✕</span>
        <span class="skin-check">✓</span>
      </span>
      <span class="skin-name">None</span>
    `
    btn.addEventListener('click', () => this.select(null, btn))
    this.buttons.set('__none__', btn)
    return btn
  }

  private swatch(item: LookItem): HTMLButtonElement {
    const btn = document.createElement('button')
    btn.className = 'skin-swatch'
    btn.type = 'button'

    const thumbStyle =
      item.category === 'filter' && item.swatch
        ? `--accent:${item.accent};background:${item.swatch}`
        : `--accent:${item.accent}`

    btn.innerHTML = `
      <span class="skin-thumb" style="${thumbStyle}">
        ${item.emoji ? `<span class="look-glyph">${item.emoji}</span>` : ''}
        <span class="skin-check">✓</span>
      </span>
      <span class="skin-name">${item.name}</span>
    `

    btn.addEventListener('click', () => this.select(item, btn))
    this.buttons.set(item.id, btn)
    return btn
  }

  private select(item: LookItem | null, btn: HTMLButtonElement): void {
    this.selectedId = item?.id ?? null
    this.controller.apply(item)
    for (const b of this.buttons.values()) b.classList.remove('active')
    btn.classList.add('active')
  }

  /** Re-highlight the active look after a tab switch (or show None selected). */
  private syncActive(): void {
    const key = this.selectedId ?? '__none__'
    for (const [id, btn] of this.buttons) btn.classList.toggle('active', id === key)
  }
}
