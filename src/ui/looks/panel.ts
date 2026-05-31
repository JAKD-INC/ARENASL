import { LOOK_GROUPS, type LookCategoryGroup, type LookItem } from './catalogue.ts'
import type { LookController } from './controller.ts'

/**
 * Reusable looks carousel — category tabs + a horizontally-scrolling swatch row
 * with a leading "None" chip. Picks apply to the live camera immediately through
 * the {@link LookController}; only one look is active at a time and the selection
 * persists across tabs. Embedded by the lobby room and the practice looks sheet
 * (no surrounding chrome of its own).
 */
export class LooksPanel {
  private controller: LookController
  private tabRow!: HTMLDivElement
  private swatchRow!: HTMLDivElement
  private activeGroup: LookCategoryGroup = LOOK_GROUPS[0]
  private selectedId: string | null = null
  private buttons = new Map<string, HTMLButtonElement>()

  constructor(root: HTMLElement, controller: LookController) {
    this.controller = controller
    root.classList.add('looks-panel')
    root.innerHTML = `
      <div class="look-tabs" data-tabs></div>
      <div class="skin-swatches" data-swatches></div>
    `
    this.tabRow = root.querySelector<HTMLDivElement>('[data-tabs]')!
    this.swatchRow = root.querySelector<HTMLDivElement>('[data-swatches]')!

    for (const group of LOOK_GROUPS) {
      const tab = document.createElement('button')
      tab.className = 'look-tab'
      tab.type = 'button'
      tab.textContent = group.label
      tab.dataset.cat = group.category
      tab.addEventListener('click', () => this.showGroup(group))
      this.tabRow.append(tab)
    }
    this.showGroup(this.activeGroup)
  }

  private showGroup(group: LookCategoryGroup): void {
    this.activeGroup = group
    for (const tab of this.tabRow.querySelectorAll<HTMLButtonElement>('.look-tab')) {
      tab.classList.toggle('active', tab.dataset.cat === group.category)
    }
    this.swatchRow.replaceChildren(this.noneSwatch(), ...group.items.map((i) => this.swatch(i)))
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

  private syncActive(): void {
    const key = this.selectedId ?? '__none__'
    for (const [id, btn] of this.buttons) btn.classList.toggle('active', id === key)
  }
}
