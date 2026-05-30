import type { SkinDefinition } from '../../game/types.ts'
import type { SkinRenderer } from './renderer.ts'
import { DEFAULT_SKIN, SKINS } from './registry.ts'
import { keyedSprite } from './sprite.ts'

/**
 * Pre-match lobby: a friendly bottom-sheet where the player picks their camera
 * look and starts the match. Choosing a skin applies it immediately to the live
 * feed:
 *  - `css-filter` skins set the video element's CSS `filter`,
 *  - `face-anchored` skins are handed to the {@link SkinRenderer}.
 * Only one skin is active at a time.
 */
export class SkinPicker {
  private root: HTMLElement
  private video: HTMLVideoElement
  private renderer: SkinRenderer
  private onStart: () => void
  private onChange?: (skin: SkinDefinition) => void
  private buttons = new Map<string, HTMLButtonElement>()
  private selected: SkinDefinition = DEFAULT_SKIN

  constructor(
    root: HTMLElement,
    video: HTMLVideoElement,
    renderer: SkinRenderer,
    onStart: () => void,
    onChange?: (skin: SkinDefinition) => void,
  ) {
    this.root = root
    this.video = video
    this.renderer = renderer
    this.onStart = onStart
    this.onChange = onChange
    this.build()
    this.select(DEFAULT_SKIN)
  }

  private build(): void {
    this.root.innerHTML = `
      <div class="lobby-sheet">
        <div class="lobby-notch">ARENA<span class="accent">SL</span></div>
        <div class="lobby-head">
          <div class="lobby-title">Lookin' good!</div>
          <div class="lobby-sub">Pick your camera look, then go.</div>
        </div>
        <div class="skin-swatches" data-swatches></div>
        <button class="btn-tactile btn-coral lobby-start" type="button" data-start>Start match</button>
      </div>
    `

    const swatches = this.root.querySelector<HTMLDivElement>('[data-swatches]')!
    for (const skin of SKINS) {
      const btn = document.createElement('button')
      btn.className = 'skin-swatch'
      btn.type = 'button'
      btn.innerHTML = `
        <span class="skin-thumb" style="--accent:${skin.accent}">
          <span class="skin-check">✓</span>
        </span>
        <span class="skin-name">${skin.name}</span>
      `
      if (skin.thumb) {
        const cv = keyedSprite(skin.thumb)
        cv.className = 'skin-thumb-img'
        btn.querySelector('.skin-thumb')?.prepend(cv)
      }
      btn.addEventListener('click', () => this.select(skin))
      this.buttons.set(skin.id, btn)
      swatches.append(btn)
    }

    this.root.querySelector<HTMLButtonElement>('[data-start]')!
      .addEventListener('click', () => this.onStart())
  }

  select(skin: SkinDefinition): void {
    this.selected = skin
    for (const [id, btn] of this.buttons) btn.classList.toggle('active', id === skin.id)

    if (skin.kind === 'css-filter') {
      this.video.style.filter = skin.cssFilter ?? 'none'
      this.renderer.setSkin(null)
    } else if (skin.kind === 'costume') {
      this.video.style.filter = 'none'
      this.renderer.setSkin(skin)
    } else {
      this.video.style.filter = 'none'
      this.renderer.setSkin(null)
    }
    this.onChange?.(skin)
  }

  current(): SkinDefinition {
    return this.selected
  }

  show(): void {
    this.root.classList.remove('hidden')
  }

  hide(): void {
    this.root.classList.add('hidden')
  }
}
