import type { ColorFilterRenderer } from '../filters/colorFilter.ts'
import type { LensRenderer } from '../lenses/lensRenderer.ts'
import type { LookItem } from './catalogue.ts'

/**
 * Routes a selected {@link LookItem} to the right rendering subsystem and makes
 * sure only one look is active at a time. Every apply() first resets both layers
 * (color grade, face lens), then turns on just what the chosen item needs —
 * lenses may additionally switch on a paired color grade.
 *
 * The color renderer is optional: when WebGL is unavailable it is null and color
 * grades are skipped, while lenses still work.
 */
export class LookController {
  constructor(
    private video: HTMLVideoElement,
    private color: ColorFilterRenderer | null,
    private lens: LensRenderer,
  ) {}

  /** Apply a look, or clear everything when `item` is null. */
  apply(item: LookItem | null): void {
    // Reset all layers.
    this.video.style.filter = 'none'
    this.color?.setFilter(null)
    this.lens.setLens(null)

    if (!item) return

    switch (item.category) {
      case 'filter':
        this.color?.setFilter(item.filter ?? null)
        break
      case 'lens':
        this.lens.setLens(item.lens ?? null)
        if (item.filter) this.color?.setFilter(item.filter)
        break
    }
  }
}
