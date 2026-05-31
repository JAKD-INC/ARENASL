import type { ColorFilter } from '../filters/colorFilter.ts'
import type { Lens } from '../lenses/lensRenderer.ts'
import { COLOR_PRESETS } from '../filters/presets.ts'
import { LENS_PRESETS } from '../lenses/presets.ts'

/**
 * One flat catalogue of every "look" the lobby offers, unified across the two
 * mechanisms (face lenses, color grades) so the picker can list them by category
 * and the {@link LookController} can route each to the right renderer.
 */

export type LookCategory = 'lens' | 'filter'

export interface LookItem {
  id: string
  name: string
  category: LookCategory
  accent: string
  /** Emoji thumbnail (lenses). */
  emoji?: string
  /** CSS gradient thumbnail (filters). */
  swatch?: string
  filter?: ColorFilter
  lens?: Lens
}

const lensItems: LookItem[] = LENS_PRESETS.map((p) => ({
  id: `lens:${p.id}`,
  name: p.name,
  category: 'lens',
  accent: p.accent,
  emoji: p.emoji,
  lens: p.lens,
  // Lenses may pair a color grade; surface it so the controller applies it too.
  filter: p.lens.filter,
}))

const filterItems: LookItem[] = COLOR_PRESETS.map((p) => ({
  id: `filter:${p.id}`,
  name: p.name,
  category: 'filter',
  accent: p.accent,
  swatch: p.swatch,
  filter: p.filter,
}))

export interface LookCategoryGroup {
  category: LookCategory
  label: string
  items: LookItem[]
}

const ALL_GROUPS: LookCategoryGroup[] = [
  { category: 'lens', label: 'Lenses', items: lensItems },
  { category: 'filter', label: 'Filters', items: filterItems },
]

/** Grouped for the picker's category tabs (skips empty groups). */
export const LOOK_GROUPS: LookCategoryGroup[] = ALL_GROUPS.filter((g) => g.items.length > 0)
