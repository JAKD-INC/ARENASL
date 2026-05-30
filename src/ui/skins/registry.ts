import type { SkinDefinition } from '../../game/types.ts'

/**
 * The skin / look catalogue offered in the lobby.
 *
 *  - `none` clears any look.
 *  - `costume` skins are pose-anchored sprite layers drawn by the renderer
 *    (hat on the head, cape/robe across the shoulders).
 *  - `css-filter` skins recolor the whole video feed (cheap, no landmarks).
 *
 * Costume layers draw in array order, so list back-to-front (cape before hat).
 * Add a costume by dropping sprites in /public/skins and adding an entry here.
 */
export const SKINS: SkinDefinition[] = [
  { id: 'none', name: 'None', kind: 'none', accent: '#94a3b8' },
  {
    id: 'wizard',
    name: 'Wizard',
    kind: 'costume',
    accent: '#6d4ec4',
    thumb: '/skins/wizard-hat.webp',
    costume: [
      { src: '/skins/wizard-cape.webp', anchor: 'torso', scale: 2.1, offsetY: -0.4 },
      { src: '/skins/wizard-hat.webp', anchor: 'head', scale: 1.2, offsetY: -0.3 },
    ],
  },
  {
    id: 'noir',
    name: 'Noir',
    kind: 'css-filter',
    cssFilter: 'grayscale(1) contrast(1.15)',
    accent: '#e5e7eb',
  },
  {
    id: 'vapor',
    name: 'Vapor',
    kind: 'css-filter',
    cssFilter: 'saturate(1.6) hue-rotate(280deg) contrast(1.05)',
    accent: '#f0abfc',
  },
  {
    id: 'arcade',
    name: 'Arcade',
    kind: 'css-filter',
    cssFilter: 'saturate(1.8) contrast(1.2) brightness(1.05)',
    accent: '#34d399',
  },
]

export const DEFAULT_SKIN = SKINS[0]
