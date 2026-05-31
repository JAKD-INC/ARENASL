import type { Lens } from './lensRenderer.ts'

/**
 * The face-lens catalogue. Each lens is a stack of sticker / effect layers hung
 * on face anchors, optionally paired with a color grade. `emoji` is the picker
 * thumbnail (instant, recognizable, no asset needed).
 *
 * Add a lens by adding an entry — anchors/scales are multiples of the detected
 * face width, so everything tracks distance and head tilt automatically.
 */
export interface LensPreset {
  id: string
  name: string
  accent: string
  emoji: string
  lens: Lens
}

export const LENS_PRESETS: LensPreset[] = [
  {
    id: 'puppy',
    name: 'Puppy',
    accent: '#8a5a3a',
    emoji: '🐶',
    lens: {
      id: 'puppy',
      layers: [
        { sticker: 'dog-ear-l', anchor: 'leftEar', scale: 0.6, offsetY: -0.05 },
        { sticker: 'dog-ear-r', anchor: 'rightEar', scale: 0.6, offsetY: -0.05 },
        { sticker: 'dog-nose', anchor: 'nose', scale: 0.42 },
        { sticker: 'tongue', anchor: 'mouth', scale: 0.36, offsetY: 0.34, trigger: 'mouthOpen' },
      ],
    },
  },
  {
    id: 'kitty',
    name: 'Kitty',
    accent: '#3a3a44',
    emoji: '🐱',
    lens: {
      id: 'kitty',
      layers: [
        { sticker: 'cat-ear-l', anchor: 'leftEar', scale: 0.5, offsetY: -0.12 },
        { sticker: 'cat-ear-r', anchor: 'rightEar', scale: 0.5, offsetY: -0.12 },
        { sticker: 'cat-nose', anchor: 'nose', scale: 0.5 },
        { sticker: 'blush', anchor: 'leftCheek', scale: 0.5 },
        { sticker: 'blush', anchor: 'rightCheek', scale: 0.5 },
      ],
    },
  },
  {
    id: 'shades',
    name: 'Aviator',
    accent: '#16a3a3',
    emoji: '🕶️',
    lens: {
      id: 'shades',
      layers: [{ sticker: 'glasses-round', anchor: 'eyes', scale: 1.25 }],
      filter: { temperature: 0.25, exposure: 0.1, glow: 0.25, vignette: 0.2 },
    },
  },
  {
    id: 'royalty',
    name: 'Royalty',
    accent: '#e0b84a',
    emoji: '👑',
    lens: {
      id: 'royalty',
      layers: [
        { sticker: 'crown', anchor: 'crown', scale: 0.95, offsetY: -0.12 },
      ],
      filter: { duotone: { dark: [0.08, 0.06, 0.02], light: [1.0, 0.84, 0.4], mix: 0.4 }, contrast: 1.1, glow: 0.3 },
    },
  },
  {
    id: 'angel',
    name: 'Angel',
    accent: '#ffe066',
    emoji: '😇',
    lens: {
      id: 'angel',
      layers: [
        { sticker: 'halo', anchor: 'crown', scale: 1.05, offsetY: -0.28, rotate: false },
      ],
      filter: { glow: 0.6, exposure: 0.15, contrast: 0.95, temperature: 0.06 },
    },
  },
  {
    id: 'devil',
    name: 'Devil',
    accent: '#ff3b3b',
    emoji: '😈',
    lens: {
      id: 'devil',
      layers: [
        { sticker: 'devil-horn-l', anchor: 'leftEar', scale: 0.5, offsetY: -0.28 },
        { sticker: 'devil-horn-r', anchor: 'rightEar', scale: 0.5, offsetY: -0.28 },
        { effect: 'fire', anchor: 'mouth', scale: 1, offsetY: 0.2, trigger: 'mouthOpen' },
      ],
      filter: { temperature: 0.3, contrast: 1.15, saturation: 1.2, vignette: 0.4, lift: [0.06, -0.02, -0.02] },
    },
  },
  {
    id: 'heart-eyes',
    name: 'Heart Eyes',
    accent: '#ff2d6b',
    emoji: '😍',
    lens: {
      id: 'heart-eyes',
      layers: [
        { sticker: 'heart', anchor: 'leftEye', scale: 0.42 },
        { sticker: 'heart', anchor: 'rightEye', scale: 0.42 },
        { effect: 'hearts', anchor: 'mouth', scale: 1, offsetY: 0.1, trigger: 'smile' },
      ],
      filter: { saturation: 1.1, exposure: 0.12, glow: 0.3, temperature: 0.08 },
    },
  },
  {
    id: 'starstruck',
    name: 'Starstruck',
    accent: '#ffd23f',
    emoji: '🤩',
    lens: {
      id: 'starstruck',
      layers: [
        { sticker: 'star', anchor: 'leftEye', scale: 0.4 },
        { sticker: 'star', anchor: 'rightEye', scale: 0.4 },
        { effect: 'sparkle', anchor: 'face', scale: 1 },
      ],
      filter: { saturation: 1.15, glow: 0.4, contrast: 1.05 },
    },
  },
  {
    id: 'clown',
    name: 'Clown',
    accent: '#ff5e5e',
    emoji: '🤡',
    lens: {
      id: 'clown',
      layers: [
        { sticker: 'party-hat', anchor: 'crown', scale: 0.8, offsetY: -0.32 },
        { sticker: 'clown-nose', anchor: 'nose', scale: 0.46 },
        { sticker: 'blush', anchor: 'leftCheek', scale: 0.6 },
        { sticker: 'blush', anchor: 'rightCheek', scale: 0.6 },
      ],
    },
  },
  {
    id: 'rainbow',
    name: 'Rainbow',
    accent: '#ff5ce0',
    emoji: '🌈',
    lens: {
      id: 'rainbow',
      layers: [
        { sticker: 'flower-crown', anchor: 'forehead', scale: 1.2, offsetY: -0.08 },
        { effect: 'rainbow', anchor: 'mouth', scale: 1, offsetY: 0.15, trigger: 'mouthOpen' },
      ],
      filter: { saturation: 1.2, vibrance: 0.3, glow: 0.25 },
    },
  },
  {
    id: 'dapper',
    name: 'Dapper',
    accent: '#2b2118',
    emoji: '🧐',
    lens: {
      id: 'dapper',
      layers: [
        { sticker: 'mustache', anchor: 'mouth', scale: 0.72, offsetY: -0.16 },
        { sticker: 'monocle', anchor: 'rightEye', scale: 0.5 },
      ],
      filter: { duotone: { dark: [0.1, 0.08, 0.06], light: [0.96, 0.92, 0.84], mix: 0.5 }, contrast: 1.1, grain: 0.25 },
    },
  },
]
