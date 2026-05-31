import type { ColorFilter } from './colorFilter.ts'

/**
 * The color-look catalogue. Each entry is just a {@link ColorFilter} parameter
 * set fed to the one grading shader — add a look by adding an object here.
 *
 * `accent` drives the picker swatch tint; `swatch` is an optional 2-stop CSS
 * gradient so a look reads at a glance before the camera preview kicks in.
 */
export interface ColorPreset {
  id: string
  name: string
  accent: string
  swatch: string
  filter: ColorFilter
}

export const COLOR_PRESETS: ColorPreset[] = [
  {
    id: 'golden-hour',
    name: 'Golden Hour',
    accent: '#ffb454',
    swatch: 'linear-gradient(135deg,#ffd28a,#ff8a3d)',
    filter: {
      temperature: 0.35, exposure: 0.12, contrast: 1.05, vibrance: 0.25,
      glow: 0.35, gain: [0.06, 0.03, -0.02], vignette: 0.25, vignetteSoftness: 0.4,
    },
  },
  {
    id: 'teal-orange',
    name: 'Cinema',
    accent: '#16a3a3',
    swatch: 'linear-gradient(135deg,#0e3a44,#ff8e53)',
    filter: {
      contrast: 1.12, saturation: 1.05, vibrance: 0.2,
      lift: [-0.04, 0.02, 0.07], gain: [0.08, 0.02, -0.05],
      vignette: 0.35, vignetteSoftness: 0.35, sharpen: 0.2,
    },
  },
  {
    id: 'noir',
    name: 'Noir',
    accent: '#cfd4d8',
    swatch: 'linear-gradient(135deg,#e7eaec,#1a1f22)',
    filter: { saturation: 0, contrast: 1.35, exposure: 0.05, grain: 0.4, vignette: 0.45, sharpen: 0.25 },
  },
  {
    id: 'vivid',
    name: 'Vivid',
    accent: '#ff3b6b',
    swatch: 'linear-gradient(135deg,#ff5e62,#ff9966)',
    filter: { saturation: 1.25, vibrance: 0.4, contrast: 1.12, sharpen: 0.25, exposure: 0.05 },
  },
  {
    id: 'vintage',
    name: 'Vintage',
    accent: '#c8a37a',
    swatch: 'linear-gradient(135deg,#d8b48a,#7c5c43)',
    filter: {
      temperature: 0.2, saturation: 0.82, contrast: 0.95, fade: 0.12,
      lift: [0.05, 0.02, -0.03], gain: [0.04, 0.0, -0.04], grain: 0.3, vignette: 0.3,
    },
  },
  {
    id: 'sepia',
    name: 'Sepia',
    accent: '#a07b4e',
    swatch: 'linear-gradient(135deg,#e3c79a,#6b4a28)',
    filter: { duotone: { dark: [0.18, 0.11, 0.05], light: [1.0, 0.86, 0.62], mix: 0.9 }, contrast: 1.05, vignette: 0.25 },
  },
  {
    id: 'cyberpunk',
    name: 'Cyberpunk',
    accent: '#ff2bd6',
    swatch: 'linear-gradient(135deg,#ff2bd6,#23d5ff)',
    filter: {
      contrast: 1.2, saturation: 1.3, glow: 0.6,
      lift: [0.06, -0.03, 0.12], gain: [0.1, -0.04, 0.12], vignette: 0.4, sharpen: 0.2,
    },
  },
  {
    id: 'dreamy',
    name: 'Dreamy',
    accent: '#f3a6d0',
    swatch: 'linear-gradient(135deg,#ffd6f0,#cbb6ff)',
    filter: { glow: 0.7, contrast: 0.92, saturation: 1.05, exposure: 0.15, fade: 0.1, temperature: 0.08 },
  },
  {
    id: 'vhs',
    name: 'VHS',
    accent: '#7df9ff',
    swatch: 'linear-gradient(135deg,#22d3ee,#a78bfa)',
    filter: {
      saturation: 1.2, contrast: 1.08, scanline: 0.25, grain: 0.45,
      lift: [0.04, 0.0, 0.05], gain: [0.05, -0.02, 0.04],
    },
  },
  {
    id: 'frost',
    name: 'Frostbite',
    accent: '#8fd3ff',
    swatch: 'linear-gradient(135deg,#cdeeff,#3b82f6)',
    filter: { temperature: -0.4, tint: -0.05, exposure: 0.12, contrast: 1.08, vibrance: 0.15, gain: [-0.04, 0.0, 0.08] },
  },
  {
    id: 'teal-mint',
    name: 'Mint',
    accent: '#4ecdc4',
    swatch: 'linear-gradient(135deg,#a8f0e6,#1ba39b)',
    filter: { temperature: -0.15, saturation: 1.1, lift: [-0.03, 0.05, 0.03], gain: [-0.04, 0.06, 0.0], contrast: 1.05 },
  },
  {
    id: 'sunburn',
    name: 'Sunburn',
    accent: '#ff6b3d',
    swatch: 'linear-gradient(135deg,#ffd86b,#ff3d3d)',
    filter: { temperature: 0.5, saturation: 1.2, contrast: 1.12, exposure: 0.1, vignette: 0.2 },
  },
  {
    id: 'mono-gold',
    name: 'Gilded',
    accent: '#e0b84a',
    swatch: 'linear-gradient(135deg,#fff1b8,#8a6a1f)',
    filter: { duotone: { dark: [0.08, 0.06, 0.02], light: [1.0, 0.84, 0.4], mix: 1.0 }, contrast: 1.15, glow: 0.3 },
  },
  {
    id: 'pastel',
    name: 'Pastel Pop',
    accent: '#ffc2e2',
    swatch: 'linear-gradient(135deg,#ffe0f0,#c2f0ff)',
    filter: { saturation: 1.15, exposure: 0.2, contrast: 0.95, fade: 0.08, vibrance: 0.2 },
  },
  {
    id: 'infrared',
    name: 'Infrared',
    accent: '#ff5ce0',
    swatch: 'linear-gradient(135deg,#ff5ce0,#5cff9e)',
    filter: { hue: 150, saturation: 1.3, contrast: 1.1, glow: 0.3 },
  },
  {
    id: 'bleach',
    name: 'Bleach',
    accent: '#e9edf0',
    swatch: 'linear-gradient(135deg,#ffffff,#9aa7ad)',
    filter: { saturation: 0.4, contrast: 1.3, exposure: 0.25, sharpen: 0.3, fade: 0.06 },
  },
  {
    id: 'neon-night',
    name: 'Neon Night',
    accent: '#7c5cff',
    swatch: 'linear-gradient(135deg,#3b1d8f,#23d5ff)',
    filter: {
      exposure: -0.15, contrast: 1.2, saturation: 1.35, glow: 0.7,
      lift: [0.0, -0.02, 0.1], vignette: 0.5, vignetteSoftness: 0.45,
    },
  },
  {
    id: 'comic',
    name: 'Comic',
    accent: '#ffd23f',
    swatch: 'linear-gradient(135deg,#ffd23f,#ff5e5e)',
    filter: { posterize: 6, saturation: 1.35, contrast: 1.25, sharpen: 0.5, vignette: 0.2 },
  },
]
