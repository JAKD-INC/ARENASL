import type { Landmark } from '../game/types.ts'

/**
 * Shared `object-fit: cover` mapping.
 *
 * MediaPipe landmarks are normalized (0..1) to the *source* video frame, but the
 * camera is shown full-screen with `object-fit: cover`. Every overlay (face
 * lenses, the WebGL color pass) has to replicate that same cover transform so
 * what it draws lines up with the pixels the player sees. This is the single
 * source of truth for that math.
 *
 * The landmark provider already mirrors x for the selfie view, so overlays draw
 * in plain (un-mirrored) screen coordinates.
 */

export interface ScreenPoint {
  x: number
  y: number
}

export interface CoverParams {
  /** Uniform scale applied to the source frame. */
  scale: number
  /** Letterbox offset (px) — negative when the frame is cropped. */
  offsetX: number
  offsetY: number
  /** Source frame size in px. */
  videoW: number
  videoH: number
  /** Destination (screen) size in CSS px. */
  width: number
  height: number
}

/** Compute the cover transform for a video painted into a `width`×`height` box. */
export function coverParams(
  video: HTMLVideoElement,
  width = window.innerWidth,
  height = window.innerHeight,
): CoverParams {
  const videoW = video.videoWidth || width
  const videoH = video.videoHeight || height
  const scale = Math.max(width / videoW, height / videoH)
  return {
    scale,
    offsetX: (width - videoW * scale) / 2,
    offsetY: (height - videoH * scale) / 2,
    videoW,
    videoH,
    width,
    height,
  }
}

/** Source-normalized landmark → screen px. */
export function toScreen(p: Landmark, c: CoverParams): ScreenPoint {
  return {
    x: c.offsetX + p.x * c.videoW * c.scale,
    y: c.offsetY + p.y * c.videoH * c.scale,
  }
}
