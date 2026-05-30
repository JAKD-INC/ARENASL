import type { Blendshapes, Landmark } from '../../game/types.ts'
import { toScreen, type CoverParams, type ScreenPoint } from '../coverFit.ts'

/**
 * Turns a raw MediaPipe face mesh into a stable "rig" of screen-space anchor
 * points the lens renderer can hang stickers on, plus the head roll angle and a
 * face-size reference for scaling. Built fresh each frame.
 *
 * Left/right are resolved by on-screen position (not landmark index) so the rig
 * is robust to the selfie mirroring the provider applies.
 */

// Canonical MediaPipe FaceLandmarker indices.
const IDX = {
  foreheadTop: 10,
  chin: 152,
  noseTip: 1,
  betweenEyes: 168,
  faceLeft: 234,
  faceRight: 454,
  eyeOuterA: 33,
  eyeOuterB: 263,
  cheekA: 205,
  cheekB: 425,
  upperLip: 13,
  lowerLip: 14,
  leftIris: 468, // present only on the 478-point (iris) model
  rightIris: 473,
} as const

export type FaceAnchor =
  | 'forehead'
  | 'crown'
  | 'leftEar'
  | 'rightEar'
  | 'leftEye'
  | 'rightEye'
  | 'eyes'
  | 'nose'
  | 'mouth'
  | 'chin'
  | 'leftCheek'
  | 'rightCheek'
  | 'face'

export type Trigger = 'always' | 'mouthOpen' | 'smile' | 'browsUp' | 'blink' | 'tongueOut'

export interface FaceRig {
  /** Reference width (px) ≈ face width; sticker sizes are multiples of this. */
  width: number
  /** Reference height (px) ≈ forehead→chin. */
  height: number
  /** Head roll in radians (0 = level). */
  angle: number
  anchor(a: FaceAnchor): ScreenPoint
}

export function buildFaceRig(face: Landmark[], cover: CoverParams): FaceRig {
  const pt = (i: number): ScreenPoint => toScreen(face[i] ?? face[0], cover)

  const faceL = pt(IDX.faceLeft)
  const faceR = pt(IDX.faceRight)
  const foreheadTop = pt(IDX.foreheadTop)
  const chin = pt(IDX.chin)
  const noseTip = pt(IDX.noseTip)
  const between = pt(IDX.betweenEyes)

  // Eyes: prefer iris centers (478-pt model), else fall back to outer corners.
  const hasIris = face.length > IDX.rightIris
  const eyeA = hasIris ? pt(IDX.leftIris) : pt(IDX.eyeOuterA)
  const eyeB = hasIris ? pt(IDX.rightIris) : pt(IDX.eyeOuterB)
  const [leftEye, rightEye] = eyeA.x <= eyeB.x ? [eyeA, eyeB] : [eyeB, eyeA]

  const cheekRaw = [pt(IDX.cheekA), pt(IDX.cheekB)]
  const [leftCheek, rightCheek] = cheekRaw[0].x <= cheekRaw[1].x ? cheekRaw : [cheekRaw[1], cheekRaw[0]]

  const mouth = mid(pt(IDX.upperLip), pt(IDX.lowerLip))

  const width = dist(faceL, faceR)
  const height = dist(foreheadTop, chin)
  const right = unit(sub(rightEye, leftEye))
  const up: ScreenPoint = { x: right.y, y: -right.x } // perpendicular, points up-screen
  const angle = Math.atan2(right.y, right.x)

  const addUp = (p: ScreenPoint, d: number): ScreenPoint => ({ x: p.x + up.x * d, y: p.y + up.y * d })
  const addRight = (p: ScreenPoint, d: number): ScreenPoint => ({ x: p.x + right.x * d, y: p.y + right.y * d })

  const points: Record<FaceAnchor, ScreenPoint> = {
    forehead: foreheadTop,
    crown: addUp(foreheadTop, height * 0.5),
    leftEar: addRight(addUp(foreheadTop, height * 0.1), -width * 0.42),
    rightEar: addRight(addUp(foreheadTop, height * 0.1), width * 0.42),
    leftEye,
    rightEye,
    eyes: between,
    nose: noseTip,
    mouth,
    chin,
    leftCheek,
    rightCheek,
    face: mid(between, chin),
  }

  return { width, height, angle, anchor: (a) => points[a] }
}

/** Whether an expression {@link Trigger} is currently active. */
export function triggerActive(trigger: Trigger, b: Blendshapes | null): boolean {
  if (trigger === 'always') return true
  if (!b) return false
  switch (trigger) {
    case 'mouthOpen':
      return (b.jawOpen ?? 0) > 0.4
    case 'smile':
      return ((b.mouthSmileLeft ?? 0) + (b.mouthSmileRight ?? 0)) / 2 > 0.4
    case 'browsUp':
      return Math.max(b.browInnerUp ?? 0, b.browOuterUpLeft ?? 0, b.browOuterUpRight ?? 0) > 0.4
    case 'blink':
      return Math.max(b.eyeBlinkLeft ?? 0, b.eyeBlinkRight ?? 0) > 0.5
    case 'tongueOut':
      return (b.tongueOut ?? 0) > 0.3
  }
}

function sub(a: ScreenPoint, b: ScreenPoint): ScreenPoint {
  return { x: a.x - b.x, y: a.y - b.y }
}
function dist(a: ScreenPoint, b: ScreenPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}
function mid(a: ScreenPoint, b: ScreenPoint): ScreenPoint {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
}
function unit(a: ScreenPoint): ScreenPoint {
  const m = Math.hypot(a.x, a.y) || 1
  return { x: a.x / m, y: a.y / m }
}
