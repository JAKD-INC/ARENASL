import type { LandmarkMessage } from './net.ts'

type XYZ = number[][]
type Category = { categoryName: string }

/** Assign detected hands to left/right slots by handedness label. */
export function splitHands(
  hands: XYZ[],
  handedness: Category[][],
): { handLeft: XYZ | null; handRight: XYZ | null } {
  let handLeft: XYZ | null = null
  let handRight: XYZ | null = null
  hands.forEach((lms, i) => {
    if (handedness[i]?.[0]?.categoryName === 'Left') handLeft = lms
    else handRight = lms
  })
  return { handLeft, handRight }
}

const toXYZ = (lms: { x: number; y: number; z: number }[]): XYZ =>
  lms.map((l) => [l.x, l.y, l.z])

// MediaPipe hand skeleton (21-landmark) bone pairs.
const HAND_BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                  // palm base
]

/** Draw the detected pose + hand vectors onto a full-viewport canvas, mapping
 *  normalized [0,1] coords through the video's object-fit:cover crop and the
 *  selfie mirror so they line up with the displayed feed. Debug visualization. */
export function drawLandmarks(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  msg: LandmarkMessage,
  opts: { fit?: 'cover' | 'fill'; mirror?: boolean } = {},
): void {
  const { fit = 'cover', mirror = true } = opts
  const ctx = canvas.getContext('2d')
  const { width: cw, height: ch } = canvas
  if (!ctx) return
  ctx.clearRect(0, 0, cw, ch)
  const vw = video.videoWidth
  const vh = video.videoHeight
  if (!vw || !vh) return

  // 'cover' = scale-to-fill + center-crop (the fullscreen feed); 'fill' =
  // stretch to the box (the small reference clip). mirror matches scaleX(-1).
  const scale = fit === 'cover' ? Math.max(cw / vw, ch / vh) : 0
  const sx = fit === 'cover' ? scale : cw / vw
  const sy = fit === 'cover' ? scale : ch / vh
  const ox = (cw - vw * sx) / 2
  const oy = (ch - vh * sy) / 2
  const map = (p: number[]): [number, number] => {
    const x = ox + p[0] * vw * sx
    return [mirror ? cw - x : x, oy + p[1] * vh * sy]
  }

  const dot = (p: number[], r: number, color: string) => {
    const [x, y] = map(p)
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fill()
  }
  const skeleton = (pts: XYZ, color: string) => {
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    for (const [a, b] of HAND_BONES) {
      const [ax, ay] = map(pts[a])
      const [bx, by] = map(pts[b])
      ctx.beginPath()
      ctx.moveTo(ax, ay)
      ctx.lineTo(bx, by)
      ctx.stroke()
    }
    for (const p of pts) dot(p, 3, color)
  }

  // Hands only — pose is still detected for normalization, just not drawn.
  if (msg.handLeft) skeleton(msg.handLeft, '#6cf')
  if (msg.handRight) skeleton(msg.handRight, '#f6c')
}

export async function createDetector() {
  // Dynamic import keeps the browser-only MediaPipe lib out of the module's
  // static graph, so unit tests (node env) can import splitHands without it.
  const { FilesetResolver, HandLandmarker, PoseLandmarker } =
    await import('@mediapipe/tasks-vision')
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm',
  )
  const hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
    numHands: 2, runningMode: 'VIDEO',
  })
  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: '/models/pose_landmarker_lite.task' },
    numPoses: 1, runningMode: 'VIDEO',
  })

  return function detect(video: HTMLVideoElement, t: number): LandmarkMessage {
    const ms = t * 1000
    const hr = hand.detectForVideo(video, ms)
    const pr = pose.detectForVideo(video, ms)
    const pose0 = pr.landmarks[0] ?? null
    const { handLeft, handRight } = splitHands(
      hr.landmarks.map(toXYZ), hr.handedness as Category[][],
    )
    return { t, pose: pose0 ? toXYZ(pose0) : null, handLeft, handRight }
  }
}
