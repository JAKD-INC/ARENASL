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
