import {
  type Category,
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
  PoseLandmarker,
} from '@mediapipe/tasks-vision'
import type { Blendshapes, Landmark, LandmarkFrame, LandmarkProvider } from '../../game/types.ts'
import type { LandmarkPayload } from '../../net/protocol.ts'

/**
 * Standalone {@link LandmarkProvider} backed by MediaPipe:
 *  - {@link FaceLandmarker} → 478 face-mesh points + blendshapes for cosmetic
 *    face lenses.
 *  - {@link HandLandmarker} → 21 points/hand for the local capture loop.
 *  - {@link PoseLandmarker} → 33 body points used (with the hands) by the
 *    server-side ASL recognizer for normalization.
 *
 * Two views come out of the one loop:
 *  - {@link latest} — cosmetic frame, x **mirrored** (1 - x) to match the
 *    selfie view the overlays draw onto.
 *  - {@link latestRecognition} — **raw, un-mirrored** `[x,y,z]` with pose +
 *    hands split by handedness, in the exact shape the backend expects
 *    (matches Alex's `landmarks.ts`). This is what we stream during a duel.
 */

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

export class MediaPipeLandmarkProvider implements LandmarkProvider {
  private video: HTMLVideoElement
  private face: FaceLandmarker | null = null
  private hands: HandLandmarker | null = null
  private pose: PoseLandmarker | null = null
  private frame: LandmarkFrame | null = null
  private recognition: LandmarkPayload | null = null
  private rafId = 0
  private lastVideoTime = -1

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  async start(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
    const [face, hands, pose] = await Promise.all([
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false,
      }),
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
      }),
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      }),
    ])
    this.face = face
    this.hands = hands
    this.pose = pose
    this.loop()
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.face?.close()
    this.hands?.close()
    this.pose?.close()
    this.face = null
    this.hands = null
    this.pose = null
    this.frame = null
    this.recognition = null
  }

  latest(): LandmarkFrame | null {
    return this.frame
  }

  /** Raw landmark frame for streaming to the server recognizer (or null). */
  latestRecognition(): LandmarkPayload | null {
    return this.recognition
  }

  private loop = (): void => {
    this.rafId = requestAnimationFrame(this.loop)
    const { face, hands, pose } = this
    if (!face || !hands || !pose || this.video.readyState < 2) return

    if (this.video.currentTime === this.lastVideoTime) return
    this.lastVideoTime = this.video.currentTime

    const ts = performance.now()
    const faceResult = face.detectForVideo(this.video, ts)
    const handResult = hands.detectForVideo(this.video, ts)
    const poseResult = pose.detectForVideo(this.video, ts)

    // --- cosmetic (mirrored) frame for overlays ---
    const faceLm = faceResult.faceLandmarks[0]
    this.frame = {
      face: faceLm ? faceLm.map(toMirrored) : null,
      hands: handResult.landmarks.map((hand) => hand.map(toMirrored)),
      pose: null,
      blendshapes: readBlendshapes(faceResult.faceBlendshapes?.[0]?.categories),
      atMs: ts,
    }

    // --- raw recognition frame (un-mirrored; hands split by handedness) ---
    let handLeft: number[][] | null = null
    let handRight: number[][] | null = null
    const handed = handResult.handednesses as Category[][]
    handResult.landmarks.forEach((lms, i) => {
      const arr = lms.map(toXYZ)
      if (handed[i]?.[0]?.categoryName === 'Left') handLeft = arr
      else handRight = arr
    })
    const pose0 = poseResult.landmarks[0]
    this.recognition = {
      t: ts,
      pose: pose0 ? pose0.map(toXYZ) : null,
      handLeft,
      handRight,
    }
  }
}

function toMirrored(p: NormalizedLandmark): Landmark {
  return { x: 1 - p.x, y: p.y, z: p.z }
}

function toXYZ(p: NormalizedLandmark): number[] {
  return [p.x, p.y, p.z]
}

function readBlendshapes(
  categories: { categoryName: string; score: number }[] | undefined,
): Blendshapes | null {
  if (!categories) return null
  const out: Blendshapes = {}
  for (const c of categories) out[c.categoryName] = c.score
  return out
}
