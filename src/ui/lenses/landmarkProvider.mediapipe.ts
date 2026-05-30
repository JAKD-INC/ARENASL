import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { Blendshapes, Landmark, LandmarkFrame, LandmarkProvider } from '../../game/types.ts'

/**
 * Standalone {@link LandmarkProvider} backed by MediaPipe's FaceLandmarker:
 * 478 face-mesh points + ARKit-style blendshapes (expression scores), which
 * drive the face lenses and their expression triggers.
 *
 * Runs on a rAF loop, only when the video presents a fresh frame. In production
 * Alex's recognizer can implement this same interface so MediaPipe runs once.
 *
 * Landmark x is mirrored here (1 - x) so coordinates match the mirrored selfie
 * view the player actually sees.
 */

// Pinned to the installed @mediapipe/tasks-vision version (see package.json).
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

export class MediaPipeLandmarkProvider implements LandmarkProvider {
  private video: HTMLVideoElement
  private face: FaceLandmarker | null = null
  private frame: LandmarkFrame | null = null
  private rafId = 0
  private lastVideoTime = -1

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  async start(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
    this.face = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    })
    this.loop()
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.face?.close()
    this.face = null
    this.frame = null
  }

  latest(): LandmarkFrame | null {
    return this.frame
  }

  private loop = (): void => {
    this.rafId = requestAnimationFrame(this.loop)
    const face = this.face
    if (!face || this.video.readyState < 2) return

    // Only run inference when the camera presents a new frame.
    if (this.video.currentTime === this.lastVideoTime) return
    this.lastVideoTime = this.video.currentTime

    const ts = performance.now()
    const result = face.detectForVideo(this.video, ts)
    const faceLm = result.faceLandmarks[0]

    this.frame = {
      face: faceLm ? faceLm.map(toMirrored) : null,
      hands: [],
      pose: null,
      blendshapes: readBlendshapes(result.faceBlendshapes?.[0]?.categories),
      atMs: ts,
    }
  }
}

function toMirrored(p: NormalizedLandmark): Landmark {
  return { x: 1 - p.x, y: p.y, z: p.z }
}

function readBlendshapes(
  categories: { categoryName: string; score: number }[] | undefined,
): Blendshapes | null {
  if (!categories) return null
  const out: Blendshapes = {}
  for (const c of categories) out[c.categoryName] = c.score
  return out
}
