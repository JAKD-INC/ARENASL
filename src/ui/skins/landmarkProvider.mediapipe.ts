import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import type { Landmark, LandmarkFrame, LandmarkProvider } from '../../game/types.ts'

/**
 * Standalone {@link LandmarkProvider} backed by MediaPipe's PoseLandmarker.
 *
 * One model gives us head points (nose/eyes/ears) and body points (shoulders,
 * hips) — enough to anchor a full costume (hat on the head, cape/robe across
 * the shoulders). This lets the skins layer develop independently of Alex's
 * recognizer; in production his pipeline can implement the same interface so
 * MediaPipe only runs once.
 *
 * Landmark x is mirrored here (1 - x) so coordinates match the mirrored selfie
 * view the player actually sees.
 */

// Pinned to the installed @mediapipe/tasks-vision version (see package.json).
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

export class MediaPipePoseProvider implements LandmarkProvider {
  private video: HTMLVideoElement
  private landmarker: PoseLandmarker | null = null
  private frame: LandmarkFrame | null = null
  private rafId = 0
  private lastVideoTime = -1

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  async start(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
    this.landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
    })
    this.loop()
  }

  stop(): void {
    cancelAnimationFrame(this.rafId)
    this.landmarker?.close()
    this.landmarker = null
    this.frame = null
  }

  latest(): LandmarkFrame | null {
    return this.frame
  }

  private loop = (): void => {
    this.rafId = requestAnimationFrame(this.loop)
    const lm = this.landmarker
    if (!lm || this.video.readyState < 2) return

    if (this.video.currentTime === this.lastVideoTime) return
    this.lastVideoTime = this.video.currentTime

    const result = lm.detectForVideo(this.video, performance.now())
    const pose = result.landmarks[0]
    this.frame = {
      face: null,
      hands: [],
      pose: pose ? pose.map(toMirrored) : null,
      atMs: performance.now(),
    }
  }
}

function toMirrored(p: { x: number; y: number }): Landmark {
  return { x: 1 - p.x, y: p.y }
}
