import type { SignOutcome } from './types.ts'

/**
 * The client-side heuristic recognizer that used to live here (it accepted any
 * raised hand held for ~1s, regardless of the actual sign) has been REMOVED:
 * recognition is now exclusively server-side (see {@link PracticeDriver} /
 * {@link NetMatchDriver}, which stream raw landmarks to the backend embedding
 * recognizer). What remains is the shared verdict vocabulary the capture UI and
 * those drivers still speak.
 */

export type VerdictTier = 'perfect' | 'great' | 'good' | 'miss'

/** Lifecycle hooks the {@link CaptureUI} implements so a driver can drive the
 * prompt, the hold ring, and the verdict display. */
export interface CaptureCallbacks {
  /** Racing began or a previous attempt finished — ready for the next sign. */
  onWaiting?: () => void
  /** Hands raised; a capture has begun. */
  onAttemptStart?: () => void
  /** Hold progress 0..1 while the player performs. */
  onProgress?: (p: number) => void
  /** Attempt resolved into an outcome + a display tier. */
  onResolved?: (outcome: SignOutcome, tier: VerdictTier) => void
}
