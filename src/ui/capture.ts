import type { SignOutcome } from '../game/types.ts'
import type { CaptureCallbacks, VerdictTier } from '../game/signCapture.ts'
import type { SoundEngine } from '../audio/sound.ts'

/**
 * The live capture feedback overlay — the moment-to-moment "the game sees me"
 * loop. Implements {@link CaptureCallbacks} so it can be handed straight to the
 * {@link SignCapture}:
 *  - waiting  → a pulsing "hands up to sign" prompt
 *  - holding  → a filling progress ring ("SIGN!") as the player performs
 *  - resolved → a big tiered PERFECT / GREAT / GOOD / MISS flash + points/combo
 *
 * Owns the capture-related audio cues (attempt, hold riser, verdict, combo) via
 * the {@link SoundEngine}; match-level cues (countdown, music, win/lose) live in
 * main.
 */

const RING_R = 54
const CIRC = 2 * Math.PI * RING_R

const TIER_LABEL: Record<VerdictTier, string> = {
  perfect: 'PERFECT!',
  great: 'GREAT!',
  good: 'GOOD',
  miss: 'MISS',
}

export class CaptureUI implements CaptureCallbacks {
  private root: HTMLElement
  private sound: SoundEngine
  private arc: SVGCircleElement
  private verdict: HTMLDivElement

  constructor(root: HTMLElement, sound: SoundEngine) {
    this.root = root
    this.sound = sound
    root.classList.add('capture')
    root.dataset.cap = 'waiting'
    root.innerHTML = `
      <div class="cap-stage">
        <div class="cap-prompt" data-prompt>
          <span class="cap-prompt-icon">✋</span>
          <span>Hands up to sign</span>
        </div>
        <div class="cap-ring">
          <svg viewBox="0 0 120 120">
            <circle class="cap-ring-track" cx="60" cy="60" r="${RING_R}" />
            <circle class="cap-ring-arc" data-arc cx="60" cy="60" r="${RING_R}"
              transform="rotate(-90 60 60)"
              stroke-dasharray="${CIRC}" stroke-dashoffset="${CIRC}" />
          </svg>
          <div class="cap-ring-label">SIGN!</div>
        </div>
      </div>
      <div class="cap-verdict" data-verdict></div>
    `
    this.arc = root.querySelector<SVGCircleElement>('[data-arc]')!
    this.verdict = root.querySelector<HTMLDivElement>('[data-verdict]')!
  }

  // --- CaptureCallbacks -----------------------------------------------------

  onWaiting = (): void => {
    this.root.dataset.cap = 'waiting'
    this.setProgress(0)
    // Returning to waiting (abort, next word, match end) must kill the hold tone.
    this.sound.stopHold()
  }

  onAttemptStart = (): void => {
    this.root.dataset.cap = 'holding'
    this.setProgress(0)
    this.sound.attemptStart()
    this.sound.startHold()
  }

  onProgress = (p: number): void => {
    this.setProgress(p)
    this.sound.updateHold(p)
  }

  onResolved = (outcome: SignOutcome, tier: VerdictTier): void => {
    this.root.dataset.cap = 'verdict'
    this.sound.stopHold()
    this.sound.verdict(tier, outcome.combo)
    if (outcome.accepted && outcome.combo > 0 && outcome.combo % 5 === 0) {
      this.sound.comboMilestone(outcome.combo)
    }
    this.showVerdict(outcome, tier)
  }

  /** Online: drive the ring straight from the server's recognition strength. */
  online = (strength: number): void => {
    this.root.dataset.cap = strength > 0.05 ? 'holding' : 'waiting'
    this.setProgress(Math.max(0, Math.min(1, strength)))
  }

  // --- rendering ------------------------------------------------------------

  private setProgress(p: number): void {
    this.arc.style.strokeDashoffset = String(CIRC * (1 - Math.max(0, Math.min(1, p))))
  }

  private showVerdict(outcome: SignOutcome, tier: VerdictTier): void {
    const combo = outcome.combo
    const comboChip =
      tier !== 'miss' && combo >= 2 ? `<span class="cap-combo">${combo}× combo</span>` : ''
    const points = outcome.accepted ? `<span class="cap-points">+${outcome.points}</span>` : ''
    this.verdict.className = `cap-verdict tier-${tier}`
    this.verdict.innerHTML = `
      <span class="cap-verdict-label">${TIER_LABEL[tier]}</span>
      ${points}
      ${comboChip}
    `
    // Restart the pop animation.
    void this.verdict.offsetWidth
    this.verdict.classList.add('show')
  }
}
