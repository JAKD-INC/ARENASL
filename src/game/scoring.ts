/**
 * Pure scoring math. No DOM, no state — easy to retune and unit-test.
 * The live tug-of-war (HP transfer + win detection) lives in the store; this
 * module just supplies the per-sign damage and point values it uses.
 * Constants here are a first pass; tweak after a playtest.
 */

export const MAX_HP = 100

/** Minimum recognition confidence for a sign to count as "landed". */
export const ACCEPT_THRESHOLD = 0.6

/** A perfectly-fast sign; signs at/under this get the full speed bonus. */
const FAST_MS = 1500
/** Past this, the speed bonus is gone. */
const SLOW_MS = 6000

/**
 * Damage a landed sign deals = word complexity scaled by how clean it was.
 * difficulty 1..5 → base 6..30, then multiplied by accuracy. This is the
 * *base* hit; {@link damageRamp} scales it up as the match goes on.
 */
export function damage(difficulty: number, accuracy: number): number {
  if (accuracy < ACCEPT_THRESHOLD) return 0
  const base = difficulty * 6
  return Math.round(base * accuracy)
}

/** Tug damage doubles every this many ms — the escalation clock. */
export const RAMP_INTERVAL_MS = 30_000
/** How much damage multiplies each interval (2 = doubles). */
const RAMP_FACTOR = 2

/**
 * Time-based damage multiplier so every match terminates deterministically
 * (no fixed time cap): hits grow exponentially, `RAMP_FACTOR ^ (elapsed /
 * RAMP_INTERVAL_MS)`. 1× at the start, 2× by 30s, 4× by 60s, 8× by 90s… Once
 * the multiplier pushes a single landed sign past the opponent's remaining HP
 * it's a knockout, so HP is guaranteed to reach 0. Continuous (not stepwise)
 * so there are no sudden difficulty cliffs.
 */
export function damageRamp(elapsedMs: number): number {
  return RAMP_FACTOR ** (Math.max(0, elapsedMs) / RAMP_INTERVAL_MS)
}

/** Speed component, 0..1: full at FAST_MS, linearly fades to 0 by SLOW_MS. */
function speedFactor(timeMs: number): number {
  if (timeMs <= FAST_MS) return 1
  if (timeMs >= SLOW_MS) return 0
  return (SLOW_MS - timeMs) / (SLOW_MS - FAST_MS)
}

/**
 * Points awarded for a landed sign: accuracy + speed, lifted by the combo
 * multiplier. A missed sign scores nothing.
 */
export function points(accuracy: number, timeMs: number, combo: number): number {
  if (accuracy < ACCEPT_THRESHOLD) return 0
  const accuracyPts = accuracy * 100
  const speedPts = speedFactor(timeMs) * 50
  const comboMult = 1 + Math.min(combo, 10) * 0.1
  return Math.round((accuracyPts + speedPts) * comboMult)
}
