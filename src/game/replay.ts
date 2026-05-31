import type { ReplayData } from './types.ts'

/**
 * Replay fetch seam.
 *
 * The overlay never records video. Each player's webcam stream is sent to the
 * server for visual analysis (Alex) and saved there for replay (David). At
 * results time we fetch the opponent's saved replay — its video URL plus the
 * opponent's sign timeline — and load it on the results screen.
 *
 * The endpoint and response shape below are a PLACEHOLDER and will be finalized
 * when this plan is consolidated with the server infra plan.
 *
 * For standalone dev (no server), `setMockReplay` lets the mock driver inject a
 * synthetic timeline so the whole results flow runs end-to-end. The replay
 * `<video>` simply shows a "loading replay" state while `videoUrl` is null.
 */

let mockReplay: ReplayData | null = null

/** Dev-only: inject the opponent replay the mock driver generated. */
export function setMockReplay(data: ReplayData): void {
  mockReplay = data
}

export async function fetchReplay(matchId: string): Promise<ReplayData> {
  if (mockReplay) return mockReplay

  // --- production path (TBD with server infra plan) ---
  const res = await fetch(`/api/match/${matchId}/replay`)
  if (!res.ok) throw new Error(`fetchReplay failed: ${res.status}`)
  return (await res.json()) as ReplayData
}
