/**
 * Friendly auth: the UI only ever asks for a display name, but the backend
 * requires a real account + JWT. So we manage a per-device throwaway account —
 * register once, cache the token (valid ~7 days), reuse it on return. This keeps
 * the "just enter a name" UX while talking to the real `be-server`.
 *
 * Requests are same-origin (the dev server proxies /auth → :8001).
 */

export interface Identity {
  token: string
  playerId: number
  displayName: string
  elo: number
}

const TOKEN_KEY = 'arenasl.token'

interface CachedToken {
  token: string
  displayName: string
}

/** Ensure we have a valid token for `displayName`, registering if needed. */
export async function ensureAuth(displayName: string): Promise<Identity> {
  const cached = readToken()
  if (cached) {
    const who = await fetchMe(cached.token)
    if (who) return { token: cached.token, ...who, displayName: cached.displayName || who.displayName }
  }
  // New device or expired token → register a fresh per-device account.
  const email = `${slug(displayName) || 'player'}-${rand()}@arena.sl`
  const password = `pw-${rand()}-${rand()}`
  const token = await register(email, password, displayName)
  const who = await fetchMe(token)
  if (!who) throw new Error('auth: /me failed right after register')
  writeToken({ token, displayName })
  return { token, displayName, playerId: who.playerId, elo: who.elo }
}

async function register(email: string, password: string, displayName: string): Promise<string> {
  const res = await fetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, display_name: displayName, experience: 'beginner' }),
  })
  if (!res.ok) throw new Error(`auth: register failed (${res.status})`)
  const body = (await res.json()) as { access_token: string }
  return body.access_token
}

async function fetchMe(token: string): Promise<Omit<Identity, 'token' | 'displayName'> & { displayName: string } | null> {
  try {
    const res = await fetch('/auth/me', { headers: { Authorization: `Bearer ${token}` } })
    if (!res.ok) return null
    const u = (await res.json()) as { player_id: number; display_name: string; elo: number }
    return { playerId: u.player_id, displayName: u.display_name, elo: u.elo }
  } catch {
    return null
  }
}

function readToken(): CachedToken | null {
  try {
    const raw = localStorage.getItem(TOKEN_KEY)
    return raw ? (JSON.parse(raw) as CachedToken) : null
  } catch {
    return null
  }
}
function writeToken(t: CachedToken): void {
  localStorage.setItem(TOKEN_KEY, JSON.stringify(t))
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}
function rand(): string {
  return Math.random().toString(36).slice(2, 8)
}
