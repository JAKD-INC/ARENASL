/**
 * TEMPORARY sign-demo placeholders for the teaching panel, keyed by word slug
 * (see `slug()` in src/ui/hud.ts — e.g. "THANK YOU" → "thank-you").
 *
 * These hotlink "Sign with Robert" GIFs on GIPHY (free to share) so the overlay
 * shows real signs while we wait for Alex's dataset. They are NOT committed
 * assets and a few are approximate/phrase-level (noted below). When Alex's
 * dataset arrives, drop the real clips into /public/signs/<slug>.(gif|mp4) and
 * delete this map — the HUD falls back to /signs/<slug>.gif automatically.
 */
export const SIGN_GIFS: Record<string, string> = {
  hello: 'https://media.giphy.com/media/3o7TKNKOfKlIhbD3gY/giphy.gif',
  yes: 'https://media.giphy.com/media/l4Jz0THKhQLo61NBK/giphy.gif',
  no: 'https://media.giphy.com/media/l4Jz4faxuS1FiSEV2/giphy.gif',
  'thank-you': 'https://media.giphy.com/media/3o7TKy2yrXO7lLY6E8/giphy.gif', // "thank you for coming"
  please: 'https://media.giphy.com/media/3oz8xxZjw8HJxLcmD6/giphy.gif', // "please help me" (approx)
  name: 'https://media.giphy.com/media/3o7TKDJBonanzESryE/giphy.gif', // "what's your name" (approx)
  friend: 'https://media.giphy.com/media/l4JzhfzkJj6Vm8Fm8/giphy.gif', // "best friends" (approx)
  water: 'https://media.giphy.com/media/l4JyOdLl2Ux9AIUJW/giphy.gif',
  learn: 'https://media.giphy.com/media/NX9XMEk3I1tKoTTPw9/giphy.gif',
  family: 'https://media.giphy.com/media/3o6Zt3FjwPm9LZhKYo/giphy.gif',
  understand: 'https://media.giphy.com/media/l4Jz6lkydXsNEfuWk/giphy.gif',
  practice: 'https://media.giphy.com/media/l0MYDmVHMKnDoKBHi/giphy.gif', // "training" (approx)
  language: 'https://media.giphy.com/media/3o6Ztqnz8kLL7LbRlu/giphy.gif', // generic "sign language" (approx)
}
