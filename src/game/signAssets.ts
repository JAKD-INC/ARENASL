/**
 * RETIRED: sign-demo placeholders for the teaching panel.
 *
 * The HUD now loads per-gloss reference clips directly from be-server at
 * `/clips/<slug>.mp4` (see src/ui/hud.ts), so the old GIPHY placeholder map is
 * no longer used. This empty map is kept only so the module stays importable;
 * it can be removed once nothing references it.
 */
export const SIGN_GIFS: Record<string, string> = {}
