import { defineConfig } from 'vite'

// In dev, proxy the backend (be-server, :8001) so the frontend talks to it
// same-origin — no CORS, and the WS upgrade just works. Override the target with
// ARENASL_BACKEND if the server runs elsewhere.
const BACKEND = process.env.ARENASL_BACKEND ?? 'http://localhost:8001'
const WS_BACKEND = BACKEND.replace(/^http/, 'ws')

export default defineConfig({
  server: {
    host: true,
    proxy: {
      '/ws': { target: WS_BACKEND, ws: true },
      '/auth': { target: BACKEND },
      '/signs': { target: BACKEND },
      '/replay': { target: BACKEND },
      '/clips': { target: BACKEND },
    },
  },
})
