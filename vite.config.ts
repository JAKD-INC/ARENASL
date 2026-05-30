/// <reference types="vitest" />
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: true,
    proxy: {
      // Browser talks to same-origin /ws; Vite proxies to the Python server.
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  test: {
    environment: 'node',
  },
})
