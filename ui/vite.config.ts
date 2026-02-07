import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: [
      'face.service.tools.thefusionapps.com',
    ],
    hmr: {
      host: 'face.service.tools.thefusionapps.com',
      protocol: 'wss',
      clientPort: 443,
    },
    origin: 'https://face.service.tools.thefusionapps.com',
  },
})
