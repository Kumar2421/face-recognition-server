import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    // Bind-mounted into the face_ui container; native inotify events don't
    // cross the mount, so poll the filesystem to keep HMR working.
    watch: { usePolling: true, interval: 300 },
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
