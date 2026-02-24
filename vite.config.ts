import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend Flask server URL — matches Dockerfile / docker-compose defaults
const BACKEND_URL = process.env.VITE_BACKEND_URL || 'http://localhost:7860'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    watch: {
      ignored: ['**/node_modules/**', '**/deploy_package/**', '**/.git/**', '**/__pycache__/**']
    },
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
        ws: true,
      },
      '/login': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/signup': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/logout': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/auth': {
        target: BACKEND_URL,
        changeOrigin: true,
        cookieDomainRewrite: 'localhost',
      },
      '/uploads': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/forgot-password': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/reset-password': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
    },
  },
})
