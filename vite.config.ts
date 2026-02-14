import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    watch: {
      ignored: ['**/node_modules/**', '**/deploy_package/**', '**/.git/**', '**/__pycache__/**']
    },
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true
      },
      '/login': {
        target: 'http://localhost:7860',
        changeOrigin: true
      },
      '/signup': {
        target: 'http://localhost:7860',
        changeOrigin: true
      },
      '/logout': {
        target: 'http://localhost:7860',
        changeOrigin: true
      },
      '/auth': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        cookieDomainRewrite: 'localhost'
      },
      '/uploads': {
        target: 'http://localhost:7860',
        changeOrigin: true
      }
    }
  }
})
