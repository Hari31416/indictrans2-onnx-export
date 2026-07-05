import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'

// Custom plugin to serve the root scratch directory in development
const serveScratchPlugin = () => ({
  name: 'serve-scratch',
  configureServer(server: any) {
    server.middlewares.use((req: any, res: any, next: any) => {
      // Handle requests to /scratch/* or /indictrans2-onnx-export/scratch/*
      const match = req.url.match(/^\/(?:indictrans2-onnx-export\/)?scratch\/(.+)$/)
      if (match) {
        // Clean query params (e.g. ?v=1)
        const cleanPath = match[1].split('?')[0]
        const filePath = path.resolve(__dirname, '../scratch', cleanPath)
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          // Set appropriate content type
          if (filePath.endsWith('.onnx')) {
            res.setHeader('Content-Type', 'application/octet-stream')
          } else if (filePath.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm')
          } else if (filePath.endsWith('.json')) {
            res.setHeader('Content-Type', 'application/json')
          } else if (filePath.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript')
          }
          res.setHeader('Access-Control-Allow-Origin', '*')
          fs.createReadStream(filePath).pipe(res)
          return
        }
      }
      next()
    })
  }
})

// https://vite.dev/config/
export default defineConfig({
  base: '/indictrans2-onnx-export/',
  plugins: [react(), serveScratchPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
