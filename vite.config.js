import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/PRG-08/',
  build: {
    outDir: 'docs'
  },
  plugins: [react()],
})
