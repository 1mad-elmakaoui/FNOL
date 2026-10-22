import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (`npm run dev`), /api is proxied to the FastAPI backend, exactly like nginx does in Docker.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_URL ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
