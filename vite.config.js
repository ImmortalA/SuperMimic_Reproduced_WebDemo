import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: { port: 5173 },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
  optimizeDeps: { exclude: ["@mujoco/mujoco"] },
  assetsInclude: ["**/*.wasm"],
});
