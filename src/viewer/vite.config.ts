/**
 * Dev-server config. Bound to 127.0.0.1: this viewer shows frames of a real
 * fixture with identifiable people in them, and there is no reason for it to be reachable
 * from anywhere but the machine it runs on.
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // see: https://vite.dev/config/server-options.html#server-host
    host: "127.0.0.1",
    port: 5173,
  },
});
