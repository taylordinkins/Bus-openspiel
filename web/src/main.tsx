import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted fonts (offline-safe builds; font-display: swap): condensed display face + text face.
import "@fontsource/barlow-condensed/latin-600.css";
import "@fontsource/barlow-condensed/latin-700.css";
import "@fontsource/barlow-condensed/latin-800.css";
import "@fontsource/barlow/latin-400.css";
import "@fontsource/barlow/latin-400-italic.css";
import "@fontsource/barlow/latin-500.css";
import "@fontsource/barlow/latin-600.css";
import "./theme/tokens.css";
import "./theme/base.css";
import App from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
