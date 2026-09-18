/** Entry point: mounts the app. */
// createRoot: https://react.dev/reference/react-dom/client/createRoot
// StrictMode runs effects twice in dev, on purpose:
// https://react.dev/reference/react/StrictMode
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
