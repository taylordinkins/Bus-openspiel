#!/usr/bin/env bash
# Start the API server and the web dev server together. Ctrl-C stops both.
# Usage: tools/dev.sh   (API on :8000, web on :5173; the vite proxy expects those ports)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d web/node_modules ]; then
  echo "Installing web dependencies (npm ci)..."
  (cd web && npm ci)
fi

for port in 8000 5173; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is already in use. Stop whatever is running there first." >&2
    echo "  (find it with: ss -ltnp | grep :$port)" >&2
    exit 1
  fi
done

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

uv run uvicorn server.app:app --reload --port 8000 &
pids+=($!)

# Wait for the API before starting the UI, so the first page load doesn't hit a dead proxy.
for _ in $(seq 1 60); do
  if curl -fs http://127.0.0.1:8000/openapi.json >/dev/null 2>&1; then break; fi
  sleep 0.5
done

(cd web && npm run dev) &
pids+=($!)

echo
echo "Bus is starting:  http://localhost:5173   (API http://127.0.0.1:8000)"
echo "Ctrl-C stops both."
wait -n
