#!/usr/bin/env bash
# =============================================================================
# scripts/start_all.sh — Orchestrator: start all three services
#
# Services started:
#   [1] FastAPI backend   →  http://localhost:8000   (logs: logs/api.log)
#   [2] FastMCP server    →  http://localhost:8001   (logs: logs/mcp.log)
#   [3] Streamlit UI      →  http://localhost:8501   (logs: logs/ui.log)
#
# Usage:
#   chmod +x scripts/start_all.sh
#   ./scripts/start_all.sh           # start all services
#   ./scripts/start_all.sh --no-mcp  # skip MCP server
#   ./scripts/start_all.sh --ui-only # Streamlit only
#
# Stop all:
#   Press Ctrl+C  (traps SIGINT/SIGTERM and kills all child processes)
#   Or run:  ./scripts/start_all.sh --stop
# =============================================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.services.pid"

# ── Ports ─────────────────────────────────────────────────────────────────────
API_PORT=8000
MCP_PORT=8001
UI_PORT=8501

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Flags (parse CLI args) ─────────────────────────────────────────────────────
START_API=true
START_MCP=true
START_UI=true

for arg in "$@"; do
  case "$arg" in
    --no-mcp)   START_MCP=false ;;
    --no-api)   START_API=false ;;
    --no-ui)    START_UI=false  ;;
    --ui-only)  START_API=false; START_MCP=false ;;
    --api-only) START_MCP=false; START_UI=false  ;;
    --stop)     stop_all; exit 0 ;;
    --help|-h)
      echo "Usage: $0 [--no-mcp] [--no-api] [--no-ui] [--ui-only] [--api-only] [--stop]"
      exit 0 ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}✔${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
err()  { echo -e "${RED}✘${RESET}  $*"; }

port_in_use() {
  lsof -i ":$1" -sTCP:LISTEN -t >/dev/null 2>&1
}

wait_for_port() {
  local port=$1 name=$2 retries=20 i=0
  while ! port_in_use "$port"; do
    sleep 0.5
    ((i++))
    if [ $i -ge $retries ]; then
      err "$name did not start on port $port within ${retries} attempts."
      return 1
    fi
  done
  ok "$name is up  →  http://localhost:$port"
}

stop_all() {
  log "Stopping all services..."
  if [ -f "$PID_FILE" ]; then
    while read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && echo "  killed PID $pid"
      fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
  # Fallback: kill by process name if PID file is stale
  pkill -f "uvicorn src.api.main"  2>/dev/null || true
  pkill -f "src.mcp.server"        2>/dev/null || true
  pkill -f "streamlit run"         2>/dev/null || true
  log "All services stopped."
}

# ── Trap Ctrl+C and SIGTERM → clean shutdown ───────────────────────────────────
cleanup() {
  echo ""
  log "Shutting down all services..."
  stop_all
  exit 0
}
trap cleanup SIGINT SIGTERM

# ── Pre-flight checks ──────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║     PDF & Turtle RAG Chatbot — Orchestrator  ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# Check Ollama is running
log "Checking Ollama..."
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  ok "Ollama is running at http://localhost:11434"
else
  err "Ollama is NOT running. Start it first:"
  echo "      ollama serve"
  echo ""
  warn "Continuing anyway — services will fail when they first call Ollama."
fi

# Create logs directory
mkdir -p "$LOG_DIR"
log "Logs will be written to: $LOG_DIR/"

# Clear stale PID file
rm -f "$PID_FILE"

# Check for port conflicts before starting
check_port() {
  local port=$1 name=$2
  if port_in_use "$port"; then
    warn "Port $port already in use — $name may already be running."
    warn "Run './scripts/start_all.sh --stop' first if you want a clean restart."
  fi
}

$START_API && check_port $API_PORT "FastAPI"
$START_MCP && check_port $MCP_PORT "FastMCP"
$START_UI  && check_port $UI_PORT  "Streamlit"

echo ""

# ── Start FastAPI ──────────────────────────────────────────────────────────────
if $START_API; then
  log "Starting FastAPI backend..."
  uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port $API_PORT \
    --reload \
    > "$LOG_DIR/api.log" 2>&1 &
  API_PID=$!
  echo "$API_PID" >> "$PID_FILE"
  wait_for_port $API_PORT "FastAPI"
fi

# ── Start FastMCP ──────────────────────────────────────────────────────────────
if $START_MCP; then
  log "Starting FastMCP server..."
  python -m src.mcp.server \
    > "$LOG_DIR/mcp.log" 2>&1 &
  MCP_PID=$!
  echo "$MCP_PID" >> "$PID_FILE"
  wait_for_port $MCP_PORT "FastMCP"
fi

# ── Start Streamlit UI ─────────────────────────────────────────────────────────
if $START_UI; then
  log "Starting Streamlit UI..."
  streamlit run scripts/demo.py \
    --server.port $UI_PORT \
    --server.address 0.0.0.0 \
    --theme.base dark \
    --theme.primaryColor "#6366f1" \
    --theme.backgroundColor "#0f1117" \
    --theme.secondaryBackgroundColor "#1a1d27" \
    --theme.textColor "#e2e8f0" \
    > "$LOG_DIR/ui.log" 2>&1 &
  UI_PID=$!
  echo "$UI_PID" >> "$PID_FILE"
  wait_for_port $UI_PORT "Streamlit"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}All services running:${RESET}"
$START_API && echo -e "  ${GREEN}●${RESET} FastAPI backend   →  ${BLUE}http://localhost:$API_PORT${RESET}       (docs: http://localhost:$API_PORT/docs)"
$START_MCP && echo -e "  ${GREEN}●${RESET} FastMCP server    →  ${BLUE}http://localhost:$MCP_PORT${RESET}"
$START_UI  && echo -e "  ${GREEN}●${RESET} Streamlit UI      →  ${BLUE}http://localhost:$UI_PORT${RESET}"
echo ""
echo -e "  Logs:  ${LOG_DIR}/"
echo -e "  Stop:  ${BOLD}Ctrl+C${RESET}  or  ${BOLD}./scripts/start_all.sh --stop${RESET}"
echo ""

# ── Tail logs from all services in the foreground ─────────────────────────────
# This keeps the script alive so Ctrl+C triggers the cleanup trap.
log "Tailing logs (Ctrl+C to stop all services)..."
echo ""

TAIL_PIDS=()
$START_API && { tail -f "$LOG_DIR/api.log" | sed "s/^/${CYAN}[API]${RESET} /" & TAIL_PIDS+=($!); }
$START_MCP && { tail -f "$LOG_DIR/mcp.log" | sed "s/^/${YELLOW}[MCP]${RESET} /" & TAIL_PIDS+=($!); }
$START_UI  && { tail -f "$LOG_DIR/ui.log"  | sed "s/^/${GREEN}[UI] ${RESET} /" & TAIL_PIDS+=($!); }

# Wait forever (until Ctrl+C fires the trap)
wait