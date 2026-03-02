#!/usr/bin/env bash
# scripts/start_mcp.sh — Start the FastMCP server
set -e

echo "🔌 Starting FastMCP server..."
cd "$(dirname "$0")/.."
python -m src.mcp.server
