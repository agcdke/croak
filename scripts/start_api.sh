#!/usr/bin/env bash
# scripts/start_api.sh — Start the FastAPI backend
set -e

echo "🚀 Starting FastAPI backend..."
cd "$(dirname "$0")/.."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
