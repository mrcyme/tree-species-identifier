#!/bin/bash
# Start script for Tree Species Identifier

echo "🌳 Tree Species Identifier"
echo "=========================="

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed."; exit 1; }
command -v Rscript >/dev/null 2>&1 || { echo "R is required but not installed."; exit 1; }
command -v pnpm >/dev/null 2>&1 || { echo "pnpm is required but not installed."; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start backend
echo ""
echo "Starting backend server..."
cd "$PROJECT_DIR"
source .venv/bin/activate 2>/dev/null || {
    echo "Creating Python virtual environment with uv..."
    uv venv .venv
    source .venv/bin/activate
    uv pip install -r backend/requirements.txt
}

cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started on http://localhost:8000 (PID: $BACKEND_PID)"

# Start frontend
echo ""
echo "Starting frontend..."
cd "$PROJECT_DIR/frontend"
pnpm install 2>/dev/null
pnpm dev &
FRONTEND_PID=$!
echo "Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"

echo ""
echo "=========================="
echo "🚀 Application is running!"
echo ""
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================="

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait









