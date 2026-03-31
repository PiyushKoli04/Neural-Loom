#!/usr/bin/env bash
# Neural-Loom — one-shot setup script
# Run: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════╗"
echo "║   Neural-Loom Setup Script   ║"
echo "╚══════════════════════════════╝"
echo ""

# ── Python version check ──────────────────────────────────────────────────
PY=$(python3 --version 2>&1 | awk '{print $2}')
echo "▸ Python detected: $PY"

# mediapipe requires ≤ 3.12; prefer python3.12 if available
if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
    echo "▸ Using python3.12 for venv (mediapipe compatibility)"
else
    PYTHON=python3
    echo "▸ Using system python3"
fi

# ── Virtual environment ───────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "▸ Creating virtual environment…"
    $PYTHON -m venv venv
else
    echo "▸ Virtual environment already exists — skipping"
fi

source venv/bin/activate
echo "▸ Activated venv"

# ── Dependencies ──────────────────────────────────────────────────────────
echo "▸ Installing dependencies (this may take a minute)…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "▸ Dependencies installed"

# ── .env setup ────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  .env created from template."
    echo "    Please edit .env and add your API keys before running the app:"
    echo ""
    echo "    GEMINI_API_KEY  → https://aistudio.google.com/app/apikey"
    echo "    GROQ_API_KEY    → https://console.groq.com/keys"
    echo "    MONGO_URI       → mongodb://localhost:27017/neuralloom"
    echo "    SECRET_KEY      → any long random string"
    echo ""
else
    echo "▸ .env already exists — skipping"
fi

# ── MongoDB check ─────────────────────────────────────────────────────────
echo "▸ Checking MongoDB…"
if command -v mongod &>/dev/null; then
    echo "▸ mongod found. Make sure it is running before starting the app."
else
    echo "⚠️  mongod not found in PATH."
    echo "   Install MongoDB: https://www.mongodb.com/docs/manual/installation/"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Setup complete!                         ║"
echo "║                                          ║"
echo "║  1. Edit .env with your API keys         ║"
echo "║  2. Start MongoDB                        ║"
echo "║  3. Run:  source venv/bin/activate       ║"
echo "║           python app.py                  ║"
echo "║  4. Open: http://localhost:5000          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
