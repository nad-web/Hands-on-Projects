#!/bin/bash
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 tools/serve.py
else
  echo "Python 3 is needed for local preview. The published GitHub Pages app needs no installation."
  read -r -p "Press Return to close."
fi
