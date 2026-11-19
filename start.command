#!/bin/zsh
set -e
cd "$(dirname "$0")"
PORT=4175
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  open "http://127.0.0.1:$PORT/"
  exit 0
fi
python3 -m http.server $PORT --bind 127.0.0.1 > .viewer.log 2>&1 &
sleep 0.8
open "http://127.0.0.1:$PORT/"
