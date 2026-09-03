#!/usr/bin/env bash

set -e

API="http://localhost:8000"
PASS=0
FAIL=0

check() {
  local desc="$1"
  local expected="$2"
  local actual="$3"
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS+1))
    echo "  PASS: $desc"
  else
    FAIL=$((FAIL+1))
    echo "  FAIL: $desc (expected: $expected, got: $actual)"
  fi
}

echo "== Input validation =="
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/new-game" -H 'Content-Type: application/json' -d '{"difficulty":99}')
check "rejects invalid difficulty" "422" "$CODE"

CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/new-game" -H 'Content-Type: application/json' -d '{"player_color":"purple"}')
check "rejects invalid color" "422" "$CODE"

echo ""
echo "== New game =="
STATE=$(curl -s -X POST "$API/api/new-game" -H 'Content-Type: application/json' -d '{"difficulty":1,"player_color":"white"}')
TURN=$(echo "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin)['turn'])")
check "starts with white to move" "white" "$TURN"

echo ""
echo "== Turn enforcement =="
curl -s -X POST "$API/api/move" -H 'Content-Type: application/json' -d '{"move":"e2e4"}' > /dev/null
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/move" -H 'Content-Type: application/json' -d '{"move":"d2d4"}')
check "rejects move when not player's turn" "409" "$CODE"

echo ""
echo "== Legal/illegal moves =="
curl -s -X POST "$API/api/new-game" -H 'Content-Type: application/json' -d '{"difficulty":1,"player_color":"white"}' > /dev/null
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/move" -H 'Content-Type: application/json' -d '{"move":"e2e5"}')
check "rejects illegal move" "422" "$CODE"

STATE=$(curl -s -X POST "$API/api/move" -H 'Content-Type: application/json' -d '{"move":"e2e4"}')
check "accepts legal move" "black" "$(echo "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin)['turn'])")"

echo ""
echo "== Engine move =="
STATE=$(curl -s -X POST "$API/api/engine-move")
INFO=$(echo "$STATE" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['engine_info']['depth'])")
check "engine reports depth" "2" "$INFO"
check "engine returns to player's turn" "white" "$(echo "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin)['turn'])")"

echo ""
echo "== Undo =="
STATE=$(curl -s -X POST "$API/api/undo")
COUNT=$(echo "$STATE" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['move_stack']))")
check "undo removes player+engine moves" "0" "$COUNT"

echo ""
echo "== Eval =="
SCORE=$(curl -s "$API/api/eval" | python3 -c "import sys,json;print(json.load(sys.stdin)['score'])")
case "$SCORE" in
  ''|*[!0-9-]*) check "eval returns integer" "numeric" "non-numeric" ;;
  *) PASS=$((PASS+1)); echo "  PASS: eval returns integer ($SCORE)" ;;
esac

echo ""
echo "== Unknown game id =="
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API/api/state?game_id=nope")
check "rejects unknown game" "404" "$CODE"

echo ""
echo "=========================================="
echo "  Result: $PASS passed, $FAIL failed"
echo "=========================================="
[ "$FAIL" -eq 0 ]
