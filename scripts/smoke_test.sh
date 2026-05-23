#!/usr/bin/env bash
# Smoke test: health check, plain chat, tool call, no-tool call
set -euo pipefail

BASE=${BASE_URL:-http://localhost:8080}
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }

echo "=== orthrus-serve smoke test ==="
echo "Base URL: ${BASE}"
echo

# 1. Health check
echo "[1/4] Health check"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health")
if [[ "$STATUS" == "200" ]]; then
    pass "GET /health returned 200"
else
    fail "GET /health returned ${STATUS}"
fi

# 2. /v1/models
echo
echo "[2/4] Models list"
MODELS=$(curl -s "${BASE}/v1/models")
MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id' 2>/dev/null || echo "")
MAX_MODEL_LEN=$(echo "$MODELS" | jq -r '.data[0].max_model_len' 2>/dev/null || echo "")
if [[ "$MODEL_ID" == "orthrus-qwen3-8b" ]]; then
    pass "GET /v1/models returned model id orthrus-qwen3-8b"
else
    fail "GET /v1/models: unexpected model id '${MODEL_ID}'"
fi
if [[ "$MAX_MODEL_LEN" == "40960" ]]; then
    pass "max_model_len = 40960"
else
    fail "max_model_len unexpected: '${MAX_MODEL_LEN}'"
fi

# 3. Plain chat (no tools)
echo
echo "[3/4] Plain chat completion"
PLAIN_RESP=$(curl -s -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "orthrus-qwen3-8b",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 64
    }')
PLAIN_FINISH=$(echo "$PLAIN_RESP" | jq -r '.choices[0].finish_reason' 2>/dev/null || echo "")
PLAIN_CONTENT=$(echo "$PLAIN_RESP" | jq -r '.choices[0].message.content' 2>/dev/null || echo "")
if [[ "$PLAIN_FINISH" == "stop" ]]; then
    pass "finish_reason=stop"
else
    fail "finish_reason=${PLAIN_FINISH} (expected stop)"
fi
if [[ -n "$PLAIN_CONTENT" ]]; then
    pass "non-empty content: ${PLAIN_CONTENT:0:80}"
else
    fail "empty content"
fi

# 4. Tool call: question that should trigger the tool
echo
echo "[4/4] Tool call completion"
TOOL_RESP=$(curl -s -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "orthrus-qwen3-8b",
        "messages": [{"role": "user", "content": "What is the current weather in Oxford?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 256
    }')
TOOL_FINISH=$(echo "$TOOL_RESP" | jq -r '.choices[0].finish_reason' 2>/dev/null || echo "")
TOOL_CALLS=$(echo "$TOOL_RESP" | jq -r '.choices[0].message.tool_calls | length' 2>/dev/null || echo "0")
TOOL_ARG_TYPE=$(echo "$TOOL_RESP" | jq -r '.choices[0].message.tool_calls[0].function.arguments | type' 2>/dev/null || echo "")
if [[ "$TOOL_FINISH" == "tool_calls" ]]; then
    pass "finish_reason=tool_calls"
else
    fail "finish_reason=${TOOL_FINISH} (expected tool_calls)"
fi
if [[ "$TOOL_CALLS" -ge 1 ]]; then
    pass "tool_calls array has ${TOOL_CALLS} entry/entries"
else
    fail "tool_calls array is empty or missing"
fi
if [[ "$TOOL_ARG_TYPE" == "string" ]]; then
    pass "tool_calls[0].function.arguments is a JSON string"
else
    fail "tool_calls[0].function.arguments type=${TOOL_ARG_TYPE} (expected string)"
fi

echo
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[[ "$FAIL" -eq 0 ]]
