#!/usr/bin/env bash
# Smoke test: health, /v1/models, plain chat, tool call, SSE w/ usage.
#
# Verifies the HTTP surface (OpenAI-compatible API behaves as expected).
# Does NOT verify quantisation is active / firing correctly -- for that, use
# the dedicated `python -m orthrus_serve.quantization --smoke --scheme <X>`
# CLI documented in quantization.md.
#
# By default MODEL_ID is auto-derived from /v1/models, so this script works
# against any precision (orthrus-qwen3-8b, -fp8, -nvfp4, -fp8-row, etc.)
# without configuration. Override MODEL_ID explicitly to assert a specific
# served model id.
set -uo pipefail

BASE=${BASE_URL:-http://localhost:8080}
if [[ -z "${MODEL_ID:-}" ]]; then
    MODEL_ID=$(curl -s "${BASE}/v1/models" | jq -r '.data[0].id' 2>/dev/null || echo "")
    if [[ -z "$MODEL_ID" || "$MODEL_ID" == "null" ]]; then
        MODEL_ID="orthrus-qwen3-8b"  # fallback for error messages if server unreachable
    fi
fi
PASS=0
FAIL=0

# `((x++))` returns the old value, which is 0 first time; with `set -e` that
# would exit the script. Use `+=1` so the arithmetic expression evaluates to 1
# (truthy) and the function's exit code stays 0.
pass() { echo "  PASS: $1"; ((PASS+=1)); }
fail() { echo "  FAIL: $1"; ((FAIL+=1)); }

echo "=== orthrus-serve smoke test ==="
echo "Base URL: ${BASE}  Model: ${MODEL_ID}"
echo

# 1. Health check
echo "[1/5] Health check"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health")
if [[ "$STATUS" == "200" ]]; then
    pass "GET /health returned 200"
else
    fail "GET /health returned ${STATUS}"
fi

# 2. /v1/models
echo
echo "[2/5] Models list"
MODELS=$(curl -s "${BASE}/v1/models")
SERVED_ID=$(echo "$MODELS" | jq -r '.data[0].id' 2>/dev/null || echo "")
MAX_MODEL_LEN=$(echo "$MODELS" | jq -r '.data[0].max_model_len' 2>/dev/null || echo "")
if [[ "$SERVED_ID" == "$MODEL_ID" ]]; then
    pass "GET /v1/models returned model id ${MODEL_ID}"
else
    fail "GET /v1/models: served '${SERVED_ID}' but expected '${MODEL_ID}'"
fi
if [[ "$MAX_MODEL_LEN" == "40960" ]]; then
    pass "max_model_len = 40960"
else
    fail "max_model_len unexpected: '${MAX_MODEL_LEN}'"
fi

# 3. Plain chat (no tools)
echo
echo "[3/5] Plain chat completion"
PLAIN_RESP=$(curl -s -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_ID}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
        \"max_tokens\": 64
    }")
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
echo "[4/5] Tool call completion"
TOOL_RESP=$(curl -s -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_ID}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is the current weather in Oxford?\"}],
        \"tools\": [{
            \"type\": \"function\",
            \"function\": {
                \"name\": \"get_weather\",
                \"description\": \"Get current weather for a city\",
                \"parameters\": {
                    \"type\": \"object\",
                    \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City name\"}},
                    \"required\": [\"location\"]
                }
            }
        }],
        \"tool_choice\": \"auto\",
        \"max_tokens\": 256
    }")
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

# 5. Streaming SSE: final chunk should carry usage.
echo
echo "[5/5] Streaming SSE with usage in final chunk"
SSE_BODY=$(curl -s -N -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_ID}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}],
        \"max_tokens\": 32,
        \"stream\": true
    }")
LAST_DATA=$(echo "$SSE_BODY" | grep '^data: ' | grep -v '^data: \[DONE\]$' | tail -1 | sed 's/^data: //')
FINISH=$(echo "$LAST_DATA" | jq -r '.choices[0].finish_reason' 2>/dev/null || echo "")
USAGE_TOTAL=$(echo "$LAST_DATA" | jq -r '.usage.total_tokens' 2>/dev/null || echo "")
DONE_PRESENT=$(echo "$SSE_BODY" | grep -c '^data: \[DONE\]$' || true)
if [[ "$FINISH" == "stop" || "$FINISH" == "tool_calls" ]]; then
    pass "final SSE chunk has finish_reason=${FINISH}"
else
    fail "final SSE chunk finish_reason='${FINISH}' (expected stop or tool_calls)"
fi
if [[ -n "$USAGE_TOTAL" && "$USAGE_TOTAL" != "null" && "$USAGE_TOTAL" -gt 0 ]] 2>/dev/null; then
    pass "final SSE chunk has usage.total_tokens=${USAGE_TOTAL}"
else
    fail "final SSE chunk missing usage.total_tokens (got '${USAGE_TOTAL}')"
fi
if [[ "$DONE_PRESENT" -ge 1 ]]; then
    pass "stream terminated with [DONE]"
else
    fail "stream did not emit [DONE] terminator"
fi

echo
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[[ "$FAIL" -eq 0 ]]
