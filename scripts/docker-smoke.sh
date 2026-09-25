#!/usr/bin/env bash
# Boot a Voicebox image, wait for the Kokoro preload, then prove a client key
# can stream speech and that a stop drains cleanly.  Used by the Docker CI
# workflow; runs against any built image:
#
#   scripts/docker-smoke.sh voicebox:ci
#
# SMOKE_PORT (default 17493) is the host port; the HuggingFace download of
# Kokoro (~350 MB) is the slow part, bounded by SMOKE_READY_TIMEOUT seconds.
set -euo pipefail

IMAGE="${1:?usage: docker-smoke.sh <image>}"
PORT="${SMOKE_PORT:-17493}"
READY_TIMEOUT="${SMOKE_READY_TIMEOUT:-900}"
KEY="vbx_smoke_$(date +%s)"
NAME="voicebox-smoke-$$"
BASE="http://127.0.0.1:${PORT}"
WORK="$(mktemp -d)"

cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "--- container log (last 60 lines) ---"
        docker logs "$NAME" 2>&1 | tail -60 || true
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    rm -rf "$WORK"
}
trap cleanup EXIT

wait_for() { # url expected-status timeout-seconds
    local url=$1 want=$2 timeout=$3 start code
    start=$(date +%s)
    while :; do
        code=$(curl -s -o /dev/null -w '%{http_code}' "$url" || true)
        [ "$code" = "$want" ] && return 0
        if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
            echo "timed out after ${timeout}s waiting for $url to answer $want (last: $code)" >&2
            return 1
        fi
        sleep 3
    done
}

expect_status() { # want actual what
    if [ "$2" != "$1" ]; then
        echo "expected HTTP $1 for $3, got $2" >&2
        return 1
    fi
}

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:17493" \
    -e VOICEBOX_API_KEY="$KEY" \
    -e VOICEBOX_PRELOAD_MODELS=kokoro \
    -e VOICEBOX_DISABLE_DOCS=1 \
    "$IMAGE" >/dev/null

echo "waiting for /health (liveness)"
wait_for "$BASE/health" 200 180
echo "waiting for /health/ready (Kokoro download and load)"
wait_for "$BASE/health/ready" 200 "$READY_TIMEOUT"

echo "checking the public surface"
curl -sf "$BASE/" | grep -qi "<html" || { echo "the SPA is not served at /" >&2; exit 1; }
expect_status 401 "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/profiles")" "anonymous GET /profiles"

echo "creating a client key and a Kokoro preset profile with the admin key"
client=$(curl -sf -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"id":"smoke","role":"client"}' "$BASE/auth/keys" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["key"])')
profile=$(curl -sf -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"name":"Smoke","language":"en","voice_type":"preset","preset_engine":"kokoro","preset_voice_id":"af_heart"}' \
    "$BASE/profiles" | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')

echo "streaming speech with the client key"
code=$(curl -s -o "$WORK/speech.wav" -D "$WORK/headers.txt" -w '%{http_code}' \
    -H "Authorization: Bearer $client" -H 'Content-Type: application/json' \
    -d "{\"profile_id\":\"$profile\",\"text\":\"Hello from the Voicebox smoke test. The server is up and speaking.\",\"engine\":\"kokoro\"}" \
    "$BASE/generate/stream")
expect_status 200 "$code" "POST /generate/stream"
grep -qi "x-voicebox-sample-rate: 24000" "$WORK/headers.txt" || { echo "missing sample-rate header" >&2; exit 1; }
size=$(wc -c < "$WORK/speech.wav" | tr -d ' ')
[ "$size" -gt 20000 ] || { echo "stream returned only $size bytes" >&2; exit 1; }
expect_status 403 "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $client" "$BASE/history")" "client GET /history"

echo "stopping (graceful drain)"
start=$(date +%s)
docker stop -t 45 "$NAME" >/dev/null
elapsed=$(( $(date +%s) - start ))
docker logs "$NAME" 2>&1 | grep -q "Draining" || { echo "no drain log line on stop" >&2; exit 1; }
[ "$elapsed" -lt 40 ] || { echo "stop took ${elapsed}s" >&2; exit 1; }

echo "smoke test passed: ${size} bytes of Kokoro audio, stopped in ${elapsed}s"
