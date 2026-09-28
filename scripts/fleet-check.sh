#!/usr/bin/env bash
# Check a running Voicebox deployment the way an application would: readiness,
# the voice catalog, one streamed /v1/audio/speech call and one
# /v1/audio/transcriptions call, repeated so every replica behind a load
# balancer gets exercised.
#
#   scripts/fleet-check.sh https://voice.example.com <client-key> [--voice NAME] [--model MODEL]
#                          [--rounds N] [--no-stt]
#
# Replicas that drifted (a different seed, a missing model) show up as a
# changing catalog fingerprint or a failed round.  Exit status is non-zero
# on any failure; transcription is skipped with a note when Whisper is not
# on the server (HTTP 409).
set -euo pipefail

BASE="${1:?usage: fleet-check.sh <base-url> <client-key> [--voice NAME] [--model MODEL] [--rounds N] [--no-stt]}"
KEY="${2:?usage: fleet-check.sh <base-url> <client-key> [--voice NAME] [--model MODEL] [--rounds N] [--no-stt]}"
shift 2
VOICE=""
MODEL="tts-1"
ROUNDS=3
STT=1
while [ $# -gt 0 ]; do
    case "$1" in
        --voice) VOICE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --no-stt) STT=0; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
BASE="${BASE%/}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
AUTH=(-H "Authorization: Bearer $KEY")
failures=0

fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }

echo "readiness"
code=$(curl -s -o "$WORK/ready.json" -w '%{http_code}' "$BASE/health/ready" || true)
if [ "$code" != "200" ]; then
    fail "GET /health/ready answered $code: $(cat "$WORK/ready.json" 2>/dev/null)"
fi

echo "identity"
code=$(curl -s -o "$WORK/whoami.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/auth/whoami" || true)
[ "$code" = "200" ] || fail "GET /auth/whoami answered $code (is the key valid?)"

echo "request id echo"
rid="fleetcheck$(date +%s)"
echoed=$(curl -s -o /dev/null -D - -H "X-Request-Id: $rid" "$BASE/health/ready" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-request-id"{print $2}')
[ "$echoed" = "$rid" ] || fail "X-Request-Id not echoed (got '$echoed')"

fingerprint() { # prints "count sha" of the visible catalog
    python3 - "$1" <<'PY'
import hashlib, json, sys
data = json.load(open(sys.argv[1]))["data"]
ids = sorted(f"{v['kind']}:{v['id']}:{v['name']}" for v in data)
print(len(ids), hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16])
PY
}

first_fp=""
for round in $(seq 1 "$ROUNDS"); do
    echo "round $round/$ROUNDS"
    code=$(curl -s -o "$WORK/voices.json" -w '%{http_code}' "${AUTH[@]}" "$BASE/v1/voices" || true)
    if [ "$code" != "200" ]; then
        fail "GET /v1/voices answered $code"
        continue
    fi
    fp=$(fingerprint "$WORK/voices.json")
    if [ -z "$first_fp" ]; then
        first_fp="$fp"
        echo "  catalog: $fp (voices, fingerprint)"
        if [ -z "$VOICE" ]; then
            VOICE=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["data"]; p=[v for v in d if v["kind"]=="profile"]; print((p or d)[0]["name"] if p else "af_heart")' "$WORK/voices.json")
            echo "  using voice: $VOICE"
        fi
    elif [ "$fp" != "$first_fp" ]; then
        fail "catalog differs between replicas: $first_fp vs $fp"
    fi

    body=$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "voice": sys.argv[2], "input": "Fleet check, round " + sys.argv[3] + ". The server is up and speaking.", "response_format": "wav"}))' "$MODEL" "$VOICE" "$round")
    code=$(curl -s -o "$WORK/speech.wav" -D "$WORK/headers.txt" -w '%{http_code}' "${AUTH[@]}" -H 'Content-Type: application/json' -d "$body" "$BASE/v1/audio/speech" || true)
    if [ "$code" != "200" ]; then
        fail "POST /v1/audio/speech answered $code: $(head -c 300 "$WORK/speech.wav")"
        continue
    fi
    size=$(wc -c < "$WORK/speech.wav" | tr -d ' ')
    engine=$(tr -d '\r' < "$WORK/headers.txt" | awk -F': ' 'tolower($1)=="x-voicebox-engine"{print $2}')
    [ "$size" -gt 10000 ] || fail "speech returned only $size bytes"
    echo "  speech: $size bytes via $engine"

    if [ "$STT" = "1" ]; then
        code=$(curl -s -o "$WORK/stt.json" -w '%{http_code}' "${AUTH[@]}" -F "file=@$WORK/speech.wav" -F "model=whisper-1" "$BASE/v1/audio/transcriptions" || true)
        case "$code" in
            200) echo "  transcription: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["text"][:60])' "$WORK/stt.json")" ;;
            409) echo "  transcription skipped: Whisper is not downloaded on the server (409)"; STT=0 ;;
            *) fail "POST /v1/audio/transcriptions answered $code: $(head -c 300 "$WORK/stt.json")" ;;
        esac
    fi
done

if [ "$failures" -gt 0 ]; then
    echo "fleet check failed ($failures problem(s))" >&2
    exit 1
fi
echo "fleet check passed: $ROUNDS round(s) against $BASE"
