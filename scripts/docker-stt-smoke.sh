#!/usr/bin/env bash
# Offline optional-runtime smoke. Uses only its own temporary containers; no host ports.
# scripts/docker-stt-smoke.sh <image> [none|faster-whisper|nemotron|all]
set -euo pipefail
IMAGE="${1:?usage: docker-stt-smoke.sh <image> [profile]}"
PROFILE="${2:-all}"
case "$PROFILE" in none|faster-whisper|nemotron|all) ;; *) echo 'invalid STT profile' >&2; exit 1 ;; esac
NAME="voicebox-stt-smoke-$$"
cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then docker logs "$NAME" 2>&1 | tail -40 || true; fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

expect_preflight_failure() {
    expected="$1"
    shift
    if output=$(docker run --rm --network none -e VOICEBOX_STT_RUNTIMES="$PROFILE" "$@" \
        "$IMAGE" python -m backend.stt_preflight 2>&1); then
        echo 'invalid runtime configuration unexpectedly passed preflight' >&2
        return 1
    fi
    case "$output" in
        *"$expected"*) ;;
        *) echo "$output" >&2; echo "expected preflight error: $expected" >&2; return 1 ;;
    esac
}

# Exercise actual startup failures offline, including whitespace preload syntax
# and accidentally pointing the Transformers 5 worker at the main interpreter.
case "$PROFILE" in
  none)
    expect_preflight_failure PackageNotFoundError -e 'VOICEBOX_PRELOAD_MODELS=kokoro faster-whisper-base' ;;
  nemotron|all)
    expect_preflight_failure 'separate Nemotron runtime interpreter' -e VOICEBOX_NEMOTRON_PYTHON=/missing/python
    expect_preflight_failure 'transformers==5.13.0 is required' -e VOICEBOX_NEMOTRON_PYTHON=/opt/venv/bin/python ;;
esac

# No network access: these checks must not fetch a model or a dependency.
docker run --rm --network none -e VOICEBOX_STT_RUNTIMES="$PROFILE" "$IMAGE" \
    python -m backend.stt_preflight
case "$PROFILE" in
  faster-whisper|all)
    docker run --rm --network none -e VOICEBOX_STT_RUNTIMES="$PROFILE" "$IMAGE" \
      python /app/backend/tests/container_stt_smoke.py faster-whisper ;;
esac
case "$PROFILE" in
  nemotron|all)
    docker run --rm --network none -e VOICEBOX_STT_RUNTIMES="$PROFILE" "$IMAGE" \
      /opt/stt/nemotron/bin/python /app/backend/tests/container_stt_smoke.py nemotron ;;
esac

docker run -d --network none --name "$NAME" \
  -e VOICEBOX_STT_RUNTIMES="$PROFILE" -e HF_HUB_OFFLINE=1 -e VOICEBOX_PRELOAD_MODELS= \
  -e VOICEBOX_STT_EXPECTED_PROFILE="$PROFILE" "$IMAGE" >/dev/null
ready=0
for _ in $(seq 1 90); do
    if docker exec "$NAME" python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:17493/health/ready", timeout=2)' >/dev/null 2>&1; then
        ready=1
        break
    fi
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || break
    sleep 1
done
[ "$ready" = 1 ] || { echo 'STT image failed offline readiness' >&2; exit 1; }
docker exec -i "$NAME" python - <<'PY'
import json, os, urllib.request
from pathlib import Path
key = Path('/app/data/api_key').read_text().strip()
request = urllib.request.Request('http://127.0.0.1:17493/v1/models', headers={'Authorization': 'Bearer ' + key})
with urllib.request.urlopen(request, timeout=10) as response:
    models = {row['id']: row for row in json.load(response)['data']}
profile = os.environ['VOICEBOX_STT_EXPECTED_PROFILE']
for runtime, model in [('faster-whisper', 'faster-whisper-base'), ('nemotron', 'nemotron-streaming-en-0.6b')]:
    expected = profile in (runtime, 'all')
    if models[model]['runtime_available'] != expected:
        raise SystemExit(f'{runtime}: registry availability does not match image profile')
print('STT image: offline readiness and model capability discovery passed')
PY
docker stop -t 45 "$NAME" >/dev/null
[ "$(docker inspect -f '{{.State.ExitCode}}' "$NAME")" = 0 ] || { echo 'server did not stop cleanly' >&2; exit 1; }
echo "optional STT smoke passed: $PROFILE"
