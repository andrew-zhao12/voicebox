# Voicebox MCP server

Local **Model Context Protocol** server — lets any MCP-aware agent
(Claude Code, Cursor, Windsurf, VS Code MCP extensions, etc.) speak text
in your cloned voices, transcribe audio, and browse captures.

The server runs inside the same `uvicorn` process as the rest of Voicebox
and is mounted at `/mcp` (Streamable HTTP transport).

## Authentication

Every request to `/mcp` needs a Voicebox API key, like the rest of the
API. HTTP clients send it as `Authorization: Bearer <key>`; the stdio
shim reads it from the environment or, on the same machine, from the
desktop app's key file. Where to get a key:

- Desktop app: **Settings → MCP → Copy API key** (an admin key).
- Dev checkout: `just api-key` prints `data/api_key`.
- A dedicated key for one agent: `python -m backend.keys create --id
  claude-code --role client` from the repo root, or `POST /auth/keys`
  with an admin key.

A `client` key can call `voicebox.speak`, `voicebox.transcribe` (base64
audio) and `voicebox.list_profiles`; `voicebox.list_captures` and
`voicebox.transcribe` with `audio_path` need an admin key. Roles, rate
limits and key management are documented in
`docs/content/docs/overview/api-keys.mdx`.

## Install into your agent

Preferred — direct HTTP:

```json
{
  "mcpServers": {
    "voicebox": {
      "url": "http://127.0.0.1:17493/mcp",
      "headers": {
        "Authorization": "Bearer <key>",
        "X-Voicebox-Client-Id": "claude-code"
      }
    }
  }
}
```

Claude Code expands `${VAR}` in `.mcp.json` headers, so the repo's
`.mcp.json` uses `"Authorization": "Bearer ${VOICEBOX_API_KEY}"` and
picks the key up from the shell you start `claude` in
(`export VOICEBOX_API_KEY="$(just api-key)"`).

Fallback — stdio shim (when the client doesn't speak HTTP MCP). The
`voicebox-mcp` binary ships inside the Voicebox.app bundle and finds the
desktop app's key by itself when it runs on the same machine:

```json
{
  "mcpServers": {
    "voicebox": {
      "command": "/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp",
      "env": { "VOICEBOX_CLIENT_ID": "claude-code" }
    }
  }
}
```

Shim environment variables: `VOICEBOX_API_KEY` (the key) or
`VOICEBOX_API_KEY_FILE` (a file containing it); when neither is set and
the host is loopback, the shim reads the desktop app's `api_key` file.
`VOICEBOX_HOST` / `VOICEBOX_PORT` point it at another server (then the
key must be given explicitly), and `VOICEBOX_CLIENT_ID` is forwarded as
`X-Voicebox-Client-Id`. Without a usable key the shim exits with a
message instead of connecting.

Claude Code one-liner:

```
claude mcp add --transport http voicebox http://127.0.0.1:17493/mcp \
  --header "Authorization: Bearer <key>" \
  --header "X-Voicebox-Client-Id: claude-code"
```

## Tools

| Name | Purpose | Key |
|---|---|---|
| `voicebox.speak`          | Speak text in a voice profile. Returns a generation id you can poll. | client or admin |
| `voicebox.transcribe`     | Whisper transcription of a base64 blob or an absolute local path. | client (base64) / admin (`audio_path`) |
| `voicebox.list_captures`  | Recent captures (dictation / recording / file) with transcripts. | admin |
| `voicebox.list_profiles`  | Available voice profiles (cloned + preset). | client or admin |

`speak` and `transcribe` count against the key's inference rate limit
(30/min for client keys, 120/min for admin keys by default), and `speak`
respects the generation queue caps. Rejections come back as tool errors
that name the reason.

All tools resolve voice profiles in this precedence:

1. Explicit `profile` arg (name or id — case-insensitive)
2. Per-client binding keyed by `X-Voicebox-Client-Id`
3. `capture_settings.default_playback_voice_id` (global default)

Bindings are managed via `GET|PUT /mcp/bindings` (admin key) or in the
app under Settings → MCP.

## Debug with MCP Inspector

```
npx @modelcontextprotocol/inspector http://127.0.0.1:17493/mcp
```

Point it at the URL, add `Authorization: Bearer <key>` in the Inspector's
authentication settings, hit "List tools," call `voicebox.list_profiles`
first to confirm wiring, then `voicebox.speak` for end-to-end.

## Non-MCP REST surface

`POST /speak` is a thin wrapper on the same code path for callers that
don't speak MCP (shell scripts, ACP, A2A):

```
curl -X POST http://127.0.0.1:17493/speak \
  -H "Authorization: Bearer $VOICEBOX_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'X-Voicebox-Client-Id: claude-code' \
  -d '{"text":"Build complete.","profile":"Morgan"}'
```

## Code layout

```
backend/mcp_server/
├── __init__.py      # re-export mount_into
├── server.py        # build_mcp_server() + mount_into(app)
├── tools.py         # @mcp.tool() implementations
├── context.py       # ClientIdMiddleware + current_client_id ContextVar
├── resolve.py       # profile resolution precedence
├── events.py        # pub/sub queue for /events/speak pill SSE
└── README.md        # you are here

backend/mcp_shim/    # stdio ↔ Streamable-HTTP proxy (see its README)
```

Authentication does not live here: the bearer-key middleware in
`backend/auth/` covers the `/mcp` mount like every other route, and
`tools.py` reads the caller's key and role from the request to enforce
the admin-only tools and the inference rate limit.

The package is **`mcp_server`**, not `mcp`, to avoid shadowing the
installed `mcp` PyPI package that FastMCP imports internally.
