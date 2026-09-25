import { invoke } from '@tauri-apps/api/core';
import { emit, listen, type UnlistenFn } from '@tauri-apps/api/event';
import { useEffect, useRef, useState } from 'react';
import { CapturePill } from '@/components/CapturePill/CapturePill';
import { apiClient } from '@/lib/api/client';
import type { FocusSnapshot } from '@/lib/api/types';
import { markCredentialsReady } from '@/lib/credentials';
import { useCaptureRecordingSession } from '@/lib/hooks/useCaptureRecordingSession';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';

/**
 * Floating dictate surface shown in a separate transparent Tauri window.
 * Mounted when the URL contains ``?view=dictate``. The main window bypasses
 * this branch and renders the full app shell.
 *
 * The pill surfaces for two independent cycles:
 *   1. User dictation — driven by ``dictate:start`` / ``dictate:stop``
 *      from the Rust hotkey monitor.
 *   2. Agent speech — driven by ``dictate:speak-start`` / ``dictate:speak-end``
 *      from the Rust ``speak_monitor`` (which owns the backend SSE stream).
 *      On speak-start we subscribe to this single generation's status SSE,
 *      then play ``/audio/{id}`` via a plain ``HTMLAudioElement`` when it
 *      lands. When the audio element's ``ended`` fires, we emit
 *      ``dictate:hide`` so Rust tucks the window away.
 */
export function DictateWindow() {
  // Force the host document chrome to be transparent so the Tauri window
  // takes on the pill's own shape.
  useEffect(() => {
    const prevHtml = document.documentElement.style.background;
    const prevBody = document.body.style.background;
    document.documentElement.style.background = 'transparent';
    document.body.style.background = 'transparent';
    return () => {
      document.documentElement.style.background = prevHtml;
      document.body.style.background = prevBody;
    };
  }, []);

  // This webview has its own JS context (zustand persist does not sync
  // across windows, and in Tauri the key is memory-only), so fetch the
  // server URL and key from Rust before anything here talks to the API.
  const platform = usePlatform();
  useEffect(() => {
    platform.lifecycle
      .getCredentials()
      .then((creds) => {
        const store = useServerStore.getState();
        if (creds.url) store.setServerUrl(creds.url);
        if (creds.apiKey) store.setApiKey(creds.apiKey);
      })
      .catch((err) => {
        console.warn('[dictate] server credentials unavailable:', err);
      })
      .finally(() => markCredentialsReady());
  }, [platform.lifecycle]);

  // Snapshot of the focused UI element at chord-start, shipped over from
  // Rust on the ``dictate:start`` payload. Held in a ref so it survives
  // the 1–2 s transcribe + refine window — the paste only fires once the
  // final text comes back.
  const focusRef = useRef<FocusSnapshot | null>(null);

  const session = useCaptureRecordingSession({
    onFinalText: async (text, _capture, allowAutoPaste) => {
      const focus = focusRef.current;
      // Consume-once: a second chord before this fires would overwrite
      // focusRef, but nulling it here guards against the late-arriving
      // refine-result firing a paste after the user has moved on.
      focusRef.current = null;
      if (!allowAutoPaste) return;
      if (!focus || !text.trim()) return;
      try {
        await invoke('paste_final_text', { text, focus });
      } catch (err) {
        // Surface accessibility failures to the main window so it can prompt
        // the user to grant permission. Other errors stay swallowed —
        // the transcription still landed in the captures list.
        const msg = err instanceof Error ? err.message : String(err);
        if (/accessibility/i.test(msg)) {
          emit('system:accessibility-missing').catch(() => {});
        }
        console.warn('[dictate] paste_final_text failed:', err);
      }
    },
  });

  // Route the chord events emitted from Rust into the session hook. Using a
  // ref so the `listen` effect only subscribes once — rebinding every render
  // would thrash the Tauri event bridge.
  const sessionRef = useRef(session);
  sessionRef.current = session;

  useEffect(() => {
    const unlistens: Promise<UnlistenFn>[] = [];
    unlistens.push(
      listen<{ focus: FocusSnapshot | null }>('dictate:start', (event) => {
        focusRef.current = event.payload?.focus ?? null;
        sessionRef.current.startRecording();
      }),
    );
    unlistens.push(
      listen('dictate:stop', () => {
        if (sessionRef.current.isRecording) sessionRef.current.stopRecording();
      }),
    );
    return () => {
      for (const p of unlistens) p.then((fn) => fn()).catch(() => {});
    };
  }, []);

  // --- Agent-speak cycle ---------------------------------------------------

  const [speaking, setSpeaking] = useState<{
    generationId: string;
    // Null while the backend is still generating audio; set to the
    // wall-clock timestamp when audio playback actually begins, so the
    // pill's elapsed counter only ticks while sound is coming out.
    startedAt: number | null;
  } | null>(null);
  const [speakElapsed, setSpeakElapsed] = useState(0);

  // Refs so handlers inside long-lived `listen()` callbacks can read the
  // latest state without re-subscribing on every render.
  const speakingRef = useRef<typeof speaking>(null);
  speakingRef.current = speaking;
  const statusSourceRef = useRef<EventSource | null>(null);
  const statusTimeoutRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const clearStatusTimeout = () => {
    if (statusTimeoutRef.current !== null) {
      window.clearTimeout(statusTimeoutRef.current);
      statusTimeoutRef.current = null;
    }
  };

  const dismissSpeak = (id?: string) => {
    // Guard against a late dismiss targeting a stale cycle (a new speak
    // already started by the time audio.ended from the previous one fired).
    if (id && speakingRef.current && speakingRef.current.generationId !== id) return;
    statusSourceRef.current?.close();
    statusSourceRef.current = null;
    clearStatusTimeout();
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    setSpeaking(null);
  };

  const startSpeakPlayback = async (generationId: string) => {
    // The audio element cannot send headers; the URL carries a media token
    // and this hidden window has no refresh timer, so fetch one on demand.
    await apiClient.ensureMediaToken();
    if (speakingRef.current?.generationId !== generationId) return;
    const audio = new Audio(apiClient.getAudioUrl(generationId));
    audio.onended = () => dismissSpeak(generationId);
    audio.onerror = () => dismissSpeak(generationId);
    // The pill window stays hidden through the ~1 s generation wait so the
    // user doesn't see a silent pill. We surface it the moment audio
    // actually starts playing, and that's also when the elapsed counter
    // arms.
    audio.onplaying = () => {
      emit('dictate:show').catch(() => {});
      setSpeaking((prev) =>
        prev && prev.generationId === generationId
          ? { ...prev, startedAt: Date.now() }
          : prev,
      );
      setSpeakElapsed(0);
    };
    audioRef.current = audio;
    audio.play().catch((err) => {
      console.warn('[dictate] audio.play failed:', err);
      dismissSpeak(generationId);
    });
  };

  // Monotonic id per speak cycle so async work from a superseded cycle
  // (token refresh, stream re-open) cannot touch the current one.
  const speakCycleRef = useRef(0);

  const openStatusSource = (id: string, cycle: number, retried: boolean) => {
    const source = new EventSource(apiClient.getGenerationStatusUrl(id));
    statusSourceRef.current = source;
    source.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data) as { status?: string };
        if (data.status === 'completed') {
          clearStatusTimeout();
          source.close();
          if (statusSourceRef.current === source) statusSourceRef.current = null;
          void startSpeakPlayback(id);
        } else if (data.status === 'failed' || data.status === 'not_found') {
          clearStatusTimeout();
          source.close();
          dismissSpeak(id);
        }
      } catch {
        // heartbeats / junk — ignore.
      }
    };
    source.onerror = () => {
      // Transient drops reconnect on their own and the 60 s timeout is the
      // backstop. CLOSED means the server refused the stream (a stale media
      // token after a restart): refresh the token once and reopen.
      if (source.readyState !== EventSource.CLOSED) return;
      if (retried || speakCycleRef.current !== cycle) return;
      if (statusSourceRef.current === source) statusSourceRef.current = null;
      useServerStore.getState().setMediaToken(null);
      void apiClient.ensureMediaToken().then(() => {
        if (speakCycleRef.current === cycle) openStatusSource(id, cycle, true);
      });
    };
  };

  useEffect(() => {
    const unlistens: Promise<UnlistenFn>[] = [];

    // Rust emits the SSE payload as a JSON *string* (not a parsed object);
    // the payload shape for speak-start is
    // {generation_id, profile_name, source, client_id}.
    unlistens.push(
      listen<string>('dictate:speak-start', (event) => {
        let parsed: { generation_id?: string } = {};
        try {
          parsed = typeof event.payload === 'string' ? JSON.parse(event.payload) : {};
        } catch {
          return;
        }
        const id = parsed.generation_id;
        if (!id) return;

        // Tear down any previous cycle — last speak wins.
        dismissSpeak();

        setSpeaking({ generationId: id, startedAt: null });
        setSpeakElapsed(0);

        // Hard cap on how long the pill can sit in the 'speaking' state
        // without ever hearing back from the backend. Covers the case where
        // the gen row is deleted mid-flight (SSE 404s and EventSource silently
        // retries) or the backend goes away while a request is in flight.
        // Clears as soon as a real status event lands.
        clearStatusTimeout();
        statusTimeoutRef.current = window.setTimeout(() => {
          statusTimeoutRef.current = null;
          if (speakingRef.current?.generationId === id && !audioRef.current) {
            dismissSpeak(id);
          }
        }, 60_000);
        // Subscribe to this one generation's status. When it completes, the
        // `/audio/{id}` endpoint will serve the WAV we need to play. The
        // stream URL carries a media token, so make sure one exists first.
        const cycle = ++speakCycleRef.current;
        void apiClient.ensureMediaToken().then(() => {
          if (speakCycleRef.current === cycle) openStatusSource(id, cycle, false);
        });
      }),
    );

    // Speak-end from the backend is advisory: the authoritative dismiss is
    // `audio.ended`. But if generation failed or nothing ever triggered
    // playback, a short grace window followed by forced dismiss avoids a
    // stuck-visible pill.
    unlistens.push(
      listen<string>('dictate:speak-end', (event) => {
        let parsed: { generation_id?: string; status?: string } = {};
        try {
          parsed = typeof event.payload === 'string' ? JSON.parse(event.payload) : {};
        } catch {
          return;
        }
        if (parsed.status && parsed.status !== 'completed') {
          // Failed / cancelled — dismiss immediately.
          if (parsed.generation_id) dismissSpeak(parsed.generation_id);
          return;
        }
        // Completed: if audio never started (shouldn't happen, but guard),
        // auto-dismiss after 15 s so the pill never stays forever.
        const id = parsed.generation_id;
        window.setTimeout(() => {
          if (speakingRef.current?.generationId === id && !audioRef.current) {
            dismissSpeak(id);
          }
        }, 15_000);
      }),
    );

    return () => {
      for (const p of unlistens) p.then((fn) => fn()).catch(() => {});
      dismissSpeak();
    };
  }, []);

  // Advance the pill's elapsed-time label while audio is playing. Paused
  // during the pre-playback generation window (startedAt is null) so the
  // counter stays at 0:00 until sound actually starts.
  useEffect(() => {
    if (!speaking?.startedAt) return;
    const anchor = speaking.startedAt;
    const iv = window.setInterval(() => {
      setSpeakElapsed(Date.now() - anchor);
    }, 250);
    return () => window.clearInterval(iv);
  }, [speaking?.generationId, speaking?.startedAt]);

  // --- Effective pill state -----------------------------------------------

  const isSpeaking = Boolean(speaking);
  const effectiveState = isSpeaking ? 'speaking' : session.pillState;
  const effectiveElapsed = isSpeaking ? speakElapsed : session.pillElapsedMs;

  // When the pill cycle ends (no capture AND no speak), tell Rust to tuck
  // the window away. Rust owns the hide + park-off-screen + click-through
  // combo because calling hide() directly from JS has been unreliable for
  // transparent always-on-top windows on macOS.
  useEffect(() => {
    if (effectiveState === 'hidden') {
      emit('dictate:hide').catch(() => {});
    }
  }, [effectiveState]);

  return (
    <div
      className="h-screen w-screen flex items-center justify-center px-3"
      style={{ background: 'transparent' }}
    >
      {effectiveState !== 'hidden' ? (
        <CapturePill
          state={effectiveState}
          elapsedMs={effectiveElapsed}
          errorMessage={session.errorMessage}
          onDismiss={session.dismissError}
          onStop={session.isRecording ? session.stopRecording : undefined}
        />
      ) : null}
    </div>
  );
}
