import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { queryClient } from '@/lib/queryClient';
import { isTauriRuntime } from '@/lib/utils/isTauriRuntime';

/**
 * Where the app stands with the backend's bearer auth.
 *  - unknown: not verified yet (startup, or the server URL just changed)
 *  - ok: `/auth/whoami` accepted the key
 *  - unauthorized: 401 (missing/invalid key, or the server restarted with a new one)
 *  - forbidden: 403 (a client-role key; the app needs an admin key)
 *  - offline: the server could not be reached at all
 */
export type AuthStatus = 'unknown' | 'ok' | 'unauthorized' | 'forbidden' | 'offline';

export interface ServerIdentity {
  key_id: string;
  role: 'admin' | 'client';
}

/** Short-lived `?token=` credential for `<img>`, `<audio>` and EventSource loads. */
export interface MediaToken {
  token: string;
  /** Epoch milliseconds. */
  expiresAt: number;
}

interface ServerStore {
  serverUrl: string;
  setServerUrl: (url: string) => void;

  /**
   * Bearer key sent on every request. Persisted in the browser shell only; in
   * Tauri the Rust side hands it over on every launch and it stays in memory.
   */
  apiKey: string | null;
  setApiKey: (key: string | null) => void;

  authStatus: AuthStatus;
  setAuthStatus: (status: AuthStatus) => void;

  identity: ServerIdentity | null;
  setIdentity: (identity: ServerIdentity | null) => void;

  /** Never persisted: a server restart invalidates every token. */
  mediaToken: MediaToken | null;
  setMediaToken: (token: MediaToken | null) => void;

  mode: 'local' | 'remote';
  setMode: (mode: 'local' | 'remote') => void;

  keepServerRunningOnClose: boolean;
  setKeepServerRunningOnClose: (keepRunning: boolean) => void;

  customModelsDir: string | null;
  setCustomModelsDir: (dir: string | null) => void;
}

/**
 * Invalidate all React Query caches so stale data from the previous
 * server is not shown. Called when the server URL changes.
 */
function invalidateAllServerData() {
  queryClient.invalidateQueries();
}

export function getDefaultServerUrl(): string {
  const fallback = 'http://127.0.0.1:17493';

  if (!import.meta.env.PROD || typeof window === 'undefined') {
    return fallback;
  }

  const { protocol, origin, hostname } = window.location;
  if (
    (protocol === 'http:' || protocol === 'https:') &&
    origin &&
    hostname !== 'tauri.localhost'
  ) {
    return origin;
  }

  return fallback;
}

export function isLoopbackVoiceboxServerUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return (
      parsed.port === '17493' &&
      (parsed.hostname === '127.0.0.1' ||
        parsed.hostname === 'localhost' ||
        parsed.hostname === '[::1]' ||
        parsed.hostname === '::1')
    );
  } catch {
    return false;
  }
}

export const useServerStore = create<ServerStore>()(
  persist(
    (set, get) => ({
      serverUrl: getDefaultServerUrl(),
      setServerUrl: (url) => {
        const prev = get().serverUrl;
        if (url === prev) return;
        // A different server means a different key store and token secret:
        // force a fresh whoami and drop anything minted by the old server.
        set({ serverUrl: url, authStatus: 'unknown', identity: null, mediaToken: null });
        invalidateAllServerData();
      },

      apiKey: null,
      setApiKey: (key) => {
        const prev = get().apiKey;
        if (key === prev) return;
        // Media tokens are bound to the key id, so a new key needs a new token.
        set({ apiKey: key, identity: null, mediaToken: null });
      },

      authStatus: 'unknown',
      setAuthStatus: (status) => set({ authStatus: status }),

      identity: null,
      setIdentity: (identity) => set({ identity }),

      mediaToken: null,
      setMediaToken: (token) => set({ mediaToken: token }),

      mode: 'local',
      setMode: (mode) => set({ mode }),

      keepServerRunningOnClose: false,
      setKeepServerRunningOnClose: (keepRunning) => set({ keepServerRunningOnClose: keepRunning }),

      customModelsDir: null,
      setCustomModelsDir: (dir) => set({ customModelsDir: dir }),
    }),
    {
      name: 'voicebox-server',
      // authStatus, identity and mediaToken are per-session by design. The
      // key is persisted for the browser shell only: the Tauri shell gets it
      // from Rust on every launch and must never leave it in localStorage.
      partialize: (state) => ({
        serverUrl: state.serverUrl,
        mode: state.mode,
        keepServerRunningOnClose: state.keepServerRunningOnClose,
        customModelsDir: state.customModelsDir,
        ...(isTauriRuntime() ? {} : { apiKey: state.apiKey }),
      }),
    },
  ),
);
