import type { PlatformLifecycle, ServerCredentials, ServerLogEntry } from '@/platform/types';
import { getDefaultServerUrl } from '@/stores/serverStore';

class WebLifecycle implements PlatformLifecycle {
  onServerReady?: () => void;

  async startServer(_remote = false, _modelsDir?: string | null): Promise<string> {
    // Web assumes server is running externally
    const serverUrl = import.meta.env.VITE_SERVER_URL || getDefaultServerUrl();
    this.onServerReady?.();
    return serverUrl;
  }

  async stopServer(): Promise<void> {
    // No-op for web - server is managed externally
  }

  async restartServer(_modelsDir?: string | null): Promise<string> {
    // No-op for web - server is managed externally
    return import.meta.env.VITE_SERVER_URL || getDefaultServerUrl();
  }

  async setKeepServerRunning(_keep: boolean): Promise<void> {
    // No-op for web
  }

  async setBackendOverride(_backend?: string | null): Promise<void> {
    // No-op for web - backend variant is managed externally
  }

  async setupWindowCloseHandler(): Promise<void> {
    // No-op for web - no window close handling needed
  }

  subscribeToServerLogs(_callback: (_entry: ServerLogEntry) => void): () => void {
    // No-op for web - server logs are not available
    return () => {};
  }

  async getCredentials(): Promise<ServerCredentials> {
    // The DEV guard is deliberate: a production bundle must never embed a
    // key, so `web/dist` always starts on the Connect screen.
    const devKey = import.meta.env.DEV ? import.meta.env.VITE_VOICEBOX_API_KEY : undefined;
    return {
      url: import.meta.env.VITE_SERVER_URL || getDefaultServerUrl(),
      apiKey: devKey || null,
    };
  }
}

export const webLifecycle = new WebLifecycle();
