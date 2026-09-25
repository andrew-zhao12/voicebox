import { ApiError, apiClient } from '@/lib/api/client';
import { markCredentialsReady } from '@/lib/credentials';
import type { Platform } from '@/platform/types';
import { isLoopbackVoiceboxServerUrl, useServerStore } from '@/stores/serverStore';

/**
 * Ask the backend who the stored key is and record the answer in the store.
 * `ok` unlocks the app; anything else routes to the Connect screen.
 */
export async function verifyConnection(): Promise<void> {
  const store = useServerStore.getState();
  if (!store.apiKey) {
    store.setAuthStatus('unauthorized');
    return;
  }
  try {
    const identity = await apiClient.whoami();
    store.setIdentity({ key_id: identity.key_id, role: identity.role });
    // The desktop UI manages the server, so it needs an admin key.
    store.setAuthStatus(identity.role === 'admin' ? 'ok' : 'forbidden');
    if (identity.role === 'admin') markCredentialsReady();
  } catch (error) {
    if (error instanceof ApiError) {
      store.setAuthStatus(error.status === 403 ? 'forbidden' : 'unauthorized');
    } else {
      store.setAuthStatus('offline');
    }
  }
}

/**
 * Seed the store from the platform (the sidecar URL and key in Tauri, the
 * dev key in `just dev-web`, nothing in a production web build) and verify.
 * A user-entered key or remote URL in the browser shell is left alone.
 */
export async function connectWithPlatformCredentials(platform: Platform): Promise<void> {
  const store = useServerStore.getState();
  try {
    const creds = await platform.lifecycle.getCredentials();
    const replaceUrl =
      platform.metadata.isTauri || !store.serverUrl || isLoopbackVoiceboxServerUrl(store.serverUrl);
    if (creds.url && replaceUrl) {
      store.setServerUrl(creds.url);
    }
    if (creds.apiKey && (platform.metadata.isTauri || !store.apiKey)) {
      store.setApiKey(creds.apiKey);
    }
  } catch (error) {
    console.warn('Could not obtain server credentials from the platform:', error);
  }
  await verifyConnection();
}
