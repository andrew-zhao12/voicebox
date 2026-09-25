/**
 * Startup gate for code that must not hit the API before the server URL and
 * API key are in `useServerStore`.
 *
 * Each webview has its own JS context, so this is per window: the main window
 * marks it once `/auth/whoami` accepts the key, and the floating dictate
 * window marks it after `platform.lifecycle.getCredentials()` resolves
 * (zustand persist does not sync across webviews, and in Tauri the key is
 * memory-only). `useCaptureRecordingSession` awaits it before uploading.
 */
let ready = false;
let resolveReady: () => void = () => {};
const readyPromise = new Promise<void>((resolve) => {
  resolveReady = resolve;
});

export function markCredentialsReady(): void {
  if (ready) return;
  ready = true;
  resolveReady();
}

export function credentialsReady(): Promise<void> {
  return readyPromise;
}
