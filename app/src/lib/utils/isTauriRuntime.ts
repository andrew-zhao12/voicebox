/**
 * Runtime check for the Tauri webview, usable from shared `app/` code that
 * must not pull in `@tauri-apps/api` (the web shell has no Tauri). Mirrors
 * `isTauri()` from `@tauri-apps/api/core`: the IPC bridge injects
 * `__TAURI_INTERNALS__` into every Tauri window regardless of `withGlobalTauri`.
 */
export function isTauriRuntime(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}
