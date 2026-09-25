interface Window {
  __voiceboxServerStartedByApp?: boolean;
}

interface ImportMetaEnv {
  readonly VITE_SERVER_URL?: string;
  readonly VITE_APP_VERSION?: string;
  /**
   * Dev-only admin key for the browser shell (`just dev-web` exports it from
   * `data/api_key`). `web/src/platform/lifecycle.ts` ignores it outside
   * `import.meta.env.DEV`, so a production build never embeds a key.
   */
  readonly VITE_VOICEBOX_API_KEY?: string;
}

declare module 'virtual:changelog' {
  const raw: string;
  export default raw;
}
