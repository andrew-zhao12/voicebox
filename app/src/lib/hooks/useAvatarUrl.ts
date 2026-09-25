import { useMemo } from 'react';
import { apiClient } from '@/lib/api/client';
import { useServerStore } from '@/stores/serverStore';

/**
 * Tokened avatar URL for `<img src>`. Subscribes to the media token so the URL
 * (and therefore the `src`) changes when the token refreshes; without that an
 * `<img>` whose `onError` latched on a stale token would never reload.
 * Returns null when the profile has no avatar.
 */
export function useAvatarUrl(profileId: string | null | undefined): string | null {
  const token = useServerStore((state) => state.mediaToken?.token ?? null);
  return useMemo(
    () => (profileId ? apiClient.getAvatarUrl(profileId, token) : null),
    [profileId, token],
  );
}
