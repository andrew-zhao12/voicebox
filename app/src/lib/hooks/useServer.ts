import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { useServerStore } from '@/stores/serverStore';

export function useServerHealth() {
  const serverUrl = useServerStore((state) => state.serverUrl);

  return useQuery({
    queryKey: ['server', 'health', serverUrl],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30000, // Check every 30 seconds
    retry: 1,
  });
}

/**
 * Who the stored API key is, refreshed like the health check. Unlike
 * `/health` (which answers anonymously), a failure here tells Offline
 * (network error) apart from Unauthorized (`ApiError` 401/403).
 */
export function useServerIdentity() {
  const serverUrl = useServerStore((state) => state.serverUrl);
  const hasKey = useServerStore((state) => Boolean(state.apiKey));

  return useQuery({
    queryKey: ['server', 'identity', serverUrl, hasKey],
    queryFn: () => apiClient.whoami(),
    refetchInterval: 30000,
    retry: 1,
  });
}
