import { zodResolver } from '@hookform/resolvers/zod';
import { KeyRound, Loader2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import * as z from 'zod';
import { TitleBarDragRegion } from '@/components/TitleBarDragRegion';
import { Button } from '@/components/ui/button';
import { Form, FormControl, FormField, FormItem, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { ApiError, apiClient } from '@/lib/api/client';
import type { WhoAmIResponse } from '@/lib/api/types';
import { verifyConnection } from '@/lib/connection';
import { TOP_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { markCredentialsReady } from '@/lib/credentials';
import { queryClient } from '@/lib/queryClient';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';

type ConnectFormValues = { serverUrl: string; apiKey: string };

/**
 * Gate shown until `/auth/whoami` accepts an admin key. Lets the user point
 * the app at a server and paste its key; a stored key is re-checked on its
 * own whenever the status is still unknown (startup, server URL change).
 */
export function ConnectScreen() {
  const { t } = useTranslation();
  const platform = usePlatform();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const apiKey = useServerStore((state) => state.apiKey);
  const authStatus = useServerStore((state) => state.authStatus);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);

  const schema = useMemo(
    () =>
      z.object({
        serverUrl: z.string().url(t('connect.invalidUrl')),
        apiKey: z.string().trim().min(1, t('connect.keyRequired')),
      }),
    [t],
  );
  const form = useForm<ConnectFormValues>({
    resolver: zodResolver(schema),
    defaultValues: { serverUrl, apiKey: apiKey ?? '' },
  });

  useEffect(() => {
    form.reset({ serverUrl, apiKey: apiKey ?? '' });
  }, [serverUrl, apiKey, form]);

  useEffect(() => {
    if (authStatus === 'unknown') {
      void verifyConnection();
    }
  }, [authStatus]);

  const describeError = (error: unknown): string => {
    if (error instanceof ApiError) {
      if (error.status === 401) return t('connect.invalidKey');
      if (error.status === 403) return t('connect.forbidden');
      return error.message;
    }
    return t('connect.unreachable');
  };

  const probe = async (values: ConnectFormValues): Promise<WhoAmIResponse | null> => {
    setBusy(true);
    setResult(null);
    try {
      const identity = await apiClient.whoami({
        baseUrl: values.serverUrl,
        apiKey: values.apiKey.trim(),
      });
      const isAdmin = identity.role === 'admin';
      setResult({
        ok: isAdmin,
        message: isAdmin
          ? t('connect.testOk', { keyId: identity.key_id })
          : t('connect.clientKey', { keyId: identity.key_id }),
      });
      return identity;
    } catch (error) {
      setResult({ ok: false, message: describeError(error) });
      return null;
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = async (values: ConnectFormValues) => {
    const identity = await probe(values);
    if (!identity) return;
    const store = useServerStore.getState();
    store.setServerUrl(values.serverUrl.replace(/\/+$/, ''));
    store.setApiKey(values.apiKey.trim());
    store.setIdentity({ key_id: identity.key_id, role: identity.role });
    if (identity.role !== 'admin') {
      store.setAuthStatus('forbidden');
      return;
    }
    store.setAuthStatus('ok');
    markCredentialsReady();
    queryClient.invalidateQueries();
  };

  const statusMessage =
    authStatus === 'unauthorized'
      ? t('connect.status.unauthorized')
      : authStatus === 'forbidden'
        ? t('connect.status.forbidden')
        : authStatus === 'offline'
          ? t('connect.status.offline')
          : null;

  return (
    <div
      className={cn(
        'min-h-screen bg-background flex items-center justify-center',
        TOP_SAFE_AREA_PADDING,
      )}
    >
      <TitleBarDragRegion />
      <div className="w-full max-w-md space-y-6 rounded-xl border border-border/60 p-8">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <KeyRound className="h-5 w-5 text-accent" />
            <h1 className="text-lg font-semibold">{t('connect.title')}</h1>
          </div>
          <p className="text-sm text-muted-foreground">{t('connect.description')}</p>
        </div>

        {authStatus === 'unknown' ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{t('connect.checking')}</span>
          </div>
        ) : statusMessage ? (
          <p className="text-sm text-destructive">{statusMessage}</p>
        ) : null}

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="serverUrl"
              render={({ field }) => (
                <FormItem>
                  <label className="text-sm font-medium" htmlFor="connect-server-url">
                    {t('connect.serverUrl')}
                  </label>
                  <FormControl>
                    <Input
                      id="connect-server-url"
                      placeholder="http://127.0.0.1:17493"
                      autoComplete="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="apiKey"
              render={({ field }) => (
                <FormItem>
                  <label className="text-sm font-medium" htmlFor="connect-api-key">
                    {t('connect.apiKey')}
                  </label>
                  <FormControl>
                    <Input
                      id="connect-api-key"
                      type="password"
                      placeholder={t('connect.apiKeyPlaceholder')}
                      autoComplete="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            {result && (
              <p className={cn('text-sm', result.ok ? 'text-accent' : 'text-destructive')}>
                {result.message}
              </p>
            )}

            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => form.handleSubmit(probe)()}
              >
                {t('connect.test')}
              </Button>
              <Button type="submit" size="sm" disabled={busy}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : t('connect.connect')}
              </Button>
            </div>
          </form>
        </Form>

        {platform.metadata.isTauri && (
          <p className="text-xs text-muted-foreground">{t('connect.tauriHint')}</p>
        )}
      </div>
    </div>
  );
}
