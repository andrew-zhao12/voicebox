import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Download, Loader2, Upload } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PronunciationEditor } from '@/components/SpeechSettings/PronunciationEditor';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { SpeechSettings } from '@/lib/api/types';
import {
  MAX_SPEECH_SETTINGS_FILE_BYTES,
  parseSpeechSettings,
  pronunciationEntries,
  speechSettingsFromForm,
} from '@/lib/utils/speech-settings';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { SettingSection } from './SettingRow';

export function SpeechPage() {
  const { t } = useTranslation();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const identity = useServerStore((state) => state.identity);
  const [selectedKey, setSelectedKey] = useState('');
  const [dirty, setDirty] = useState(false);
  const isAdmin = identity?.role === 'admin';
  const target = isAdmin ? selectedKey : '';
  const keyQuery = useQuery({
    queryKey: ['speech-settings-keys', serverUrl, identity?.key_id],
    queryFn: () => apiClient.listApiKeys(),
    enabled: isAdmin,
  });
  const queryKey = ['speech-settings', serverUrl, identity?.key_id, target];
  const settings = useQuery({
    queryKey,
    queryFn: () => apiClient.getSpeechSettings(target || undefined),
    enabled: !!identity,
    refetchOnWindowFocus: false,
  });

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h2 className="text-lg font-semibold">{t('speechSettings.title')}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{t('speechSettings.description')}</p>
      </div>
      {isAdmin && (
        <div className="space-y-2">
          <Label htmlFor="speech-client">{t('speechSettings.client')}</Label>
          <Select
            value={selectedKey || '__current__'}
            disabled={dirty}
            onValueChange={(value) => setSelectedKey(value === '__current__' ? '' : value)}
          >
            <SelectTrigger id="speech-client">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__current__">
                {t('speechSettings.currentClient', { id: identity.key_id })}
              </SelectItem>
              {keyQuery.data?.keys
                .filter((key) => key.id !== identity.key_id)
                .map((key) => (
                  <SelectItem key={key.id} value={key.id}>
                    {key.id}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          {dirty && (
            <p className="text-xs text-muted-foreground">{t('speechSettings.switchHint')}</p>
          )}
          {keyQuery.error && (
            <p role="alert" className="text-sm text-destructive">
              {keyQuery.error.message}
            </p>
          )}
        </div>
      )}
      {settings.isLoading && <output>{t('speechSettings.loading')}</output>}
      {settings.error ? (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>{settings.error.message}</p>
          <Button type="button" variant="outline" onClick={() => settings.refetch()}>
            {t('speechSettings.retry')}
          </Button>
        </div>
      ) : (
        settings.data && (
          <SpeechSettingsForm
            key={`${serverUrl}/${identity?.key_id}/${target}`}
            initial={settings.data}
            target={target}
            queryKey={queryKey}
            onDirtyChange={setDirty}
          />
        )
      )}
    </div>
  );
}

function SpeechSettingsForm({
  initial,
  target,
  queryKey,
  onDirtyChange,
}: {
  initial: SpeechSettings;
  target: string;
  queryKey: (string | undefined)[];
  onDirtyChange: (dirty: boolean) => void;
}) {
  const { t } = useTranslation();
  const { toast } = useToast();
  const platform = usePlatform();
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const [saved, setSaved] = useState(initial);
  const [entries, setEntries] = useState(() =>
    pronunciationEntries(initial.pronunciation_replacements),
  );
  const [prompt, setPrompt] = useState(initial.stt_prompt);
  const [vocabulary, setVocabulary] = useState(initial.stt_hotwords.join('\n'));
  const [fileBusy, setFileBusy] = useState(false);
  let data: SpeechSettings | undefined;
  let validationError = '';
  try {
    data = speechSettingsFromForm(entries, prompt, vocabulary);
  } catch (error) {
    const code = error instanceof Error ? error.message : 'invalidImport';
    validationError = t(`speechSettings.errors.${code}`);
  }
  const dirty = !data || JSON.stringify(data) !== JSON.stringify(saved);
  useEffect(() => {
    if (!dirty && JSON.stringify(initial) !== JSON.stringify(saved)) {
      setSaved(initial);
      setEntries(pronunciationEntries(initial.pronunciation_replacements));
      setPrompt(initial.stt_prompt);
      setVocabulary(initial.stt_hotwords.join('\n'));
    }
  }, [initial, saved, dirty]);
  useEffect(() => {
    onDirtyChange(dirty);
    return () => onDirtyChange(false);
  }, [dirty, onDirtyChange]);

  function load(value: SpeechSettings) {
    setEntries(pronunciationEntries(value.pronunciation_replacements));
    setPrompt(value.stt_prompt);
    setVocabulary(value.stt_hotwords.join('\n'));
  }

  const mutation = useMutation({
    mutationFn: (value: SpeechSettings) =>
      apiClient.updateSpeechSettings(value, target || undefined),
    onSuccess: (value) => {
      setSaved(value);
      load(value);
      queryClient.setQueryData(queryKey, value);
      toast({ title: t('speechSettings.saved') });
    },
    onError: (error) =>
      toast({
        title: t('speechSettings.saveFailed'),
        description: error.message,
        variant: 'destructive',
      }),
  });
  const busy = mutation.isPending || fileBusy;

  async function importSettings(file: File) {
    setFileBusy(true);
    try {
      if (file.size > MAX_SPEECH_SETTINGS_FILE_BYTES) throw new Error('largeImport');
      load(parseSpeechSettings(JSON.parse(await file.text())));
      toast({ title: t('speechSettings.imported'), description: t('speechSettings.importHint') });
    } catch (error) {
      const code =
        error instanceof SyntaxError
          ? 'invalidImport'
          : error instanceof Error
            ? error.message
            : 'invalidImport';
      toast({
        title: t('speechSettings.importFailed'),
        description: t(`speechSettings.errors.${code}`),
        variant: 'destructive',
      });
    } finally {
      setFileBusy(false);
    }
  }

  async function exportSettings(value: SpeechSettings) {
    setFileBusy(true);
    try {
      await platform.filesystem.saveFile(
        'voicebox-speech-settings.json',
        new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }),
        [{ name: 'Speech settings', extensions: ['json'] }],
      );
    } catch (error) {
      toast({
        title: t('speechSettings.exportFailed'),
        description: error instanceof Error ? error.message : undefined,
        variant: 'destructive',
      });
    } finally {
      setFileBusy(false);
    }
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (data && !busy) mutation.mutate(data);
      }}
    >
      <fieldset disabled={busy} className="space-y-8">
        <SettingSection
          title={t('speechSettings.pronunciations')}
          description={t('speechSettings.pronunciationDescription')}
        >
          <PronunciationEditor value={entries} onChange={setEntries} disabled={busy} />
        </SettingSection>
        <SettingSection
          title={t('speechSettings.transcription')}
          description={t('speechSettings.transcriptionDescription')}
        >
          <div className="space-y-2 py-3">
            <Label htmlFor="speech-prompt">{t('speechSettings.prompt')}</Label>
            <Textarea
              id="speech-prompt"
              value={prompt}
              rows={3}
              onChange={(event) => setPrompt(event.target.value)}
            />
            <p className="text-xs text-muted-foreground">{t('speechSettings.promptHint')}</p>
          </div>
          <div className="space-y-2 py-3">
            <Label htmlFor="speech-vocabulary">{t('speechSettings.vocabulary')}</Label>
            <Textarea
              id="speech-vocabulary"
              value={vocabulary}
              rows={5}
              onChange={(event) => setVocabulary(event.target.value)}
            />
            <p className="text-xs text-muted-foreground">{t('speechSettings.vocabularyHint')}</p>
          </div>
        </SettingSection>
        {validationError && (
          <p role="alert" className="text-sm text-destructive">
            {validationError}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <Button type="submit" disabled={!dirty || !data || busy}>
            {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t('speechSettings.save')}
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={!dirty || busy}
            onClick={() => load(saved)}
          >
            {t('speechSettings.discard')}
          </Button>
          <Button type="button" variant="outline" onClick={() => fileInput.current?.click()}>
            <Upload className="mr-2 h-4 w-4" />
            {t('speechSettings.import')}
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={!data || busy}
            onClick={() => {
              if (data) void exportSettings(data);
            }}
          >
            <Download className="mr-2 h-4 w-4" />
            {t('speechSettings.export')}
          </Button>
          <input
            ref={fileInput}
            type="file"
            accept=".json,application/json"
            className="hidden"
            aria-label={t('speechSettings.import')}
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = '';
              if (file) void importSettings(file);
            }}
          />
        </div>
      </fieldset>
    </form>
  );
}
