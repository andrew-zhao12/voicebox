import type { SpeechSettings } from '../api/types';

export const MAX_SPEECH_SETTINGS_FILE_BYTES = 128 * 1024;

export interface PronunciationEntry {
  id: string;
  term: string;
  spoken: string;
}

const characters = (value: string) => Array.from(value).length;

export function newPronunciationEntry(term = '', spoken = ''): PronunciationEntry {
  // Row identity is local UI state. getRandomValues also works on HTTP LAN servers,
  // where randomUUID is unavailable because the origin is not a secure context.
  const id = Array.from(crypto.getRandomValues(new Uint32Array(4)), (part) =>
    part.toString(16),
  ).join('-');
  return { id, term, spoken };
}

export function pronunciationEntries(
  value: Record<string, string> | null | undefined,
): PronunciationEntry[] {
  return Object.entries(value ?? {}).map(([term, spoken]) => newPronunciationEntry(term, spoken));
}

/** Keep literal spelling intact and reject duplicate rows instead of silently losing one. */
export function pronunciationDictionary(entries: PronunciationEntry[]): Record<string, string> {
  if (entries.length > 100) throw new Error('tooManyPronunciations');
  const terms = new Set<string>();
  let total = 0;
  for (const { term, spoken } of entries) {
    if (!term.trim() || !spoken.trim()) throw new Error('emptyPronunciation');
    if (characters(term) > 128 || characters(spoken) > 256) {
      throw new Error('longPronunciation');
    }
    if (terms.has(term)) throw new Error('duplicatePronunciation');
    terms.add(term);
    total += characters(term) + characters(spoken);
  }
  if (total > 8192) throw new Error('largeDictionary');
  return Object.fromEntries(entries.map(({ term, spoken }) => [term, spoken]));
}

export function speechSettingsFromForm(
  entries: PronunciationEntry[],
  prompt: string,
  vocabulary: string,
): SpeechSettings {
  const hotwords = vocabulary
    .split(/\r?\n/)
    .map((word) => word.trim())
    .filter(Boolean);
  if (characters(prompt) > 4096) throw new Error('longPrompt');
  if (hotwords.length > 100 || hotwords.some((word) => characters(word) > 100)) {
    throw new Error('longVocabulary');
  }
  if (hotwords.reduce((total, word) => total + characters(word), 0) > 4096) {
    throw new Error('largeVocabulary');
  }
  return {
    pronunciation_replacements: pronunciationDictionary(entries),
    stt_prompt: prompt,
    stt_hotwords: hotwords,
  };
}

/** Import the complete portable GET/PUT payload, without identifiers or credentials. */
export function parseSpeechSettings(value: unknown): SpeechSettings {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('invalidImport');
  }
  const data = value as Record<string, unknown>;
  const dictionary = data.pronunciation_replacements;
  if (
    !dictionary ||
    typeof dictionary !== 'object' ||
    Array.isArray(dictionary) ||
    Object.values(dictionary).some((spoken) => typeof spoken !== 'string') ||
    typeof data.stt_prompt !== 'string' ||
    !Array.isArray(data.stt_hotwords) ||
    data.stt_hotwords.some(
      (word) => typeof word !== 'string' || !word.trim() || /[\r\n]/.test(word),
    )
  ) {
    throw new Error('invalidImport');
  }
  return speechSettingsFromForm(
    pronunciationEntries(dictionary as Record<string, string>),
    data.stt_prompt,
    data.stt_hotwords.join('\n'),
  );
}
