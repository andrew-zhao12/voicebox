import { describe, expect, spyOn, test } from 'bun:test';
import {
  MAX_SPEECH_SETTINGS_FILE_BYTES,
  newPronunciationEntry,
  parseSpeechSettings,
  pronunciationDictionary,
  pronunciationEntries,
  speechSettingsFromForm,
} from '../../app/src/lib/utils/speech-settings';

describe('portable speech defaults', () => {
  test('creates distinct rows without requiring secure-context randomUUID', () => {
    const uuid = spyOn(crypto, 'randomUUID').mockImplementation(() => {
      throw new Error('insecure origin');
    });
    try {
      const entries = pronunciationEntries({ API: 'A P I' });
      expect(newPronunciationEntry().id).not.toBe(entries[0].id);
      expect(entries[0].term).toBe('API');
      expect(uuid).not.toHaveBeenCalled();
    } finally {
      uuid.mockRestore();
    }
  });
  test('preserves literal dictionary text and empty values used to clear defaults', () => {
    const value = {
      pronunciation_replacements: { 'API.v2': 'A P I version two', ' Name ': ' Name ' },
      stt_prompt: ' Product names ',
      stt_hotwords: ['Voicebox', 'Live Kit'],
    };
    expect(parseSpeechSettings(JSON.parse(JSON.stringify(value)))).toEqual(value);
    expect(
      parseSpeechSettings({ pronunciation_replacements: {}, stt_prompt: '', stt_hotwords: [] }),
    ).toEqual({ pronunciation_replacements: {}, stt_prompt: '', stt_hotwords: [] });
  });

  test('rejects duplicates instead of silently overwriting a pronunciation', () => {
    const entries = pronunciationEntries({ API: 'A P I' });
    expect(() =>
      pronunciationDictionary([...entries, { id: 'other', term: 'API', spoken: 'api' }]),
    ).toThrow('duplicatePronunciation');
  });

  test('handles special object keys without changing prototypes', () => {
    const result = parseSpeechSettings(
      JSON.parse(
        '{"pronunciation_replacements":{"__proto__":"prototype","constructor":"constructor"},"stt_prompt":"","stt_hotwords":[]}',
      ),
    );
    expect(Object.keys(result.pronunciation_replacements)).toEqual(['__proto__', 'constructor']);
    expect(Object.getPrototypeOf(result.pronunciation_replacements)).toBe(Object.prototype);
  });

  test('counts Unicode code points consistently with backend character limits', () => {
    const term = '😀'.repeat(128);
    expect(pronunciationDictionary(pronunciationEntries({ [term]: 'smile' }))).toHaveProperty(term);
    expect(() => pronunciationDictionary(pronunciationEntries({ [`${term}😀`]: 'smile' }))).toThrow(
      'longPronunciation',
    );
  });

  test('allows valid Unicode exports larger than 64 KiB to be imported again', () => {
    const value = {
      pronunciation_replacements: Object.fromEntries(
        Array.from({ length: 32 }, (_, i) => [String(i), '😀'.repeat(254)]),
      ),
      stt_prompt: '😀'.repeat(4096),
      stt_hotwords: [...Array(40).fill('😀'.repeat(100)), '😀'.repeat(96)],
    };
    const exported = JSON.stringify(value, null, 2);
    expect(Buffer.byteLength(exported)).toBeGreaterThan(64 * 1024);
    expect(Buffer.byteLength(exported)).toBeLessThan(MAX_SPEECH_SETTINGS_FILE_BYTES);
    expect(parseSpeechSettings(JSON.parse(exported))).toEqual(value);
  });

  test('rejects oversized total dictionaries and vocabulary', () => {
    const dictionary = Object.fromEntries(
      Array.from({ length: 40 }, (_, i) => [`term${i}`, 'a'.repeat(256)]),
    );
    expect(() => pronunciationDictionary(pronunciationEntries(dictionary))).toThrow(
      'largeDictionary',
    );
    expect(() =>
      speechSettingsFromForm([], '', Array(42).fill('a'.repeat(100)).join('\n')),
    ).toThrow('largeVocabulary');
    expect(() => speechSettingsFromForm([], 'a'.repeat(4097), '')).toThrow('longPrompt');
    expect(() => speechSettingsFromForm([], '', Array(101).fill('a').join('\n'))).toThrow(
      'longVocabulary',
    );
  });

  test.each(
    [
      null,
      [],
      {},
      { stt_prompt: 3 },
      {
        pronunciation_replacements: { x: 5 },
        stt_prompt: '',
        stt_hotwords: [],
      },
      { pronunciation_replacements: {}, stt_prompt: '', stt_hotwords: ['one\ntwo'] },
    ].map((value) => [value]),
  )('rejects malformed imports', (value) => {
    expect(() => parseSpeechSettings(value)).toThrow('invalidImport');
  });
});
