import { Plus, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { newPronunciationEntry, type PronunciationEntry } from '@/lib/utils/speech-settings';

export function PronunciationEditor({
  value,
  onChange,
  disabled = false,
}: {
  value: PronunciationEntry[];
  onChange: (value: PronunciationEntry[]) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <div className="space-y-3">
      {value.map((entry, index) => (
        <div key={entry.id} className="flex items-center gap-2">
          <Input
            aria-label={t('speechSettings.termNumber', { number: index + 1 })}
            placeholder={t('speechSettings.term')}
            value={entry.term}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                value.map((row) =>
                  row.id === entry.id ? { ...row, term: event.target.value } : row,
                ),
              )
            }
          />
          <span className="text-muted-foreground" aria-hidden="true">
            →
          </span>
          <Input
            aria-label={t('speechSettings.spokenNumber', { number: index + 1 })}
            placeholder={t('speechSettings.spoken')}
            value={entry.spoken}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                value.map((row) =>
                  row.id === entry.id ? { ...row, spoken: event.target.value } : row,
                ),
              )
            }
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={t('speechSettings.removeNumber', { number: index + 1 })}
            disabled={disabled}
            onClick={() => onChange(value.filter((row) => row.id !== entry.id))}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={disabled || value.length >= 100}
        onClick={() => onChange([...value, newPronunciationEntry()])}
      >
        <Plus className="mr-2 h-4 w-4" />
        {t('speechSettings.addTerm')}
      </Button>
      <p className="text-xs text-muted-foreground">{t('speechSettings.dictionaryHint')}</p>
    </div>
  );
}
