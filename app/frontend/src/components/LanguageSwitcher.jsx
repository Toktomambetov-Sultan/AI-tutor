import { useTranslation } from 'react-i18next';

export default function LanguageSwitcher() {
    const { i18n, t } = useTranslation();

    return (
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {t('language.label')}:
            </span>
            <select
                value={i18n.language}
                onChange={(e) => i18n.changeLanguage(e.target.value)}
                style={{
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    padding: '0.25rem 0.4rem',
                    fontSize: '0.75rem',
                    background: 'var(--surface)',
                    color: 'var(--text)'
                }}
            >
                <option value="en">{t('language.en')}</option>
                <option value="ru">{t('language.ru')}</option>
                <option value="kg">{t('language.kg')}</option>
            </select>
        </label>
    );
}
