import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './en.json';
import ru from './ru.json';
import kg from './kg.json';

const browserLang = (navigator.language || 'en').split('-')[0].toLowerCase();
const initialLang = ['en', 'ru', 'kg'].includes(browserLang) ? browserLang : 'en';

void i18n
    .use(initReactI18next)
    .init({
        resources: {
            en: { translation: en },
            ru: { translation: ru },
            kg: { translation: kg },
        },
        lng: initialLang,
        fallbackLng: 'en',
        interpolation: {
            escapeValue: false,
        },
    });

export default i18n;
