import json

from core.utils import detect_language, extract_text_from_lesson_context


def testdetect_language_ru_from_text():
    assert detect_language("Привет мир") == "ru"


def testdetect_language_en_from_text():
    assert detect_language("Hello world") == "en"


def testdetect_language_from_lesson_context_json_uses_content():
    lesson_context = json.dumps(
        {
            "lesson_title": "Урок русского языка",
            "course_title": "Русский 101",
            "materials": [
                {
                    "type": "text",
                    "content": "Это тестовый русский материал для распознавания речи.",
                    "title": "Текст урока",
                }
            ],
        }
    )

    extracted = extract_text_from_lesson_context(lesson_context)
    assert "Это тестовый русский материал" in extracted
    assert detect_language(extracted) == "ru"
