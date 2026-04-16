import json

from core.conversation import _detect_language, _extract_text_from_lesson_context


def test_detect_language_ru_from_text():
    assert _detect_language("Привет мир") == "ru"


def test_detect_language_en_from_text():
    assert _detect_language("Hello world") == "en"


def test_detect_language_from_lesson_context_json_uses_content():
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

    extracted = _extract_text_from_lesson_context(lesson_context)
    assert "Это тестовый русский материал" in extracted
    assert _detect_language(extracted) == "ru"
