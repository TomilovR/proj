"""
Скрипт для извлечения и подсветки именованных сущностей (NER) в тексте.
Может быть запущен как самостоятельный скрипт или импортирован как модуль.
"""

import sys
from transformers import pipeline
from typing import List, Dict, Any

def load_ner_model():
    """Загружает и инициализирует NER-модель."""
    print("Загрузка NER-модели...")
    ner_pipeline = pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')
    print("Модель успешно загружена.")
    return ner_pipeline

def read_text_from_file(file_path: str) -> str:
    """Читает текст из указанного файла с обработкой ошибок."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {file_path}", file=sys.stderr)
        sys.exit(1)

def recognize_entities(ner_pipeline, text: str) -> List[Dict[str, Any]]:
    """Распознает именованные сущности в тексте."""
    return ner_pipeline(text)

def highlight_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """Подсвечивает распознанные сущности в исходном тексте."""
    for entity in reversed(entities):
        start = entity['start']
        end = entity['end']
        label = entity['entity_group']
        text = f"{text[:start]}[{text[start:end]}|{label}]{text[end:]}"
    return text

def process_ner_from_file(file_path: str) -> Dict[str, Any]:
    """
    Выполняет полный цикл NER для файла: загрузка, чтение, распознавание.

    Эта функция является основной точкой входа при использовании скрипта как библиотеки.

    Args:
        file_path (str): Путь к входному файлу.

    Returns:
        Dict[str, Any]: Словарь с результатами, содержащий:
                        - 'entities': список найденных сущностей.
                        - 'highlighted_text': текст с подсветкой.
    """
    ner_model = load_ner_model()
    text_to_analyze = read_text_from_file(file_path)
    found_entities = recognize_entities(ner_model, text_to_analyze)
    highlighted_version = highlight_entities(text_to_analyze, found_entities)
    
    return {
        'entities': found_entities,
        'highlighted_text': highlighted_version
    }

# --- Основной блок для прямого запуска скрипта ---
if __name__ == "__main__":
    
    # 1. Укажите путь к файлу здесь
    INPUT_FILE_PATH = 'data/input.txt'

    # 2. Запуск основной функции
    results = process_ner_from_file(INPUT_FILE_PATH)

    # 3. Вывод результатов
    print("\nНайденные сущности:")
    for entity in results['entities']:
        print(f"- {entity['word']} ({entity['entity_group']})")

    print("\nТекст с подсветкой:")
    print(results['highlighted_text'])