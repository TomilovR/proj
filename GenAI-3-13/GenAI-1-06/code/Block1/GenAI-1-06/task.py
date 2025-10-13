# imports and constants
from pathlib import Path
from enum import Enum
import os.path

import torch
from transformers import pipeline

# Этот код был изменен, чтобы его можно было импортировать как модуль.
# Оригинальная функциональность для прямого запуска сохранена в блоке if __name__ == "__main__".

class Labels(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


def convert_to_readable(predicts: list[dict], texts: list[str]) -> list[dict]:
    """
    Преобразует вывод модели (звезды) в читаемый формат (POSITIVE/NEGATIVE/NEUTRAL).
    Эта функция используется как внешним интегратором, так и при прямом запуске.
    """
    star_to_sentiment = {
        '1 star': Labels.NEGATIVE.value,
        '2 stars': Labels.NEGATIVE.value,
        '3 stars': Labels.NEUTRAL.value, 
        '4 stars': Labels.POSITIVE.value,
        '5 stars': Labels.POSITIVE.value
    }
    
    readable_results = []
    for text, pred in zip(texts, predicts):
        label = pred['label']
        score = pred['score']
        readable_results.append({
            'text': text,
            'label': star_to_sentiment.get(label, Labels.NEUTRAL.value),
            'confidence': round(score, 4)
        })
    
    return readable_results


def analyze_sentiment_from_texts(texts: list[str]) -> list[dict]:
    """
    Анализирует список текстов на тональность.
    Это основная функция для импорта и использования в других модулях.
    """
    if not texts:
        return []
        
    try:
        # Загружаем модель только при вызове функции
        classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        print("Модель анализа тональности (GenAI-1-06) загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели sentiment-analysis: {e}")
        return []

    predicts = classifier(texts)
    
    # Преобразуем результат в наш стандартный формат
    readable_predicts = convert_to_readable(predicts, texts)
    
    return readable_predicts


# --- Блок для обратной совместимости при прямом запуске ---

def check_labels(labels: list[str]) -> bool:
    '''Check that labels-file is valid.''' 
    for label in labels:
        try:
            Labels(label)
        except ValueError:
            return False
    return True


def count_metrics(labels_path: str | Path, data: list[dict]) -> int:
    '''Calculate some metric(now only accuracy).''' 
    good_preds_count = 0
    with open(labels_path, "r", encoding='utf-8') as file:
        lines = [line.strip().lower() for line in file.readlines() if line.strip()]
        if not check_labels(lines):
            raise ValueError("Неправильный формат меток!")
        for i, label in enumerate(lines):
            dict_i = data[i]
            print(f"{dict_i['text']} : {dict_i['label']} : {label}")
            if label == dict_i['label']:
                good_preds_count += 1
    return good_preds_count


def sentiment_classification(opts):
    '''Original function for direct script execution.''' 
    # Динамический импорт, чтобы не мешать внешнему использованию
    from src.tools.config_loader import Config
    
    if not os.path.exists(opts.data_path):
        raise Exception(f"Файл {opts.data_path} не найден!")
    if not os.path.exists(opts.labels_path):
        raise Exception(f"Файл {opts.labels_path} не найден!")

    with open(opts.data_path, "r", encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
    
    # Используем новую основную функцию
    readable_predicts = analyze_sentiment_from_texts(lines)

    if not readable_predicts:
        print("Анализ тональности не дал результатов.")
        return

    print("\nРезультаты классификации:")
    print("-" * 50)
    print(f"<phrase> : <predict> : <label>")
    good_preds_count = count_metrics(opts.labels_path, readable_predicts)
    accuracy = float(good_preds_count) / len(readable_predicts) if readable_predicts else 0
    print("-" * 50)
    print(f"Точность(Accuracy) предсказаний модели: {accuracy}")


def main():
    # Динамический импорт, чтобы не мешать внешнему использованию
    from src.tools.parser import get_parser
    from src.tools.config_loader import load_config

    parser = get_parser()
    args = parser.parse_args()
    opts = load_config(args.config_path)
    sentiment_classification(opts)


if __name__ == "__main__":
    # Этот код выполнится только при запуске файла напрямую
    # и не будет мешать при импорте в review_integrator.py
    main()