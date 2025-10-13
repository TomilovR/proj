"""
Скрипт-оркестратор для комплексного анализа отзывов на продукты.

Этот скрипт интегрирует функциональность из трех предыдущих заданий:
- GenAI-1-06: Анализ тональности (Sentiment Analysis).
- GenAI-1-20: Извлечение именованных сущностей (NER) для поиска аспектов.
- GenAI-1-04: Суммаризация текста.

Процесс работы:
1. Загружает отзывы из CSV-файла.
2. Для каждого отзыва определяет тональность и извлекает аспекты (сущности).
3. Группирует аспекты по продуктам и тональности.
4. Генерирует текстовое описание мнений по каждому продукту.
5. Суммаризирует это описание для получения краткой сводки.
6. Сохраняет итоговый отчет в файл.
"""

import pandas as pd
from transformers import pipeline, Pipeline
from collections import defaultdict
import sys
import os

# --- Блок 1: Настоящая интеграция через импорт ---

# Добавляем пути к модулям в sys.path, чтобы Python мог их найти
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, 'GenAI-1-04'))
    sys.path.append(os.path.join(current_dir, 'GenAI-1-20'))
    # Путь к отрефакторенному модулю GenAI-1-06
    sys.path.append(os.path.join(current_dir, 'GenAI-1-06', 'code', 'Block1', 'GenAI-1-06'))

    # Импортируем функции напрямую из файлов заданий
    from summarizer import summarize_text
    from recognize_entities import recognize_entities
    from task import analyze_sentiment_from_texts

    print("Все функции из модулей GenAI-1-04, GenAI-1-06 и GenAI-1-20 успешно импортированы.")

except ImportError as e:
    print(f"Критическая ошибка импорта! Не удалось найти модули или функции.", file=sys.stderr)
    print(f"Убедитесь, что структура папок верна и файлы существуют.", file=sys.stderr)
    print(f"Детали ошибки: {e}", file=sys.stderr)
    sys.exit(1)


def load_models() -> dict:
    """
    Загружает и инициализирует модели для NER и Суммаризации.
    Модель для анализа тональности загружается внутри своей импортированной функции.
    """
    print("Загрузка моделей для NER и Суммаризации...")
    try:
        models = {
            'ner': pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple'),
            'summarizer': pipeline('summarization', model='facebook/bart-large-cnn')
        }
        print("Модели успешно загружены.")
        return models
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}", file=sys.stderr)
        sys.exit(1)


def load_reviews(file_path: str):
    """Загружает отзывы из CSV-файла с валидацией."""
    try:
        # Читаем CSV с правильными параметрами для обработки пробелов
        df = pd.read_csv(file_path, skipinitialspace=True)
        
        # Валидация: проверяем наличие нужных колонок
        required_columns = ['product_id', 'review_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Ошибка: В CSV отсутствуют колонки: {missing_columns}", file=sys.stderr)
            return None
        
        # Валидация: проверяем, что DataFrame не пустой
        if df.empty:
            print(f"Ошибка: CSV-файл '{file_path}' пустой.", file=sys.stderr)
            return None
        
        # Очищаем данные от лишних пробелов
        df['product_id'] = df['product_id'].str.strip()
        df['review_text'] = df['review_text'].str.strip()
        
        # Проверяем, что нет пустых значений
        if df[required_columns].isnull().any().any():
            print(f"Предупреждение: Обнаружены пустые значения в данных.", file=sys.stderr)
            df = df.dropna(subset=required_columns)
        
        return df
        
    except FileNotFoundError:
        print(f"Ошибка: Файл с отзывами '{file_path}' не найден.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Ошибка при чтении CSV-файла: {e}", file=sys.stderr)
        return None


def generate_report(reviews_df: pd.DataFrame, models: dict) -> str:
    """
    Генерирует полный отчет на основе анализа отзывов, вызывая импортированные функции.
    """
    product_aspects = defaultdict(lambda: defaultdict(list))
    all_texts = reviews_df['review_text'].tolist()

    # Шаг 1: Анализ тональности для всех отзывов одним пакетом (вызов функции из GenAI-1-06)
    print("Выполнение анализа тональности (используется модуль GenAI-1-06)...")
    sentiment_results = analyze_sentiment_from_texts(all_texts)
    text_to_sentiment = {res['text']: res['label'] for res in sentiment_results}

    print("Выполнение извлечения аспектов (используется модуль GenAI-1-20)...")
    for index, row in reviews_df.iterrows():
        product = row['product_id']
        review_text = row['review_text']
        
        sentiment = text_to_sentiment.get(review_text, 'NEUTRAL')
        
        # Шаг 2: Извлечение аспектов (вызов функции из GenAI-1-20)
        all_entities = recognize_entities(models['ner'], review_text)
        aspects = [
            entity['word'] for entity in all_entities
            if entity['entity_group'] not in ['PER', 'LOC', 'DATE', 'MISC']
        ]
        
        if sentiment != 'NEUTRAL' and aspects:
            product_aspects[product][sentiment].extend(aspects)
    
    print("Анализ завершен. Генерация сводок (используется модуль GenAI-1-04)...")
    
    # Создаем словарь для хранения текстов отзывов по продуктам и тональности
    product_reviews = defaultdict(lambda: defaultdict(list))
    for index, row in reviews_df.iterrows():
        product = row['product_id']
        review_text = row['review_text']
        sentiment = text_to_sentiment.get(review_text, 'NEUTRAL')
        if sentiment != 'NEUTRAL':
            product_reviews[product][sentiment].append(review_text)
    
    final_report = "--- Сводный отчет по анализу отзывов ---\n\n"
    for product, sentiments in product_aspects.items():
        final_report += f"Продукт: {product}\n"
        
        # Собираем статистику по аспектам
        aspect_summary = []
        if sentiments['positive']:
            unique_positive = set(sentiments['positive'])
            # Фильтруем странные токены из NER
            clean_positive = [a for a in unique_positive if not a.startswith('##') and len(a) > 1]
            if clean_positive:
                aspect_summary.append(f"Положительные отзывы упоминают: {', '.join(sorted(clean_positive)[:5])}")
        
        if sentiments['negative']:
            unique_negative = set(sentiments['negative'])
            # Фильтруем странные токены из NER
            clean_negative = [a for a in unique_negative if not a.startswith('##') and len(a) > 1]
            if clean_negative:
                aspect_summary.append(f"Негативные отзывы связаны с: {', '.join(sorted(clean_negative)[:5])}")
        
        # Собираем текст для суммаризации из реальных отзывов
        reviews_for_summary = []
        if product in product_reviews:
            if product_reviews[product]['positive']:
                reviews_for_summary.extend(product_reviews[product]['positive'][:2])
            if product_reviews[product]['negative']:
                reviews_for_summary.extend(product_reviews[product]['negative'][:2])
        
        if reviews_for_summary:
            # Объединяем отзывы для суммаризации
            combined_reviews = " ".join(reviews_for_summary)
            
            # Ограничиваем длину входного текста
            words = combined_reviews.split()
            if len(words) > 500:
                combined_reviews = " ".join(words[:500])
            
            # Суммаризация
            try:
                summary = summarize_text(models['summarizer'], combined_reviews, 
                                       max_length=100, min_length=30)
            except Exception as e:
                # Если суммаризация не удалась, используем простое описание
                summary = " ".join(aspect_summary)
        else:
            summary = " ".join(aspect_summary) if aspect_summary else "Недостаточно данных для анализа."
        
        # Добавляем детали по аспектам
        if aspect_summary:
            final_report += f"Ключевые аспекты:\n"
            for aspect in aspect_summary:
                final_report += f"  - {aspect}\n"
        
        final_report += f"Краткая сводка: {summary}\n"
        final_report += "-" * 50 + "\n\n"
        
    return final_report


def save_report(report: str, file_path: str):
    """Сохраняет отчет в файл."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Отчет успешно сохранен в файл: {file_path}")


def main():
    """Главная функция-оркестратор."""
    REVIEWS_FILE = "reviews_data.csv"
    REPORT_FILE = "analysis_report.txt"

    # Сначала проверяем файл, потом загружаем модели
    reviews_df = load_reviews(REVIEWS_FILE)
    if reviews_df is None:
        return
    
    print(f"Загружено {len(reviews_df)} отзывов.")
    
    models = load_models()
    
    try:
        report = generate_report(reviews_df, models)
        save_report(report, REPORT_FILE)
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()