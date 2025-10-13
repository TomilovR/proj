# Извлечение именованных сущностей (NER)

Найти имена, места, организации.

## Данные

Синтетический текст с именами.

## Зависимости

- transformers
- torch

## Ссылка

https://huggingface.co/dslim/bert-base-NER (EN)

## Критерии успеха

Baseline: сущности найдены. 
Метрика: >3 сущности распознаны.

## Пример работы

Пример вывода для текста из `data/input.txt`:
```
Найденные сущности:
- Wolfgang (PER)
- Berlin (LOC)
- Acme Corporation (ORG)
- Eiffel Tower (LOC)
- Paris (LOC)
- Sarah (PER)
- New York (LOC)

Текст с подсветкой:
My name is [Wolfgang|PER] and I live in [Berlin|LOC]. I work for a company called [Acme Corporation|ORG].
Yesterday, I visited the [Eiffel Tower|LOC] in [Paris|LOC] with my friend [Sarah|PER].
We are planning a trip to [New York|LOC] next month.
```