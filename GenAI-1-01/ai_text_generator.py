from transformers import pipeline, set_seed
import torch
import re
from typing import Any, Dict

def word_count(text: str) -> int:
    """
    Подсчет количества слов в тексте (латиница и кириллица).

    Параметры:
        text (str): Входной текст.

    Возвращает:
        int: Количество слов.
    """
    if not isinstance(text, str):
        raise TypeError("text должен быть строкой")
    return len(re.findall(r"[A-Za-zА-Яа-яЁё]+", text))

def run_model(model_id: str, prompt: str, **gen_kwargs) -> Dict[str, Any]:
    """
    Запускает генерацию текста с помощью Hugging Face pipeline и возвращает результат без вывода в консоль.

    Параметры:
        model_id (str): Идентификатор модели в Hub (например, 'gpt2').
        prompt (str): Промпт для генерации (не должен быть пустым или состоять только из пробелов).
        **gen_kwargs: Произвольные параметры генерации, например:
            - max_new_tokens (int)
            - do_sample (bool)
            - temperature (float)
            - top_p (float)
            - repetition_penalty (float)
            - num_return_sequences (int)
            - pad_token_id (int)

    Возвращает:
        dict: Словарь с ключами:
            - 'model' (str): Идентификатор модели.
            - 'prompt' (str): Исходный промпт.
            - 'generated_text' (str): Сгенерированный полный текст (включая промпт).
            - 'total_words' (int): Количество слов во всем тексте.
            - 'new_words' (int|None): Количество слов после промпта, если он является префиксом; иначе None.
            - 'gen_kwargs' (dict): Итоговые параметры генерации.
    """
    # Валидация входных данных
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id должен быть непустой строкой")
    if not isinstance(prompt, str):
        raise TypeError("prompt должен быть строкой")
    if prompt.strip() == "":
        raise ValueError("prompt не должен быть пустым или состоять только из пробелов")

    # Определяем устройство: GPU если доступно, иначе CPU
    device_idx = 0 if torch.cuda.is_available() else -1

    # Загружаем модель и токенизатор
    generator = pipeline("text-generation", model=model_id, device=device_idx)

    # Фиксируем сид для воспроизводимости
    set_seed(42)

    # Значения по умолчанию, можно переопределить в gen_kwargs
    pad_id = getattr(generator.tokenizer, "pad_token_id", None) or generator.tokenizer.eos_token_id
    defaults: Dict[str, Any] = {
        "max_new_tokens": 80,
        "do_sample": False,
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1,
        "pad_token_id": pad_id,
    }
    defaults.update(gen_kwargs)

    # Генерируем текст
    out = generator(prompt, **defaults)
    text = out[0]["generated_text"]

    # Считаем слова
    total_words = word_count(text)
    new_words = word_count(text[len(prompt):]) if text.startswith(prompt) else None

    # Возвращаем результат вместо печати
    return {
        "model": model_id,
        "prompt": prompt,
        "generated_text": text,
        "total_words": total_words,
        "new_words": new_words,
        "gen_kwargs": defaults,
    }


if __name__ == "__main__":
    # Пример использования: генерация текста с разными моделями и параметрами
    prompt = "In a village of La Mancha"

    examples = [
        ("gpt2", {"max_new_tokens": 60, "temperature": 0.7}),
        ("distilgpt2", {"do_sample": False, "max_new_tokens": 40}),
    ]

    for model_id, params in examples:
        result = run_model(model_id, prompt, **params)
        print(f"\nmodel = {result['model']}")
        print(result["generated_text"])
        if result["new_words"] is None:
            print(f"\nСлов всего: {result['total_words']}")
        else:
            print(f"\nСлов всего: {result['total_words']} | Новых (без промпта): {result['new_words']}")
