from transformers import pipeline, set_seed
import torch
import re

def word_count(text: str) -> int:
    # Считаем слова (буквенные последовательности), RU/EN
    return len(re.findall(r"[A-Za-zА-Яа-яЁё]+", text))

def run_model(model_id: str, prompt: str, **gen_kwargs) -> None:
    # Определяем устройство: GPU если доступно, иначе CPU
    device_idx = 0 if torch.cuda.is_available() else -1
    # Загружаем модель и токенизатор
    generator = pipeline("text-generation", model=model_id, device=device_idx)
    # Фиксируем сид для воспроизводимости
    set_seed(42)

    # Значения по умолчанию, можно переопределить в gen_kwargs
    pad_id = getattr(generator.tokenizer, "pad_token_id", None) or generator.tokenizer.eos_token_id
    defaults = {
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

    # Вывод результатов
    print(f"\nmodel = {model_id}")
    print(text)
    if new_words is None:
        print(f"\nСлов всего: {total_words}")
    else:
        print(f"\nСлов всего: {total_words} | Новых (без промпта): {new_words}")


if __name__ == "__main__":
     # Пример использования: генерация текста с разными моделями и параметрами
    prompt = "In a village of La Mancha"
    run_model("gpt2", prompt, max_new_tokens=60, temperature=0.7)
    run_model("distilgpt2", prompt, do_sample=False, max_new_tokens=40)
