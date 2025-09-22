from ai_text_generator import run_model

if __name__ == "__main__":
    model_id = "gpt2"

    # 1) Базовый вариант: короткий промпт
    prompt_simple = "Story about Mars"
    base = run_model(model_id, prompt_simple)

    # 2) Промпт-инжиниринг: детальный промпт (английский) + ограничения
    #    Требование задания: max_length=120, no_repeat_ngram_size=2
    prompt_engineered = "Science fiction story: The spacecraft commander looked at Mars through the viewport. The red planet was closer than ever before. After months of travel, the crew was finally approaching their destination. The mission to Mars had been dangerous, but now they could see the ancient surface below them. Captain Sarah thought about the discoveries waiting"
    engineered = run_model(
        model_id,
        prompt_engineered,
        max_length=120,
        no_repeat_ngram_size=2,
    )
    # Вывод результатов
    print("Базовый вариант")
    print(base["generated_text"])
    print(f"\nСлов всего: {base['total_words']} | Новых (без промпта): {base['new_words']}")
    print("\nПромпт-инжиниринг")
    print(engineered["generated_text"])
    print(f"\nСлов всего: {engineered['total_words']} | Новых (без промпта): {engineered['new_words']}"
