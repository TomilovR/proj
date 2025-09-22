## Описание

Этот проект демонстрирует базовый и продвинутый подходы к генерации текста с помощью внешнего модуля `ai_text_generator`. В примере показано, как использовать промпт-инжиниринг для управления результатом генерации.

## Зависимости

- Python 3.8+
- Внешний модуль [`ai_text_generator`](https://github.com/TomilovR/proj/tree/main/GenAI-1-01)

## Использование

Запустите основной скрипт: python prompt_engineering.py
В результате вы увидите сравнение базового и инженерного промпта, а также статистику по количеству сгенерированных слов.

## Пример вывода
prompt_simple = "Story about Mars"
prompt_engineered = "Science fiction story: The spacecraft commander looked at Mars through the viewport. The red planet was closer than ever before. After months of travel, the crew was finally approaching their destination. The mission to Mars had been dangerous, but now they could see the ancient surface below them. Captain Sarah thought about the discoveries waiting"

Базовый вариант
Story about Mars:
The first time I saw the Martian landscape was in a movie called "Mars." It's an amazing place. The only thing that makes it different is how you can see things from space, and what we're seeing here on Earth right now are very real places where people live their lives with dignity and respect for one another. And there were some really beautiful landscapes out of this film when they

Слов всего: 76 | Новых (без промпта): 73

Промпт-инжиниринг
Science fiction story: The spacecraft commander looked at Mars through the viewport. The red planet was closer than ever before. After months of travel, the crew was finally approaching their destination. The mission to Mars had been dangerous, but now they could see the ancient surface below them. Captain Sarah thought about the discoveries waiting for her in space and decided that she would be a good candidate as an astronaut on board the craft.

The first thing he did when his ship came into contact with Earth's atmosphere is take off from home base by wayof or landing safely aboard another spaceship (the one which has no human onboard). He then took out all three cameras attached onto the back deck so that it wouldn't interfere

Слов всего: 129 | Новых (без промпта): 74
