Программа генерирует продолжение текста и печатает количество слов (всего и «новых» без промпта).
Требуется Python 3.9+ и пакеты: transformers, torch, tokenizers.

Пример вывода c prompt = "In a village of La Mancha", run_model("gpt2", prompt, max_new_tokens=60, temperature=0.7), run_model("distilgpt2", prompt, do_sample=False, max_new_tokens=40):
  Device set to use cuda:0
  The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.

  model = gpt2
  In a village of La Mancha, in the southern province of Lombardy, there is an old man who has been living for over 40 years. He was born on May 1st 1848 and died at his home by accident when he fell down from one tree into another while walking along with two other         children to school yesterday morning.

  Слов всего: 57 | Новых (без промпта): 51
  Device set to use cuda:0

  model = distilgpt2
  In a village of La Mancha, the town is home to many people who have been displaced by war.
  The villagers are now in their 20s and 30's but they say that it has become increasingly difficult for them to find

  Слов всего: 41 | Новых (без промпта): 35
