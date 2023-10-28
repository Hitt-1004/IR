import random

def generate_trigram_text(trigram_model, start_trigram, num_words):
    generated_text = list(start_trigram)

    for _ in range(num_words - 2):
        last_two_words = tuple(generated_text[-2:])
        next_word_options = trigram_model.get(last_two_words, [])
        if not next_word_options:
            break

        next_word = random.choice(next_word_options)
        generated_text.append(next_word)

    return " ".join(generated_text)

# Usage
start_trigram = ("your", "seed", "phrase")
num_words_to_generate = 50
generated_text = generate_trigram_text(trigram_model, start_trigram, num_words_to_generate)
print(generated_text)
