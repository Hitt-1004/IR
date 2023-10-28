import random

def generate_bigram_text(bigram_model, start_word, num_words):
    generated_text = [start_word]
    current_word = start_word

    for _ in range(num_words - 1):
        next_word_options = bigram_model.get(current_word, [])
        if not next_word_options:
            break

        next_word = random.choice(next_word_options)
        generated_text.append(next_word)
        current_word = next_word

    return " ".join(generated_text)

# Usage
start_word = "your_seed_word"
num_words_to_generate = 50
generated_text = generate_bigram_text(bigram_model, start_word, num_words_to_generate)
print(generated_text)
