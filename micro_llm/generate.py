import tensorflow as tf
import numpy as np

# 🔴 IMPORTANT: this import registers the custom layer
from model.micro_llm import TransformerBlock  

from tokenizer import CharTokenizer


# Load corpus and tokenizer
with open("data/corpus.txt", "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

# Load trained model (Keras 3 format)
model = tf.keras.models.load_model("micro_llm_model.keras")


def generate(start_text, length=100):
    SEQ_LEN = 16

    # Encode seed text
    input_ids = tokenizer.encode(start_text)

    # 🔥 FIX: left-pad to SEQ_LEN
    if len(input_ids) < SEQ_LEN:
        input_ids = [0] * (SEQ_LEN - len(input_ids)) + input_ids
    else:
        input_ids = input_ids[-SEQ_LEN:]

    for _ in range(length):
        x = np.array([input_ids])

        preds = model(x, training=False)

        next_id = tf.random.categorical(
            preds[:, -1, :],
            num_samples=1
        )[0, 0].numpy()

        input_ids.append(next_id)
        input_ids = input_ids[-SEQ_LEN:]

    return tokenizer.decode(input_ids)



# 🔥 Run generation
if __name__ == "__main__":
    print("\n=== MICRO-LLM OUTPUT ===\n")
    print(generate("machine", length=120))
