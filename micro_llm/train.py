import numpy as np
import tensorflow as tf
from tokenizer import CharTokenizer
from model.micro_llm import build_micro_llm

# Load corpus
text = open("data/corpus.txt").read()
tokenizer = CharTokenizer(text)
encoded = tokenizer.encode(text)

# Sequence length
SEQ_LEN = 16

# Build training data
X, y = [], []
for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i:i + SEQ_LEN])
    y.append(encoded[i + 1:i + SEQ_LEN + 1])

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

# Build model
model = build_micro_llm(tokenizer.vocab_size, SEQ_LEN)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy"
)

# Train
model.fit(X, y, epochs=20, batch_size=16)

# Save model
model.save("micro_llm_model.keras")
print("MODEL SAVED SUCCESSFULLY")
