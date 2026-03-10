import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn = self.att(x, x)
        x = self.norm1(x + attn)
        ffn = self.ffn(x)
        return self.norm2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config


def build_micro_llm(vocab_size, seq_len):
    embed_dim = 64
    num_heads = 2
    ff_dim = 128

    inputs = tf.keras.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model