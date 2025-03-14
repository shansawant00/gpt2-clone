class TransformerBlock:
    def __init__(self, num_heads, d_model, d_ff, dropout_rate):
        self.attention = MultiHeadAttention(num_heads, d_model)
        self.ffn = self.feed_forward_network(d_model, d_ff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def feed_forward_network(self, d_model, d_ff):
        return Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_model)
        ])

    def call(self, x, training, mask):
        attn_output, _ = self.attention.call(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output, training=training))