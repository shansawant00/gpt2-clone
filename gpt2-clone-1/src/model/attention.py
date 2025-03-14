def scaled_dot_product_attention(query, key, value, mask=None):
    import numpy as np

    # Calculate the dot product of the query and key
    matmul_qk = np.dot(query, key.T)

    # Scale the dot product
    d_k = key.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # Apply the mask (if any)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax to get the attention weights
    attention_weights = np.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights by the value
    output = np.dot(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention:
    def __init__(self, num_heads, d_model):
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Initialize weights for query, key, value, and output
        self.Wq = np.random.rand(d_model, d_model)
        self.Wk = np.random.rand(d_model, d_model)
        self.Wv = np.random.rand(d_model, d_model)
        self.Wo = np.random.rand(d_model, d_model)

    def split_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return np.transpose(x, perm=(0, 2, 1, 3))

    def call(self, query, key, value, mask=None):
        # Linear transformations
        query = np.dot(query, self.Wq)
        key = np.dot(key, self.Wk)
        value = np.dot(value, self.Wv)

        # Split into multiple heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Scaled dot-product attention
        output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        # Concatenate heads and apply final linear transformation
        output = np.transpose(output, perm=(0, 2, 1, 3))
        output = output.reshape(output.shape[0], -1, self.d_model)
        return np.dot(output, self.Wo), attention_weights