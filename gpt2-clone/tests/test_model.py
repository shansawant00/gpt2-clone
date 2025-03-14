def test_multi_head_attention():
    import numpy as np
    from src.model.attention import MultiHeadAttention

    # Initialize parameters
    num_heads = 8
    d_model = 64
    batch_size = 2
    seq_length = 10

    # Create a MultiHeadAttention instance
    mha = MultiHeadAttention(num_heads, d_model)

    # Create dummy input data
    query = np.random.rand(batch_size, seq_length, d_model)
    key = np.random.rand(batch_size, seq_length, d_model)
    value = np.random.rand(batch_size, seq_length, d_model)

    # Call the MultiHeadAttention
    output, attention_weights = mha.call(query, key, value)

    # Assertions to check the output shape
    assert output.shape == (batch_size, seq_length, d_model), "Output shape mismatch"
    assert attention_weights.shape == (batch_size, num_heads, seq_length, seq_length), "Attention weights shape mismatch"

def test_scaled_dot_product_attention():
    import numpy as np
    from src.model.attention import scaled_dot_product_attention

    # Create dummy input data
    query = np.random.rand(2, 10, 64)
    key = np.random.rand(2, 10, 64)
    value = np.random.rand(2, 10, 64)

    # Call the scaled dot product attention
    output, attention_weights = scaled_dot_product_attention(query, key, value)

    # Assertions to check the output shape
    assert output.shape == (2, 10, 64), "Output shape mismatch"
    assert attention_weights.shape == (2, 10, 10), "Attention weights shape mismatch"