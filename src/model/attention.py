def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implements scaled dot-product attention mechanism.
    Compute Requirements:
    - Matrix multiplication: O(N^2 * d) operations on CPU/GPU
    - Memory: O(N^2) for attention weights
    - Recommended: GPU with at least 8GB VRAM for reasonable sequence lengths
    
    Args:
        query: Query tensor of shape (batch_size, seq_len_q, depth)
        key: Key tensor of shape (batch_size, seq_len_k, depth)
        value: Value tensor of shape (batch_size, seq_len_v, depth)
        mask: Optional mask tensor for padding/causal attention
    
    Returns:
        output: Attention output
        attention_weights: Attention distribution
    """
    import numpy as np  # Can be replaced with torch/tensorflow for GPU acceleration

    # Calculate attention scores - requires matrix multiplication
    # Computational complexity: O(N^2 * d) where N is sequence length
    matmul_qk = np.dot(query, key.T)

    # Scale to prevent softmax saturation
    d_k = key.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # Apply mask for causal/padding attention
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax normalization - requires exponential operations
    attention_weights = np.softmax(scaled_attention_logits, axis=-1)

    # Apply attention weights to values
    # Computational complexity: O(N^2 * d)
    output = np.dot(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism.
    Compute Requirements:
    - Parallel processing capability for multiple attention heads
    - Memory: O(batch_size * num_heads * seq_len * d_model)
    - Recommended: GPU with tensor cores for efficient matrix operations
    """
    
    def __init__(self, num_heads, d_model):
        """
        Initialize multi-head attention.
        
        Args:
            num_heads: Number of attention heads
            d_model: Model dimension size
        """
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Initialize trainable weight matrices
        # Could use torch.nn.Parameter or tf.Variable for GPU training
        self.Wq = np.random.rand(d_model, d_model)
        self.Wk = np.random.rand(d_model, d_model)
        self.Wv = np.random.rand(d_model, d_model)
        self.Wo = np.random.rand(d_model, d_model)

    def split_heads(self, x):
        """
        Split input tensor into multiple heads.
        Memory requirement: O(batch_size * seq_len * d_model)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Reshaped tensor with separated heads
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return np.transpose(x, perm=(0, 2, 1, 3))

    def call(self, query, key, value, mask=None):
        """
        Process inputs through multi-head attention.
        Total Computation: O(batch_size * num_heads * seq_len^2 * depth)
        
        Args:
            query, key, value: Input tensors
            mask: Optional attention mask
        Returns:
            output: Processed attention output
            attention_weights: Attention distributions
        """
        # Linear projections - can be parallelized on GPU
        query = np.dot(query, self.Wq)
        key = np.dot(key, self.Wk)
        value = np.dot(value, self.Wv)

        # Split heads for parallel processing
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Parallel attention computation across heads
        output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        # Merge heads and apply output transformation
        output = np.transpose(output, perm=(0, 2, 1, 3))
        output = output.reshape(output.shape[0], -1, self.d_model)
        return np.dot(output, self.Wo), attention_weights