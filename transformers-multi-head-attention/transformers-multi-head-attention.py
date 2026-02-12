import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch, seq_len, d_m = Q.shape
    d_k = d_m // num_heads
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    q = q.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = (q @ np.transpose(k, axes=(0, 1, 3, 2)) / np.sqrt(d_k))
    attn = softmax(scores)

    out = attn @ v

    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_m)
    out = out @ W_o
    
    return out