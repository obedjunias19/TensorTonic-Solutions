import numpy as np
import torch
def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    # not efficient
    PE = torch.zeros(seq_length, d_model, dtype=torch.float64)
    for pos in range(seq_length):
        for j in range(d_model):
            if j % 2 == 0:
                PE[pos, j] = np.sin(pos/(10000 ** ((2*j)/d_model)))
            else:
                PE[pos, j] = np.cos(pos/(10000 ** ((2*j)/d_model)))
    
    return PE.numpy()