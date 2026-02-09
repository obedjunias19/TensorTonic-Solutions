import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    num = Q @ K.permute(0,2,1)
    den = math.sqrt(Q.shape[-1])
    return F.softmax(num/den, dim = 2) @ V