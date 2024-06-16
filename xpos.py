import torch

# https://arxiv.org/pdf/2212.10554.pdf
# https://github.com/syncdoth/RetNet/blob/main/retnet/xpos_relative_position.py
from einops import rearrange

# [1,2]    [1,1,2,2]
# [3,4] -> [3,3,4,4]
# [5,6]    [5,5,6,6]
def duplicate_interleave(m):
    return m.view(-1, 1).repeat(1, 2).view(m.shape[0], -1)

# 0,1,2,3,4,5,6,7 -> -1,0,-3,2,-5,4,-7,6
def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


class XPosEmbedding(torch.nn.Module):
    """Implementation of xPos based on RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=10000,
        scale_base=512
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None
        self.scale_base = scale_base
        self.register_buffer("scale",
                             (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim))

    def cos_sin(
        self,
        seq_len: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = duplicate_interleave(freqs).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        scale = self.scale**torch.arange(seq_len).to(self.scale).div(self.scale_base)[:, None]
        scale = duplicate_interleave(scale).to(q.device)
        return (q * cos * scale) + (rotate_every_two(q) * sin * scale), (k * cos * (1 / scale)) + (rotate_every_two(k) * sin * (1 / scale))
