from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import MultiheadAttention


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class MultiheadAttention(torch.nn.MultiheadAttention):
    """Wrapper around torch.nn.MultiheadAttention to support customized scaled dot-product attention.

    The __init__ method accepts an additional argument `sdpa_fn` which defines how the
    scaled dot-product attention is computed. If `sdpa_fn` is not provided, the default
    behavior of torch.nn.MultiheadAttention is used (i.e., F.scaled_dot_product_attention).

    The `sdpa_fn` function should have the following signature:
    ```
    def sdpa_fn(q, k, v, **kwargs) -> Tensor:
    ```
    """

    def __init__(self, *args, **kwargs):
        self.qk_norm = kwargs.pop("qk_norm", False)
        super().__init__(*args, **kwargs)
        if self.qk_norm:
            d_model, nhead = args[0], args[1]
            self.q_norm = RMSNorm(d_model // nhead)
            self.k_norm = RMSNorm(d_model // nhead)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        sdpa_fn: Optional[Callable] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if sdpa_fn is None:
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            # manually compute mha with customized sdpa_fn
            assert self.bias_k is None and self.bias_v is None
            assert not self.add_zero_attn
            assert not need_weights

            if not self.batch_first:
                query, key, value = (x.permute(1, 0, 2) for x in (query, key, value))

            if self._qkv_same_embed_dim:
                q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(
                    3, dim=0
                )
            else:
                q_proj_weight = self.q_proj_weight
                k_proj_weight = self.k_proj_weight
                v_proj_weight = self.v_proj_weight

            if self.in_proj_bias is not None:
                q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(
                    3, dim=0
                )
            else:
                q_proj_bias = k_proj_bias = v_proj_bias = None

            q = F.linear(query, q_proj_weight, q_proj_bias)
            k = F.linear(key, k_proj_weight, k_proj_bias)
            v = F.linear(value, v_proj_weight, v_proj_bias)
            q, k, v = (
                rearrange(x, "b t (h c) -> b t h c", h=self.num_heads)
                for x in [q, k, v]
            )
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q, k, v = (rearrange(x, "b t h c -> b h t c") for x in [q, k, v])
            o = sdpa_fn(q, k, v, dropout_p=self.dropout)
            o = rearrange(o, "b h t c -> b t (h c)", h=self.num_heads)
            attn_output = F.linear(o, self.out_proj.weight, self.out_proj.bias)

            return attn_output, None


if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda:0"
    mha = MultiheadAttention(embed_dim=128, num_heads=8).to(device)

    q = torch.randn(10, 32, 128).to(device)
    k = torch.randn(20, 32, 128).to(device)
    v = torch.randn(20, 32, 128).to(device)

    output, _ = mha(q, k, v, sdpa_fn=lambda q, k, v, **kwargs: v, need_weights=False)
    print(output.size(), output.sum())
