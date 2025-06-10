# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
from functools import partial
from typing import Callable

import jax
import jax.nn
import jax.numpy as jnp


def prope_dot_product_attention(
    q: jax.Array,  # (batch, seqlen, num_heads, head_dim)
    k: jax.Array,  # (batch, seqlen, num_heads, head_dim)
    v: jax.Array,  # (batch, seqlen, num_heads, head_dim)
    *,
    viewmats: jax.Array,  # (batch, cameras, 4, 4)
    Ks: jax.Array,  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    **kwargs,
) -> jax.Array:
    """Similar to jax.nn.dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """

    # We're going to assume self-attention: all inputs are the same shape.
    (batch, seqlen, num_heads, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    # Normalize camera intrinsics.
    Ks_norm = jnp.zeros_like(Ks)
    Ks_norm = Ks_norm.at[..., 0, 0].set(Ks[..., 0, 0] / image_width)
    Ks_norm = Ks_norm.at[..., 1, 1].set(Ks[..., 1, 1] / image_height)
    Ks_norm = Ks_norm.at[..., 0, 2].set(Ks[..., 0, 2] / image_width - 0.5)
    Ks_norm = Ks_norm.at[..., 1, 2].set(Ks[..., 1, 2] / image_height - 0.5)
    Ks_norm = Ks_norm.at[..., 2, 2].set(1.0)
    del Ks

    # Compute the camera projection matrices we use in PRoPE.
    # - K is an `image<-camera` transform.
    # - viewmats is a `camera<-world` transform.
    # - P = lift(K) @ viewmats is an `image<-world` transform.
    P = jnp.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
    P_T = P.swapaxes(-1, -2)
    P_inv = jnp.einsum(
        "...ij,...jk->...ik",
        _invert_SE3(viewmats),
        _lift_K(_invert_K(Ks_norm)),
    )
    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting, XLA should optimize these away.
    coeffs_x = _rope_precompute_coeffs(
        jnp.tile(jnp.arange(patches_x), patches_y * cameras),
        freq_base=100.0,
        freq_scale=1.0,
        feat_dim=head_dim // 4,
    )
    coeffs_y = _rope_precompute_coeffs(
        jnp.tile(jnp.repeat(jnp.arange(patches_y), patches_x), cameras),
        freq_base=100.0,
        freq_scale=1.0,
        feat_dim=head_dim // 4,
    )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, projmat=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, projmat=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, projmat=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]
    out = jax.nn.dot_product_attention(
        query=_apply_block_diagonal(q, transforms_q),
        key=_apply_block_diagonal(k, transforms_kv),
        value=_apply_block_diagonal(v, transforms_kv),
        **kwargs,
    )
    out = _apply_block_diagonal(out, transforms_o)
    assert out.shape == (batch, seqlen, num_heads, head_dim)
    return out


def _apply_tiled_projmat(
    feats: jax.Array,  # (batch, seqlen, num_heads, feat_dim)
    projmat: jax.Array,  # (batch, cameras, 4, 4)
) -> jax.Array:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, seqlen, num_heads, feat_dim) = feats.shape
    cameras = projmat.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    assert projmat.shape == (batch, cameras, 4, 4)
    return jnp.einsum(
        "bcij,bcpnkj->bcpnki",
        projmat,
        feats.reshape((batch, cameras, -1, num_heads, feat_dim // 4, 4)),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: jax.Array,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (-jnp.arange(num_freqs)[None, None, None, :] / num_freqs)
    )
    angles = positions[None, :, None, None] * freqs
    # Shape should be: `(batch, seqlen, num_heads, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (1, positions.shape[0], 1, num_freqs)
    return jnp.cos(angles), jnp.sin(angles)


def _rope_apply_coeffs(
    feats: jax.Array,  # (batch, seqlen, num_heads, feat_dim)
    coeffs: tuple[jax.Array, jax.Array],
    inverse: bool = False,
) -> jax.Array:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return jnp.concatenate(
        [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
        if not inverse
        else [cos * x_in - sin * y_in, sin * x_in + cos * y_in],
        axis=-1,
    )


def _apply_block_diagonal(
    feats: jax.Array,  # (..., dim)
    func_size_pairs: list[tuple[Callable[[jax.Array], jax.Array], int]],
) -> jax.Array:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Array) -> Array, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = jnp.split(feats, tuple(itertools.accumulate(block_sizes[:-1])), axis=-1)
    out = jnp.concatenate(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        axis=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: jax.Array) -> jax.Array:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].swapaxes(-1, -2)
    out = jnp.zeros_like(transforms)
    out = out.at[..., :3, :3].set(Rinv)
    out = out.at[..., :3, 3].set(
        -jnp.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    )
    out = out.at[..., 3, 3].set(1.0)
    return out


def _lift_K(Ks: jax.Array) -> jax.Array:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = jnp.zeros(Ks.shape[:-2] + (4, 4))
    out = out.at[..., :3, :3].set(Ks)
    out = out.at[..., 3, 3].set(1.0)
    return out


def _invert_K(Ks: jax.Array) -> jax.Array:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = jnp.zeros(Ks.shape)
    out = out.at[..., 0, 0].set(1.0 / Ks[..., 0, 0])
    out = out.at[..., 1, 1].set(1.0 / Ks[..., 1, 1])
    out = out.at[..., 0, 2].set(-Ks[..., 0, 2] / Ks[..., 0, 0])
    out = out.at[..., 1, 2].set(-Ks[..., 1, 2] / Ks[..., 1, 1])
    out = out.at[..., 2, 2].set(1.0)
    return out
