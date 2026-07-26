import torch as th
from typing import Sequence, Tuple


# Canonical Basis
# ------------------------------------------------------------------------------------------

def canonical_basis(shape: Tuple[int, int, int], x: int, y: int) -> th.Tensor:
    """
    Generate a canonical basis vector.
    """

    vec = th.zeros((1, *shape))
    vec[..., y, x] = 1
    return vec[0] / vec.norm()


# DCT, DST Bases
# ------------------------------------------------------------------------------------------

def _dct_dst_base(x: th.Tensor, axis: int, sign: int) -> th.Tensor:
    """
    Base function to compute the DCT-II or DST-II of a tensor via FFT.

    Args:
        x: input data to be transformed.
        axis: axis along which to compute the transform.
        sign: 1 for DCT-II, -1 for DST-II.

    Returns:
        The transformed tensor along the specified axis.
    """
    N = x.shape[axis]
    exp_vec = 2 * th.exp(-1j * th.pi * th.arange(N, dtype=x.dtype, device=x.device) / (2 * N))

    x_perm = x.moveaxis(axis, -1)
    v = th.cat([x_perm[..., ::2], sign * th.flip(x_perm[..., 1::2], dims=(-1,))], dim=-1)
    V = th.fft.fft(v)
    result = (V * exp_vec).real

    if sign == -1:
        result = th.flip(result, dims=(-1,))

    return result.moveaxis(-1, axis)


def dctII(x: th.Tensor, axis: int = -1) -> th.Tensor:
    """
    Computes the Discrete Cosine Transform (DCT) Type-II of a tensor.

    Args:
        x: input data to be transformed.
        axis: axis along which to compute the transform.

    Returns:
        The transformed tensor along the specified axis.
    """
    return _dct_dst_base(x, axis, sign=1)


def dstII(x: th.Tensor, axis: int = -1) -> th.Tensor:
    """
    Computes the Discrete Sine Transform (DST) Type-II of a tensor.

    Args:
        x: input data to be transformed.
        axis: axis along which to compute the transform.

    Returns:
        The transformed tensor along the specified axis.
    """
    return _dct_dst_base(x, axis, sign=-1)


def idctII(x: th.Tensor, axis: int = -1) -> th.Tensor:
    """
    Computes the inverse Discrete Cosine Transform (IDCT) Type-II of a tensor.

    Args:
        x: input data to be transformed.
        axis: axis along which to compute the transform.

    Returns:
        The transformed tensor along the specified axis.
    """
    N = x.shape[axis]
    exp_vec = th.exp(1j * th.pi * th.arange(N, dtype=x.dtype, device=x.device) / (2 * N))

    x_perm = x.moveaxis(axis, -1)
    x_rev = th.flip(x_perm, dims=(-1,))[..., :-1]
    v = th.zeros_like(x_perm, dtype=th.complex128)
    v[..., 0] = x_perm[..., 0]
    v[..., 1:N] = exp_vec[1:N] * (x_perm[..., 1:N] - 1j * x_rev)
    v = v / 2

    V = th.fft.ifft(v)
    y = th.zeros_like(x_perm)
    y[..., ::2] = V[..., :N - N // 2].real
    y[..., 1::2] = th.flip(V, dims=(-1,))[..., :N // 2].real
    return y.moveaxis(-1, axis)


def idstII(x: th.Tensor, axis: int = -1) -> th.Tensor:
    """
    Computes the inverse Discrete Sine Transform (IDST) Type-II of a tensor.

    Args:
        x: input data to be transformed.
        axis: axis along which to compute the transform.

    Returns:
        The transformed tensor along the specified axis.
    """
    N = x.shape[axis]
    x_flip = th.flip(x, dims=(axis,))
    idct_x_flip = idctII(x_flip, axis=axis)

    factors = (-1) ** th.arange(N, dtype=idct_x_flip.dtype, device=x.device)
    shape = [1] * x.ndim
    shape[axis] = N
    factors = factors.view(shape)

    return idct_x_flip * factors


def dct_basis(shape: Tuple[int, int, int],
              fx: int, fy: int) -> th.Tensor:
    """
    Generate a DCT basis vector.
    """

    vec = th.zeros((1, *shape))

    vec_dct = dctII(dctII(vec, axis=-2), axis=-1)
    vec_dct[..., fy, fx] = 1
    vec = idctII(idctII(vec_dct, axis=-1), axis=-2)[0]

    return vec / vec.norm()


def dst_basis(shape: Tuple[int, int, int],
              fx: int, fy: int) -> th.Tensor:
    """
    Generate a DST basis vector.
    """

    vec = th.zeros((1, *shape))

    vec_dst = dstII(dstII(vec, axis=-2), axis=-1)
    vec_dst[..., fy, fx] = 1
    vec = idstII(idstII(vec_dst, axis=-1), axis=-2)[0]

    return vec / vec.norm()


# Hadamard Basis
# ------------------------------------------------------------------------------------------

def hadamard(x: th.Tensor, axis: int = -1) -> th.Tensor:
    """
    Computes the (ordered) Hadamard transform of a tensor along a specified axis.

    Args:
        x: input data to be transformed.
        axis: axis along which to compute the transform.

    Returns:
        The transformed tensor along the specified axis.
    """

    N = x.shape[axis]
    n_bits = N.bit_length() - 1

    x_perm = x.moveaxis(axis, -1)
    y = x_perm.clone()

    block_size = 2
    while block_size <= N:
        half = block_size // 2
        new_shape = list(y.shape)[:-1] + [-1, block_size]
        y_view = y.view(*new_shape)
        a = y_view[..., :half]
        b = y_view[..., half:]
        yy = th.cat([a + b, a - b], dim=-1)
        y = yy.view(y.shape)
        block_size *= 2

    k = th.arange(N, device=x.device, dtype=th.long)
    gray = k ^ (k >> 1)
    rev = th.zeros_like(gray)
    for i in range(n_bits):
        rev |= ((gray >> i) & 1) << (n_bits - 1 - i)

    result = y[..., rev]
    return result.moveaxis(-1, axis).contiguous()


def hadamard_basis(shape: Tuple[int, int, int],
                   sx: int, sy: int) -> th.Tensor:
    """
    Generate a Hadamard basis vector.
    """

    vec = th.zeros((1, *shape))

    vec_dst = hadamard(hadamard(vec, axis=-2), axis=-1)
    vec_dst[..., sy, sx] = 1
    vec = hadamard(hadamard(vec_dst, axis=-1), axis=-2)[0]

    return vec / vec.norm()


# Haar Basis
# ------------------------------------------------------------------------------------------

def lwt2d(inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Lazy wavelet transform.
    (N, C, H, W) -> (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2)
    """

    ecer = inpt[..., ::2, ::2]
    ecor = inpt[..., 1::2, ::2]
    ocer = inpt[..., ::2, 1::2]
    ocor = inpt[..., 1::2, 1::2]
    return ecer, th.cat([ecor, ocer, ocor], dim=1)


def ilwt2d(coarse: th.Tensor, details: th.Tensor) -> th.Tensor:
    """
    Inverse lazy wavelet transform.
    (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2) -> (N, C, H, W)
    """

    ecer, [ecor, ocer, ocor] = coarse, details.chunk(3, dim=1)
    x = th.empty(coarse.shape[0],
                 coarse.shape[1],
                 2 * coarse.shape[2],
                 2 * coarse.shape[3],
                 device=coarse.device)
    x[..., ::2, ::2] = ecer
    x[..., 1::2, ::2] = ecor
    x[..., ::2, 1::2] = ocer
    x[..., 1::2, 1::2] = ocor
    return x


def haar2d(inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Haar transform.
    (N, C, H, W) -> (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2)
    """

    top_left, details = lwt2d(inpt)
    bottom_left, top_right, bottom_right = details.chunk(3, dim=1)
    coarse = (top_left + bottom_left + top_right + bottom_right) / 4
    return coarse, th.cat(((top_left + top_right - bottom_left - bottom_right) / 4,
                           (top_left + bottom_left - top_right - bottom_right) / 4,
                           (top_left + bottom_right - top_right - bottom_left) / 4), dim=1)


def ihaar2d(coarse: th.Tensor, details: th.Tensor) -> th.Tensor:
    """
    Inverse haar transform.
    (N, C, H / 2, W / 2) + (N, 3 * C, H / 2, W / 2) -> (N, C, H, W)
    """

    d1, d2, d3 = details.chunk(3, dim=1)
    return ilwt2d(coarse + d1 + d2 + d3,
                  th.cat((coarse - d1 + d2 - d3,
                          coarse + d1 - d2 - d3,
                          coarse - d1 - d2 + d3), dim=1))


def full_haar2d(inpt: th.Tensor) -> Sequence[th.Tensor]:
    """
    Full haar transform.
    (N, C, H, W) -> [(N, 3 * C, H / 2, W / 2), ..., (N, 3 * C, 1, 1)]
    """

    out = []

    while inpt.shape[-1] > 1:
        inpt, details = haar2d(inpt)
        out.append(details)
    out.append(inpt)
    return out


def full_ihaar2d(inpt: Sequence[th.Tensor]) -> th.Tensor:
    """
    Inverse full haar transform.
    [(N, 3 * C, H / 2, W / 2), ..., (N, 3 * C, 1, 1)] -> (N, C, H, W)
    """

    coarse = inpt.pop()
    for details in reversed(inpt):
        coarse = ihaar2d(coarse, details)
    return coarse


def haar_basis(shape: Tuple[int, int, int],
               scale: int, c: int, x: int, y: int) -> th.Tensor:
    """
    Generate a haar basis vector.
    """

    vec = th.zeros((1, *shape))

    vec = full_haar2d(vec)
    vec[scale][0, c, y, x] = 1
    vec = full_ihaar2d(vec)[0]

    return vec / vec.norm()
