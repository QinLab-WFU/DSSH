import numpy as np
import torch


def ratio_of_bessel_approx(kappa, d):
    """
    Approximates the ratio of modified Bessel functions of the first kind.
    See Eq. 8 & Eq. 10 of paper
    Args:
      kappa: The value of the concentration parameter for a vMF distribution.
      d: Dimensionality of the embedding space.

    Returns:
      The approximation to the ratio of modified Bessel functions of first kind.
      The output shape is B x K
    """
    # NOTE: This is an approximation from https://arxiv.org/pdf/1606.02008.pdf
    kappa_squared = kappa**2

    d_m_half = (d / 2.0) - 0.5
    sqrt_d_m_half = torch.sqrt(d_m_half**2 + kappa_squared)

    d_p_half = (d / 2.0) + 0.5
    sqrt_d_p_half = torch.sqrt(d_p_half**2 + kappa_squared)

    return 0.5 * ((kappa / (d_m_half + sqrt_d_p_half)) + (kappa / (d_m_half + sqrt_d_m_half)))


def log_vmf_normalizer_approx(k_squared, d):
    """
    Approximates log C_d(kappa) from the vMF probability density function.
    See Eq. 12
    Args:
      k_squared: The value of the concentration parameter for a vMF distribution
        squared.
      d: Dimensionality of the embedding space.

    Returns:
      The approximation to log C_d(kappa).
    """
    d_m_half = (d / 2.0) - 0.5
    sqrt_d_m_half = torch.sqrt(d_m_half**2 + k_squared)

    d_p_half = (d / 2.0) + 0.5
    sqrt_d_p_half = torch.sqrt(d_p_half**2 + k_squared)

    return 0.5 * (
        d_m_half * torch.log(d_m_half + sqrt_d_m_half)
        - sqrt_d_m_half
        + d_m_half * torch.log(d_m_half + sqrt_d_p_half)
        - sqrt_d_p_half
    )


def get_norms_and_dim(x, use_torch):
    """Computes L2-norm and returns the dimensionality of the embedding space."""
    if use_torch:
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        dim = x.size(1)
    else:
        norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        dim = x.shape[1]
    return norms, dim


def get_norm_method_by_name(name):
    """Returns a norm function based on provided name."""
    if name == "l2":
        method = l2_norm
    elif name == "vmf":
        method = vmf_projection
    else:
        raise ValueError("Unknown normalization method: {}".format(name))
    return method


def l2_norm(x, use_torch=True, return_norms=False):
    """Computes L2 norm."""
    norms = get_norms_and_dim(x, use_torch)[0]
    x = x / norms

    if not return_norms:
        return x
    return x, norms


def vmf_projection(x, use_torch=True, return_norms=False):
    """Projects x via expectation of a vMF distribution."""
    norms, dim = get_norms_and_dim(x, use_torch)
    # L2-normalize the input (i.e., get mu of the vMF distribution)
    x = x / norms

    # NOTE: This is an approximation from https://arxiv.org/pdf/1606.02008.pdf
    # Scale mu by the ratio of modified bessel functions using our approximations
    # Here, the L2-norm represents kappa
    if use_torch:
        x = x * ratio_of_bessel_approx(norms, dim)
    else:
        x = x * ratio_of_bessel_approx(torch.from_numpy(norms), dim).numpy()

    if not return_norms:
        return x
    return x, norms
