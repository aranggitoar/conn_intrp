"""
Centered Kernel Alignment (CKA) for comparing representations.

Computes linear CKA between two representation matrices to measure
how similarly two layers encode the same set of inputs.  Used here
to quantify the restructuring performed by the cross-modal connector
(vision encoder output vs connector output).

Example::

    >>> from conn_intrp.cka import linear_cka
    >>> vision_out = adapter.extract_vision(inputs)       # (B, P, D1)
    >>> conn_out = adapter.run_connector(vision_out)      # (B, P, D2)
    >>> cka = linear_cka(vision_out.flatten(0, 1), conn_out.flatten(0, 1))

Main Functions:
    linear_cka: Linear CKA similarity between two representation matrices.
"""

import torch


def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """
    Compute linear CKA between representations *X* and *Y*.

    Both inputs are ``(N, D)`` matrices where *N* is the number of
    examples (must match) and *D* is the feature dimension (may differ).
    Representations are mean-centred before computing CKA.

    :param X: First representation matrix, shape ``(N, D1)``.
    :type X: torch.Tensor
    :param Y: Second representation matrix, shape ``(N, D2)``.
    :type Y: torch.Tensor
    :returns: CKA similarity in ``[0, 1]``.
    :rtype: float
    """
    X = X.float()
    Y = Y.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    gram_xy = Y.T @ X
    hsic_xy = (gram_xy * gram_xy).sum()
    gram_xx = X.T @ X
    gram_yy = Y.T @ Y
    hsic_xx = (gram_xx * gram_xx).sum()
    hsic_yy = (gram_yy * gram_yy).sum()
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()
