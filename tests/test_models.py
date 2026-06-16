import torch

from conn_intrp.models.base import SVDLayer


def test_svd_layer_fields():
    layer = SVDLayer(
        name="proj",
        U=torch.eye(3),
        S=torch.ones(3),
        Vt=torch.eye(3),
        bias=torch.zeros(3),
        n_dirs=3,
    )
    assert layer.name == "proj"
    assert layer.n_dirs == 3
    assert layer.S.shape == (3,)
