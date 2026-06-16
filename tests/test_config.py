from dataclasses import asdict

from conn_intrp.config import DirectionalMaskingConfig


def test_config_fields():
    config = DirectionalMaskingConfig(
        category="table/list",
        model="smolvlm2",
        component="proj",
        optimizer="SGD",
        sparsity_coef=1.5e-3,
        lr=0.1,
        epochs=3,
        step=5,
        kl_per_epoch=[0.01, 0.005, 0.003],
        l1_per_epoch=[0.5, 0.4, 0.3],
        below_half_per_epoch=[120, 150, 180],
        near_zero_per_epoch=[80, 100, 120],
    )
    assert config.category == "table/list"
    assert config.model == "smolvlm2"
    assert config.epochs == 3


def test_config_asdict():
    config = DirectionalMaskingConfig(
        category="heading",
        model="internvl",
        component="linear_2",
        optimizer="SGD",
        sparsity_coef=1e-3,
        lr=1.0,
        epochs=1,
        step=1,
        kl_per_epoch=[0.02],
        l1_per_epoch=[0.6],
        below_half_per_epoch=[50],
        near_zero_per_epoch=[10],
    )
    d = asdict(config)
    assert d["category"] == "heading"
    assert d["component"] == "linear_2"
    assert isinstance(d["kl_per_epoch"], list)
