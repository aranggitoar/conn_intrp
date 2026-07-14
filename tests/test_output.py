import json
from pathlib import Path

import numpy as np
import torch

from conn_intrp.output import _make_serializable, make_run_dir, save_json


def test_make_serializable_primitives():
    assert _make_serializable(42) == 42
    assert _make_serializable("hello") == "hello"
    assert _make_serializable(3.14) == 3.14


def test_make_serializable_tensor():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = _make_serializable(t)
    assert result == [1.0, 2.0, 3.0]


def test_make_serializable_ndarray():
    a = np.array([1, 2, 3])
    result = _make_serializable(a)
    assert result == [1, 2, 3]


def test_make_serializable_numpy_scalar():
    assert _make_serializable(np.int64(5)) == 5
    assert _make_serializable(np.float32(3.14)) == np.float32(3.14).item()


def test_make_serializable_path():
    assert _make_serializable(Path("/tmp/test")) == "/tmp/test"


def test_make_serializable_nested():
    data = {"tensor": torch.tensor([1.0]), "list": [Path("/a"), np.int64(2)]}
    result = _make_serializable(data)
    assert result == {"tensor": [1.0], "list": ["/a", 2]}


def test_make_run_dir(tmp_path):
    run_dir = make_run_dir(tmp_path, "smolvlm2", "dm")
    assert run_dir.exists()
    assert "smolvlm2_docvqa_dm_" in run_dir.name


def test_make_run_dir_with_tag(tmp_path):
    run_dir = make_run_dir(tmp_path, "smolvlm2", "dm", tag="test")
    assert run_dir.name.endswith("_test")


def test_save_json(tmp_path):
    data = {"count": 5, "values": torch.tensor([1.0, 2.0])}
    path = tmp_path / "test.json"
    save_json(path, data)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == {"count": 5, "values": [1.0, 2.0]}
