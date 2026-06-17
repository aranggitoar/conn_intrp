import pytest
from conn_intrp.data import relaxed_anls, best_anls, filter_categories


def test_relaxed_anls_exact_match():
    assert relaxed_anls("hello", "hello") == 1.0


def test_relaxed_anls_substring():
    assert relaxed_anls("the answer is hello", "hello") == 1.0


def test_relaxed_anls_case_insensitive():
    assert relaxed_anls("Hello", "hello") == 1.0


def test_relaxed_anls_distant_strings():
    assert relaxed_anls("abc", "xyz") == 0.0


def test_relaxed_anls_close_strings():
    score = relaxed_anls("helo", "hello")
    assert 0.0 < score < 1.0


def test_relaxed_anls_strips_whitespace():
    assert relaxed_anls("  hello  ", "hello") == 1.0


def test_best_anls_picks_max():
    assert best_anls("hello", ["xyz", "hello", "abc"]) == 1.0


def test_best_anls_partial():
    score = best_anls("helo", ["xyz", "hello"])
    assert score == relaxed_anls("helo", "hello")


def test_filter_categories_select():
    data = {"a": [1, 2, 3], "b": [4, 5], "c": [6]}
    out = filter_categories(data, categories=["a", "c"])
    assert set(out.keys()) == {"a", "c"}
    assert out["a"] == [1, 2, 3]


def test_filter_categories_max_samples():
    data = {"a": [1, 2, 3, 4, 5], "b": [6, 7]}
    out = filter_categories(data, max_samples=2)
    assert out["a"] == [1, 2]
    assert out["b"] == [6, 7]


def test_filter_categories_both():
    data = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7, 8]}
    out = filter_categories(data, categories=["a"], max_samples=2)
    assert set(out.keys()) == {"a"}
    assert out["a"] == [1, 2]


def test_filter_categories_unknown_raises():
    data = {"a": [1]}
    with pytest.raises(KeyError):
        filter_categories(data, categories=["z"])


def test_filter_categories_no_mutation():
    data = {"a": [1, 2, 3]}
    filter_categories(data, max_samples=1)
    assert data["a"] == [1, 2, 3]
