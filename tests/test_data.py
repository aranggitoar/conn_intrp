from conn_intrp.data import relaxed_anls, best_anls


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
