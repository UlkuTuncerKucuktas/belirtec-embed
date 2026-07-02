from belirtec.json_utils import parse_json, clean


def test_object():
    assert parse_json('{"query": "soru"}') == {"query": "soru"}


def test_array():
    assert parse_json('[{"a":1}]') == [{"a": 1}]


def test_fenced():
    assert parse_json('```json\n{"query":"x"}\n```') == {"query": "x"}


def test_preamble():
    assert parse_json('İşte cevap:\n{"query":"x"}') == {"query": "x"}


def test_truncated_returns_non_list():
    # truncated array falls through to first object -> a dict, which callers reject
    assert not isinstance(parse_json('[{"a":1},{"b":'), list)


def test_clean():
    assert clean("  a\n b  ") == "a b"
