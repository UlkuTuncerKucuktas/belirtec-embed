
from belirtec.contamination import _ESAS, _KARAR, _norm


def test_case_number_extraction():
    txt = "Hukuk Genel Kurulu 2012/497 E. , 2013/150 K. MECURUN"
    assert _ESAS.findall(txt) == ["2012/497"]
    assert _KARAR.findall(txt) == ["2013/150"]


def test_norm_strips_nbsp():
    assert _norm("a\xa0\xa0b   c") == "a b c"
