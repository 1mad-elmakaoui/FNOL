import pytest

from app.privacy.redact import Redactor


@pytest.mark.parametrize(
    "text,category",
    [
        ("card 4111 1111 1111 1111 thanks", "CARD"),
        ("IBAN FR76 3000 6000 0112 3456 7890 189", "IBAN"),
        ("my social security number is 1 85 05 78 006 084 36", "FR_NIR"),
        ("CIN BK123456", "MA_CIN"),
        ("mail me at a.b+c@example.co.uk", "EMAIL"),
    ],
)
def test_redacts_categories(text, category):
    out = Redactor()(text)
    assert f"[REDACTED:{category}]" in out


@pytest.mark.parametrize(
    "text",
    [
        "call me on 06 12 45 78 90",
        "+212 6 62 48 19 07",
        "policy FNX-204518, plate GE-418-TR, report PV-2026-091344",
        "plate 1234-B-56 and 48213-A-1",
        "card-like but not Luhn 4111 1111 1111 1112",
    ],
)
def test_keeps_form_data(text):
    assert Redactor()(text) == text


def test_allow_list_and_disable():
    assert Redactor(allow=["email"])("x@y.fr") == "x@y.fr"
    assert Redactor(enabled=False)("4111 1111 1111 1111") == "4111 1111 1111 1111"
    r = Redactor()
    r("a@b.cd and 4111111111111111")
    assert r.count == 2
