from datetime import datetime, timedelta, timezone

import pytest

from app.agent.schema import SchemaNotFound, load_schema
from app.agent.validators import (
    ValidationContext,
    is_phone,
    is_plate,
    normalize_phone,
    normalize_plate,
    not_in_future,
)

CTX = ValidationContext(reference_time=datetime(2026, 9, 14, 9, 12, tzinfo=timezone(timedelta(hours=2))))


@pytest.fixture(scope="module")
def fnol():
    return load_schema("fnol")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ab-123-cd", "AB-123-CD"),
        ("AB 123 CD", "AB-123-CD"),
        ("ab123cd", "AB-123-CD"),
        ("1234 b 56", "1234-B-56"),
        ("12345 | A | 6", "12345-A-6"),
        ("48213-a-1", "48213-A-1"),
    ],
)
def test_plate_normalization(raw, expected):
    assert normalize_plate(raw) == expected
    assert is_plate(expected)


@pytest.mark.parametrize("bad", ["AB-123-OU", "A-123-BC", "ABC-12-DE", "123456-A-1", "1234-AB-56", "hello"])
def test_plate_rejects(bad):
    assert not is_plate(normalize_plate(bad))


def test_moroccan_plate_with_arabic_letter():
    assert is_plate("12345-ب-6")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("06 12 45 78 90", "0612457890"),
        ("06.12.45.78.90", "0612457890"),
        ("+33 6 12 45 78 90", "+33612457890"),
        ("+33 (0)6 12 45 78 90", "+33612457890"),
        ("0033 6 12 45 78 90", "+33612457890"),
        ("+212 6 62 48 19 07", "+212662481907"),
        ("0661 23 45 67", "0661234567"),
    ],
)
def test_phone_normalization(raw, expected):
    assert normalize_phone(raw) == expected
    assert is_phone(expected)


@pytest.mark.parametrize("bad", ["12345", "0012", "+33 06 12", "phone", "06 12 45 78 9", "+1 23"])
def test_phone_rejects(bad):
    assert not is_phone(normalize_phone(bad))


def test_datetime_not_in_future():
    assert not_in_future("2026-09-13T17:40:00", CTX) == []
    errors = not_in_future("2026-09-15T10:00:00", CTX)
    assert errors and "future" in errors[0]
    # Relative to the call, not only to "now": after the call is still the future.
    assert not_in_future("2026-09-14T11:00:00", CTX)


def test_field_validation_normalizes(fnol):
    res = fnol.validate_field("vehicle_plate", "ge 418 tr", CTX)
    assert res.valid and res.value == "GE-418-TR"
    assert res.as_dict("ge 418 tr")["normalized_value"] == "GE-418-TR"
    res = fnol.validate_field("policy_number", " fnx-204518 ", CTX)
    assert res.valid and res.value == "FNX-204518"


def test_field_validation_errors(fnol):
    res = fnol.validate_field("incident_datetime", "yesterday at 5", CTX)
    assert not res.valid and "date-time" in res.errors[0]
    res = fnol.validate_field("incident_datetime", "2026-09-20T10:00:00", CTX)
    assert not res.valid and "future" in res.errors[0]
    res = fnol.validate_field("injuries_reported", "no", CTX)
    assert not res.valid and "boolean" in res.errors[0]
    res = fnol.validate_field("callback_number", "call me maybe", CTX)
    assert not res.valid and "phone" in res.errors[0]
    res = fnol.validate_field("witnesses", [{"name": "Julie", "phone": "06 87 45 12 36"}], CTX)
    assert res.valid and res.value == [{"name": "Julie", "phone": "0687451236"}]
    res = fnol.validate_field("witnesses", [{"phone": "0687451236"}], CTX)
    assert not res.valid


def test_unknown_field(fnol):
    with pytest.raises(KeyError):
        fnol.validate_field("colour", "red", CTX)


def test_conditional_requirements(fnol):
    base = {
        "policy_number": "FNX-204518",
        "policyholder_name": "Daniel Morgan",
        "incident_datetime": "2026-09-13T17:40:00",
        "incident_location": "Lyon",
        "vehicle_plate": "GE-418-TR",
        "damage_description": "Rear bumper",
        "injuries_reported": False,
        "third_party_involved": False,
        "police_report_filed": False,
        "callback_number": "0612457890",
    }
    assert fnol.missing_required(base) == []
    assert fnol.missing_required({**base, "injuries_reported": True}) == ["injury_details"]
    assert fnol.missing_required({**base, "police_report_filed": True}) == ["police_report_number"]
    assert fnol.missing_required({**base, "third_party_involved": True}) == ["third_party_name", "third_party_plate"]
    partial = {k: v for k, v in base.items() if k != "policy_number"}
    assert fnol.missing_required(partial) == ["policy_number"]


def test_schemas_are_swappable():
    kyc = load_schema("kyc")
    assert "address.city" in kyc.by_path
    assert kyc.pii_allow == ["email"]
    assert "address.city" in kyc.missing_required({"address": {"street": "1 rue X"}})
    with pytest.raises(SchemaNotFound):
        load_schema("does-not-exist")
    with pytest.raises(SchemaNotFound):
        load_schema("../etc/passwd")
