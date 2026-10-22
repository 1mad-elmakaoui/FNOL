"""Custom format checkers, semantic validators and normalizers.

Formats (JSON Schema ``format``): ``date-time``, ``date``, ``phone``,
``license-plate``, ``email``. Semantic validators are referenced from a schema
with ``"x-validators": [...]``; normalizers with ``"x-normalize": [...]``
(format-based normalization for phone numbers and plates is automatic).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from jsonschema import FormatChecker

# ------------------------------------------------------------------ plates
# French SIV (2009+): AB-123-CD. Letters I, O and U are never issued.
FR_PLATE = re.compile(r"^[A-HJ-NP-TV-Z]{2}-\d{3}-[A-HJ-NP-TV-Z]{2}$")
# Moroccan: up to 5 digits, one Arabic letter (often transliterated A, B, D, H, W...),
# then the 1-2 digit prefecture code. Written 12345-A-6 / 12345-أ-6.
MA_PLATE = re.compile(r"^\d{1,5}-(?:[A-Z]|[ء-ي])-\d{1,2}$")

_SEP = re.compile(r"[\s.\-_/|]+")


def normalize_plate(value: str) -> str:
    raw = value.strip().upper()
    compact = _SEP.sub("", raw)
    m = re.fullmatch(r"([A-Z]{2})(\d{3})([A-Z]{2})", compact)
    if m:
        return "-".join(m.groups())
    m = re.fullmatch(r"(\d{1,5})([A-Z]|[ء-ي])(\d{1,2})", compact)
    if m:
        return "-".join(m.groups())
    return _SEP.sub("-", raw).strip("-")


def is_plate(value: object) -> bool:
    if not isinstance(value, str):
        return True  # type errors are reported by the "type" keyword
    return bool(FR_PLATE.match(value) or MA_PLATE.match(value))


# ------------------------------------------------------------------ phones
_FR_PHONE = re.compile(r"^(?:0[1-9]\d{8}|\+33[1-9]\d{8})$")
_MA_PHONE = re.compile(r"^(?:0[5-8]\d{8}|\+212[5-8]\d{8})$")
_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


def normalize_phone(value: str) -> str:
    compact = re.sub(r"[\s.\-()/]", "", value.strip())
    if compact.startswith("00"):
        compact = "+" + compact[2:]
    # "+33 (0)6 12..." -> "+336 12..."
    compact = re.sub(r"^\+(33|212)0", r"+\1", compact)
    return compact


def is_phone(value: object) -> bool:
    if not isinstance(value, str):
        return True
    return bool(_FR_PHONE.match(value) or _MA_PHONE.match(value) or _E164.match(value))


# ------------------------------------------------------------------ dates
def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if "T" not in text and " " not in text:
        raise ValueError("date-time must include a time component (YYYY-MM-DDTHH:MM[:SS])")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def is_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return True
    try:
        parse_datetime(value)
        return True
    except ValueError:
        return False


def is_date(value: object) -> bool:
    if not isinstance(value, str):
        return True
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return False
    try:
        date.fromisoformat(value.strip())
        return True
    except ValueError:
        return False


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_email(value: object) -> bool:
    return not isinstance(value, str) or bool(_EMAIL.match(value))


def build_format_checker() -> FormatChecker:
    checker = FormatChecker(formats=())
    checker.checks("date-time")(is_datetime)
    checker.checks("date")(is_date)
    checker.checks("phone")(is_phone)
    checker.checks("license-plate")(is_plate)
    checker.checks("email")(is_email)
    return checker


FORMAT_CHECKER = build_format_checker()

FORMAT_HINTS = {
    "date-time": "expected ISO 8601 local date-time, e.g. 2026-09-13T17:40:00",
    "date": "expected ISO 8601 date, e.g. 1988-04-02",
    "phone": "expected a French (0612345678 / +33612345678), Moroccan (0661234567 / +212661234567) or E.164 number",
    "license-plate": "expected French SIV AB-123-CD (letters I, O, U are never used) or Moroccan 12345-A-6",
    "email": "expected name@domain.tld",
}


# ------------------------------------------------------------------ semantic validators
class ValidationContext:
    def __init__(self, reference_time: datetime | None = None, tz: str = "Europe/Paris"):
        self.tz = ZoneInfo(tz)
        self.reference_time = reference_time or datetime.now(timezone.utc)
        if self.reference_time.tzinfo is None:
            self.reference_time = self.reference_time.replace(tzinfo=self.tz)


FUTURE_SLACK = timedelta(minutes=5)


def not_in_future(value: Any, ctx: ValidationContext) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        if "T" in value or " " in value.strip():
            moment = parse_datetime(value)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=ctx.tz)
        else:
            moment = datetime.combine(date.fromisoformat(value.strip()), datetime.min.time(), tzinfo=ctx.tz)
    except ValueError:
        return []  # format errors are reported by the format checker
    limit = min(ctx.reference_time, datetime.now(timezone.utc)) + FUTURE_SLACK
    if moment > limit:
        return [f"{value} is in the future (reference time {ctx.reference_time.isoformat(timespec='minutes')})"]
    return []


SEMANTIC_VALIDATORS: dict[str, Callable[[Any, ValidationContext], list[str]]] = {
    "not_in_future": not_in_future,
}


# ------------------------------------------------------------------ normalizers
NORMALIZERS: dict[str, Callable[[str], str]] = {
    "strip": str.strip,
    "upper": str.upper,
    "lower": str.lower,
    "collapse_spaces": lambda s: re.sub(r"\s+", " ", s).strip(),
    "phone": normalize_phone,
    "license-plate": normalize_plate,
}


def normalize_value(value: Any, subschema: dict[str, Any]) -> Any:
    """Apply format-driven and x-normalize normalizers to string values (recursing into arrays/objects)."""
    if isinstance(value, str):
        steps = list(subschema.get("x-normalize", []))
        fmt = subschema.get("format")
        if fmt in NORMALIZERS:
            steps.append(fmt)
        for step in steps:
            value = NORMALIZERS[step](value)
        return value
    if isinstance(value, list) and isinstance(subschema.get("items"), dict):
        return [normalize_value(v, subschema["items"]) for v in value]
    if isinstance(value, dict) and isinstance(subschema.get("properties"), dict):
        props = subschema["properties"]
        return {k: normalize_value(v, props[k]) if k in props else v for k, v in value.items()}
    return value
