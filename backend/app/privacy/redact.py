"""PII redaction for text that leaves the machine.

Only transcript *text* is ever sent to the Anthropic API (never audio). Before
it goes out, categories of personal data that no form field needs are masked:
payment card numbers (Luhn-checked), IBANs, French social-security numbers
(NIR), Moroccan national ID numbers (CIN) and e-mail addresses.

A form schema can opt a category back in with ``"x-pii-allow": ["email"]``
(the KYC schema does this because it collects an e-mail address). The raw
transcript stays in the local Postgres; redaction only affects the outbound
copy, so audit and human review keep the full record.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class _Rule:
    category: str
    pattern: re.Pattern[str]
    check: object = None  # optional callable(match_text) -> bool


RULES: tuple[_Rule, ...] = (
    _Rule("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    _Rule("iban", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    _Rule(
        "card",
        re.compile(r"\b[2-6](?:[ -]?\d){12,18}\b"),  # IINs start with 2-6; phone numbers with 0/00 never match
        lambda m: _luhn_ok(re.sub(r"\D", "", m)),
    ),
    _Rule(
        "fr_nir",
        re.compile(r"\b[12] ?\d{2} ?(?:0[1-9]|1[0-2]) ?(?:2[AB]|\d{2}) ?\d{3} ?\d{3}(?: ?\d{2})?\b"),
    ),
    _Rule("ma_cin", re.compile(r"\b[A-Z]{1,2}\d{5,6}\b")),
)

CATEGORIES = tuple(r.category for r in RULES)


class Redactor:
    def __init__(self, enabled: bool = True, allow: Iterable[str] = ()):
        self.enabled = enabled
        self.allow = set(allow)
        self.count = 0

    def __call__(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for rule in RULES:
            if rule.category in self.allow:
                continue

            def _sub(m: re.Match[str], rule: _Rule = rule) -> str:
                if rule.check is not None and not rule.check(m.group(0)):  # type: ignore[operator]
                    return m.group(0)
                self.count += 1
                return f"[REDACTED:{rule.category.upper()}]"

            text = rule.pattern.sub(_sub, text)
        return text
