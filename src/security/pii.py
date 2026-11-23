"""Find and redact personal data (PII) in text: regular expressions plus checksums.

WHAT IT CATCHES
    EMAIL   jane.doe@example.com
    CARD    4111 1111 1111 1111   (13-19 digits, and the Luhn checksum must pass)
    IBAN    GB82 WEST 1234 5698 7654 32   (country code + check digits; the mod-97 check must pass)
    SSN     123-45-6789   (US format; area 000, 666 and 900-999 are never issued, so skipped)
    PHONE   (555) 123-4567, 555-123-4567, +44 20 7946 0958
    IP      192.168.10.24

    >>> redact("Mail jane.doe@example.com or call (555) 123-4567; card 4111 1111 1111 1111.")
    ('Mail [EMAIL] or call [PHONE]; card [CARD].', {'EMAIL': 1, 'CARD': 1, 'PHONE': 1})
    >>> redact("I paid 1234 5678 9012 3456 in fees")[1]   # 16 digits, but the Luhn check fails
    {}

WHY CHECKSUMS
    Finance text is full of long numbers (amounts, account sizes, years). A pattern alone would redact
    "1234 5678 9012 3456" as a card; the Luhn check removes most of those false alarms. Same for IBANs.

WHAT IT MISSES (on purpose, see SECURITY.md)
    Names and street addresses need a named-entity model (e.g. Presidio + spaCy). Regexes stay instant,
    explainable and dependency-free; the doc lists the upgrade path.
"""
from __future__ import annotations

import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
PHONE = re.compile(r"(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)\s?|\b\d{3}[ .-])\d{3}[ .-]\d{4}\b"
                   r"|\+\d{1,3}[ .-]\d{2,4}(?:[ .-]\d{2,4}){2,3}\b")
IPV4 = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")


def luhn_ok(digits: str) -> bool:
    """The card-number checksum: double every second digit from the right, sum the digits, mod 10 == 0.

    >>> luhn_ok("4111111111111111"), luhn_ok("4111111111111112")
    (True, False)
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(s: str) -> bool:
    """IBAN check: move the first 4 chars to the end, letters -> numbers (A=10..Z=35), mod 97 == 1.

    >>> iban_ok("GB82WEST12345698765432"), iban_ok("GB82WEST12345698765433")
    (True, False)
    """
    s = s.replace(" ", "")
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


# (label, pattern, extra check on the match). Order matters: an IBAN or a card must be claimed before the
# phone pattern sees pieces of it, and emails before anything else.
DETECTORS = [
    ("EMAIL", EMAIL, None),
    ("IBAN", IBAN, iban_ok),
    ("CARD", CARD, lambda m: luhn_ok(re.sub(r"\D", "", m))),
    ("SSN", SSN, None),
    ("PHONE", PHONE, None),
    ("IP", IPV4, None),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace every detected item with [LABEL]. Returns (new text, {label: count})."""
    counts: dict[str, int] = {}
    for label, pattern, check in DETECTORS:
        def sub(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"
        text = pattern.sub(sub, text)
    return text, counts


def find(text: str) -> dict[str, int]:
    """Counts only (what redact() would replace)."""
    return redact(text)[1]
