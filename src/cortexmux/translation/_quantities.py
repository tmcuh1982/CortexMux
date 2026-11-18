"""Quantity checks for Chinese and Japanese technical text."""

import re
import unicodedata
from collections import Counter

from ._measures import consume

# Each fact binds the magnitude to its unit/subject and, for durations, its bound.
RULES = [
    ("axes=6", r"6\s*軸", r"\b(?:6|six) axes\b"),
    (
        "warehouses=100",
        r"(?<![数几幾數千十百一二三四五六七八九约約近超])百家(?=即时零售仓)",
        r"\bcent entrepôts\b",
    ),
    (
        "warehouses=hundreds",
        r"(?:数|數|几|幾)百家(?=即时零售仓)",
        r"\b(?:des|plusieurs) centaines d'entrepôts\b",
    ),
    (
        "coverage=24/7",
        r"7\s*[\u00d7x]\s*24\s*小时",
        r"\b(?:7\s*[\u00d7x]\s*24\s*h|24\s*h\s*/\s*24[, ]+7\s*j\s*/\s*7)\b",
    ),
    ("duration>1.5y", r"超一年半", r"\bplus d'un an et demi\b"),
    ("experience>20y", r"二十余年", r"\bplus de (?:vingt|20) ans\b"),
]
SOURCE_RESIDUE = r"[0-9一二三四五六七八九十百千万余半]|若干|若幹"
FRENCH_RESIDUE = (
    r"\d|\b(?:deux|trois|quatre|cinq|six|sept|huit|neuf|dix|vingt|"
    r"cent|milliers?|centaines?|demi|plusieurs|quelques)\b"
)
QUALIFIER = r"(?:environ|près de|presque|moins de|plus de|au moins|au plus|jusqu'à|exactement)\s*$"


def extract(text: str, *, source: bool) -> Counter[str] | None:
    """Extract supported numeric facts, returning None for unresolved quantities."""
    remaining = unicodedata.normalize("NFKC", text).replace("\u2019", "'")
    facts: Counter[str] = Counter()
    rest = consume(remaining, facts)
    if rest is None:
        return None
    remaining = rest.replace("半徑", "rayon").replace("半径", "rayon").casefold()
    for label, chinese, french in RULES:
        pattern = chinese if source else french
        matches = list(re.finditer(pattern, remaining))
        for match in reversed(matches):
            if not source and re.search(QUALIFIER, remaining[: match.start()]):
                return None
            facts[label] += 1
            remaining = remaining[: match.start()] + " " + remaining[match.end() :]
    if re.search(SOURCE_RESIDUE if source else FRENCH_RESIDUE, remaining):
        return None
    return facts


def reference_pattern(reference: str) -> re.Pattern[str]:
    """Match a literal identifier without accepting an adjacent ASCII identifier character."""
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(reference) + r"(?![A-Za-z0-9_])")


def gate(
    text: str, citation: str, candidates: dict[str, str], *, references: tuple[str, ...] = ()
) -> tuple[list[str] | None, str | dict[str, int]]:
    """Filter candidates by source quantity evidence before any model arbitrates."""
    # Only quantities in the target span belong in the translation.
    if text not in citation:
        return None, "fragment absent de la citation"

    def quantities_only(fragment: str) -> str:
        for ref in sorted(references, key=len, reverse=True):
            fragment = reference_pattern(ref).sub(" ", fragment)
        return fragment

    expected = extract(quantities_only(text), source=True)
    if expected and text != citation:
        spans = list(re.finditer(re.escape(text), citation))
        if len(spans) != 1:
            return None, "rattachement du fragment ambigu"
        start, end = spans[0].span()
        if re.search(
            r"(?:约|約|大约|大約|近|超过|超過|不到|至少|至多|最多|高達|可達|高达|可达|最大|[数數几幾])\s*$",
            citation[:start],
        ) or re.match(r"\s*(?:左右|上下|以上|以下)", citation[end:]):
            return None, "le contexte adjacent qualifie la quantité"
    if expected is None:
        return None, "quantité source non résolue"
    parsed = {model: extract(quantities_only(t), source=False) for model, t in candidates.items()}
    if any(v is None for v in parsed.values()):
        return None, "interprétation numérique non résolue"
    return [m for m, v in parsed.items() if v == expected], dict(expected)
