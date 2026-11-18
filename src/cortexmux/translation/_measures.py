"""Recognize measurement spans before the legacy quantity checks."""

import re
from collections import Counter
from decimal import Decimal

NUMBER = r"(?:\d{1,3}(?: \d{3})+|\d+)(?:[.,]\d+)?"
UNIT = r"公斤|千克|キログラム|kg|毫米|公釐|ミリメートル|mm|kilogrammes?|millimètres?|Ko|ko|KO|[kK]B"
PREFIX = (
    r"jusqu'[aà]|allant jusqu'à|atteignant|au maximum|高達|可達|高达|可达|最大|"
    r"environ|約|约|plus de|超過|超过|moins de|au moins|au plus"
)
MEASURE = re.compile(
    rf"(?P<prefix>(?i:{PREFIX}))?\s*(?<![\d.,])(?P<lo>{NUMBER})"
    rf"(?:\s*(?:[\u2013\u2014~〜\uff5e-]|à|至)\s*(?P<hi>{NUMBER}))?\s*(?P<unit>{UNIT})(?![a-zA-Z])"
)
FEW = re.compile(r"數分鐘內|数分钟内|数分以内|en quelques minutes\b", re.I)
EXTRA = re.compile(
    r"(?:[+\u2212±-]|(?:moins|plus) de|près de|environ|jusqu'à|au moins|au plus|"
    r"大約|大约|約|约|超|不足|至少|未満)\s*$",
    re.I,
)


def _number(raw: str) -> str:
    value = Decimal(raw.replace(" ", "").replace(",", "."))
    whole, _, fraction = format(value, "f").partition(".")
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def consume(text: str, facts: Counter[str]) -> str | None:
    """Consume supported measurement spans and record canonical quantities."""

    def few(match: re.Match[str]) -> str:
        facts["duration:within_few_minutes"] += 1
        return " "

    text = FEW.sub(few, text)
    for match in reversed(list(MEASURE.finditer(text))):
        if EXTRA.search(text[: match.start()]) or re.match(
            r"\s*(?:以上|以下|左右|前後|environ|au maximum|au minimum)", text[match.end() :], re.I
        ):
            return None
        unit = match["unit"].casefold()
        unit = (
            "kg"
            if unit in ("公斤", "千克", "キログラム", "kg") or unit.startswith("kilogramme")
            else "mm"
            if unit in ("毫米", "公釐", "ミリメートル", "mm") or unit.startswith("millimètre")
            else "kB"
        )
        prefix = (match["prefix"] or "").casefold()
        bound = "exact"
        if prefix:
            if prefix in ("environ", "約", "约"):
                bound = "approx"
            elif prefix in ("plus de", "超過", "超过"):
                bound = "gt"
            elif prefix == "moins de":
                bound = "lt"
            elif prefix == "au moins":
                bound = "ge"
            else:
                bound = "le"

        lo = _number(match["lo"])
        if match["hi"]:
            hi = _number(match["hi"])
            if prefix or Decimal(lo) > Decimal(hi):
                return None
            bound = "range"
            lo += ".." + hi
        facts[f"{unit}:{bound}:{lo}"] += 1
        text = text[: match.start()] + " " + text[match.end() :]
    return text
