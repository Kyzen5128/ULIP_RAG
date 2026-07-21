from __future__ import annotations

import re
from typing import Any


UNICODE_FRACTIONS = {
    "¼": 0.25, "½": 0.5, "¾": 0.75, "⅛": 0.125,
    "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}


def inch_text_to_mm(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("\u00a0", " ")
    for token, fraction in UNICODE_FRACTIONS.items():
        text = text.replace(token, f" {fraction}")
    text = re.sub(r'["\u201d\u2033]', "", text).strip()
    match = re.match(r"^(\d+)\s+(\d+)/(\d+)$", text)
    if match:
        inches = int(match.group(1)) + int(match.group(2)) / int(match.group(3))
    else:
        match = re.match(r"^(\d+)\s+([0-9.]+)$", text)
        if match:
            inches = int(match.group(1)) + float(match.group(2))
        elif re.match(r"^[0-9.]+$", text):
            inches = float(text)
        else:
            return None
    return round(inches * 25.4, 1)


def parse_named_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for item in measurements:
        name = str(item.get("name") or item.get("label") or "").strip().casefold()
        value = item.get("measure") or item.get("text")
        if name and value:
            raw[name] = str(value)

    def first(*names: str) -> tuple[float | None, str | None]:
        for name in names:
            if name in raw:
                parsed = inch_text_to_mm(raw[name])
                if parsed is not None:
                    return parsed, name
        return None, None

    def parsed_range(min_names: tuple[str, ...], max_names: tuple[str, ...]) -> dict[str, Any] | None:
        minimum, minimum_key = first(*min_names)
        maximum, maximum_key = first(*max_names)
        if minimum is None or maximum is None:
            return None
        if minimum > maximum:
            minimum, maximum = maximum, minimum
            minimum_key, maximum_key = maximum_key, minimum_key
        return {
            "min_mm": minimum, "max_mm": maximum,
            "min_source": minimum_key, "max_source": maximum_key,
        }

    # IKEA uses either Width+Depth+Height or Length+Width+Height. In the
    # latter, Length is the long footprint axis and Width is the short axis.
    diameter, diameter_key = first("diameter")
    if diameter is not None and not any(name in raw for name in ("width", "length", "depth")):
        width, width_key = diameter, diameter_key
        depth, depth_key = diameter, diameter_key
    elif "length" in raw and "width" in raw and "depth" not in raw:
        width, width_key = first("length")
        depth, depth_key = first("width")
    else:
        width, width_key = first("width", "length")
        depth, depth_key = first("depth")
    height, height_key = first(
        "height", "height including back cushions", "max. headboard height", "headboard height"
    )
    ranges = {
        "width": parsed_range(("min. width", "minimum width"), ("max. width", "maximum width")),
        "depth": parsed_range(("min. depth", "minimum depth"), ("max. depth", "maximum depth")),
        "height": parsed_range(("min. height", "minimum height"), ("max. height", "maximum height")),
        "length": parsed_range(("min. length", "minimum length"), ("max. length", "maximum length")),
    }
    ranges = {axis: value for axis, value in ranges.items() if value is not None}
    dims = {k: v for k, v in (("width", width), ("depth", depth), ("height", height)) if v is not None}
    status = "three_axes" if len(dims) == 3 else "partial" if dims else "missing"
    return {
        "dimensions_mm": dims,
        "axis_source": {"width": width_key, "depth": depth_key, "height": height_key},
        "ranges_mm": ranges,
        "raw": raw,
        "axis_count": len(dims),
        "status": status,
        "dimension_quality_status": "range_present" if ranges else status,
        "hard_filter_eligible": status == "three_axes" and not ranges,
    }
