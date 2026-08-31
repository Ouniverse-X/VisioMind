from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any


def extract_json_object(content: str, *, label: str) -> dict[str, Any]:
    stripped = content.strip()
    candidates = [stripped]
    if "```" in stripped:
        for block in stripped.split("```"):
            candidate = block.strip()
            if not candidate:
                continue
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            candidates.append(candidate)

    decoder = JSONDecoder()
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            for start_index, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(candidate[start_index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Failed to parse JSON object from {label} response")
