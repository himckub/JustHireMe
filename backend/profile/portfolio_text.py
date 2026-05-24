"""Pure text/URL/noise helpers for portfolio ingestion.

Stateless string and URL utilities (canonicalization, navigation-noise
detection, whitespace normalization, dedup) shared by the portfolio crawl and
extraction modules. No dependency on the rest of the portfolio package, so it
sits at the base of that subsystem's dependency graph.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse


def _canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", parsed.query, ""))


def _same_origin(root: str, other: str) -> bool:
    a = urlparse(root)
    b = urlparse(other)
    return a.scheme in {"http", "https"} and b.scheme in {"http", "https"} and a.netloc.lower() == b.netloc.lower()


def _looks_like_asset(url: str) -> bool:
    return bool(re.search(r"\.(png|jpe?g|gif|webp|svg|pdf|zip|mp4|mov|css|js|ico)(\?|$)", url, re.I))


def _normalize_block_text(value: str) -> str:
    value = re.sub(r"\r", "\n", value or "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _nav_noise(line: str) -> bool:
    lower = line.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "", lower)
    if len(lower) <= 2:
        return True
    if _is_concatenated_nav(normalized):
        return True
    if lower in {"home", "about", "projects", "work", "portfolio", "contact", "resume", "blog", "menu", "close"}:
        return True
    return bool(
        len(lower.split()) <= 5
        and re.fullmatch(r"(home|about|projects?|work|contact|resume|blog|services?)(\s+[a-z]+)*", lower)
    )


def _is_concatenated_nav(value: str) -> bool:
    if not value or len(value) > 80:
        return False
    tokens = ("home", "about", "projects", "project", "work", "portfolio", "contact", "resume", "blog", "menu", "github", "linkedin")
    remaining = value
    hits = 0
    while remaining:
        match = next((token for token in tokens if remaining.startswith(token)), "")
        if not match:
            return False
        remaining = remaining[len(match):]
        hits += 1
    return hits >= 2


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text or "")
    return match.group(0) if match else ""


def _same_key(a: str, b: str) -> bool:
    return re.sub(r"[^a-z0-9]+", "", a.lower()) == re.sub(r"[^a-z0-9]+", "", b.lower())


def _repo_title_from_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[1].replace("-", " ").replace("_", " ").title() if len(parts) >= 2 else ""


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = _normalize_block_text(str(value))
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out
