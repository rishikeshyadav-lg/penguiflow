"""The learning-control-plane tests and examples hold synthetic data only.

The judge kit was built from real customer runs, and those runs (prompts, answers, customer and
brand names) must never be copied into this repository. This scan fails if a fixture links to a
real host or mailbox, or names one of the customers the kit's lessons came from. The customer names
are kept only as truncated SHA-256 hashes, so the denylist itself reveals nothing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCANNED_FOLDERS = (
    REPOSITORY_ROOT / "tests" / "learning_control_plane",
    REPOSITORY_ROOT / "examples" / "lcp_plain_python_agent",
)
SCANNED_SUFFIXES = frozenset({".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"})
ALLOWED_HOSTS = frozenset({"example.com", "example.org", "localhost", "127.0.0.1"})
ALLOWED_MAIL_DOMAINS = frozenset({"example.com", "example.org"})
# Truncated SHA-256 of lower-cased customer and brand names (single words and two-word names).
DENIED_NAME_HASHES = frozenset(
    {
        "d6d679422d6a9759",
        "aa59343f9ebb354f",
        "34a533dfd7273a9b",
        "16ab4b65810d48bb",
        "c04be78d495991cb",
        "ac1588384430e1d7",
        "ea68b29d0870cffb",
        "42dc67684180f492",
        "be78386834154e1a",
        "fe081509b01186f1",
    }
)
_URL_HOST = re.compile(r"https?://([A-Za-z0-9.-]+)")
_MAIL_DOMAIN = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_WORD = re.compile(r"[a-z0-9]+")


def _scanned_files() -> list[Path]:
    return [
        path
        for folder in SCANNED_FOLDERS
        if folder.is_dir()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.suffix in SCANNED_SUFFIXES
        and "__pycache__" not in path.parts
        # This file holds the scanner's own negative examples.
        and path != Path(__file__).resolve()
    ]


def _name_hashes(text: str) -> set[str]:
    words = _WORD.findall(text.casefold())
    phrases = [*words, *(f"{first} {second}" for first, second in zip(words, words[1:], strict=False))]
    return {hashlib.sha256(phrase.encode()).hexdigest()[:16] for phrase in phrases}


def _problems(path: Path, denied_hashes: frozenset[str] = DENIED_NAME_HASHES) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    problems = [
        f"{path.name}: link to {host}"
        for host in _URL_HOST.findall(text)
        if host.casefold() not in ALLOWED_HOSTS and not host.casefold().endswith(".example.com")
    ]
    problems.extend(
        f"{path.name}: mailbox at {domain}"
        for domain in _MAIL_DOMAIN.findall(text)
        if domain.casefold() not in ALLOWED_MAIL_DOMAINS
    )
    if _name_hashes(text) & denied_hashes:
        problems.append(f"{path.name}: names a real customer")
    return problems


def test_learning_control_plane_fixtures_hold_only_synthetic_data() -> None:
    files = _scanned_files()

    problems = [problem for path in files for problem in _problems(path)]

    assert files, "the privacy scan found nothing to scan"
    assert problems == []


def test_the_scan_catches_a_real_host_and_a_denied_name(tmp_path: Path) -> None:
    denied_hashes = frozenset({hashlib.sha256(b"zebra corp").hexdigest()[:16]})
    fixture = tmp_path / "fixture.md"
    fixture.write_text("Ask Zebra Corp at https://real-customer.net or ops@real-customer.net")

    assert _problems(fixture, denied_hashes) == [
        "fixture.md: link to real-customer.net",
        "fixture.md: mailbox at real-customer.net",
        "fixture.md: names a real customer",
    ]
