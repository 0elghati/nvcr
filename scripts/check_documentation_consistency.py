#!/usr/bin/env python3
"""Check durable public-documentation invariants for NVCR."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlsplit


MARKDOWN_DIRECTORIES = ("docs", "results", "evidence", "docker", "scripts")
EXCLUDED_NAMES = {"AGENTS.md", "CHANGELOG.md"}
ACTIVE_ONBOARDING_FILES = {
    "README.md",
    "docs/first-run.md",
    "docs/installation.md",
    "docker/docker-hub.md",
}

INLINE_LINK = re.compile(
    r"!?\[[^\]\n]*\]\((?P<destination><[^>\n]+>|[^)\n]+)\)"
)
REFERENCE_LINK = re.compile(
    r"(?m)^[ \t]*\[[^\]\n]+\]:[ \t]*(?P<destination><[^>\n]+>|\S+)"
)
HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
EXPLICIT_ANCHOR = re.compile(
    r"<[A-Za-z][^>]*\s(?:id|name)=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE
)

PROHIBITED_PROSE = (
    ("journal wording", re.compile(r"\bjournals?\b", re.IGNORECASE)),
    ("venue name", re.compile(r"\bSoftwareX\b", re.IGNORECASE)),
    ("manuscript wording", re.compile(r"\bmanuscripts?\b", re.IGNORECASE)),
    ("submission wording", re.compile(r"\bsubmissions?\b", re.IGNORECASE)),
    ("reviewer wording", re.compile(r"\breviewers?\b", re.IGNORECASE)),
    ("peer-review wording", re.compile(r"\bpeer[- ]review\b", re.IGNORECASE)),
    ("editorial wording", re.compile(r"\beditorial\b", re.IGNORECASE)),
    (
        "publication-readiness wording",
        re.compile(
            r"\bpublication(?:-ready|\s+(?:strategy|evidence|matrix|run|rows?|"
            r"table|package|prerequisites?|readiness))\b",
            re.IGNORECASE,
        ),
    ),
)

# These names are immutable interfaces even though their original name contains
# venue-specific wording. Inline code is removed before this exception applies.
LEGACY_IDENTIFIER = re.compile(
    r"\b(?:[A-Za-z0-9.-]+[_\.]softwarex[_\.][A-Za-z0-9_.-]+|"
    r"softwarex[_\.][A-Za-z0-9_.-]+)\b",
    re.IGNORECASE,
)

SECRET_PATTERNS = (
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "private key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "signed URL credential",
        re.compile(
            r"[?&](?:X-Amz-(?:Credential|Signature)|sig|signature|token|access_token)="
            r"[^\s&#]{8,}",
            re.IGNORECASE,
        ),
    ),
)

LOCAL_USER_PATH = re.compile(
    r"(?:/home/|/Users/)(?P<unix>[A-Za-z0-9._-]+)/|"
    r"[A-Za-z]:\\Users\\(?P<windows>[A-Za-z0-9._-]+)\\"
)
GENERIC_USER_NAMES = {"nvcr", "runner", "ubuntu", "user", "username"}
SEMVER = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
PINNED_NVCR_IMAGE = re.compile(
    r"\b(?:[A-Za-z0-9._-]+/)*nvcr:v?\d+\.\d+\.\d+(?=[^0-9]|$)", re.IGNORECASE
)
UNQUALIFIED_LATEST_IMAGE = re.compile(
    r"\bomarelghati/nvcr:latest(?![-A-Za-z0-9_])", re.IGNORECASE
)
HARDCODED_LATEST_STABLE = re.compile(
    r"\blatest stable[^\n]{0,120}\bv?\d+\.\d+\.\d+\b", re.IGNORECASE
)


@dataclass(frozen=True, order=True)
class Issue:
    path: str
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _read(path: Path, root: Path, issues: list[Issue]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        issues.append(Issue(_relative(root, path), 1, f"cannot read file: {exc}"))
        return None


def _markdown_files(root: Path) -> list[Path]:
    candidates = list(root.glob("*.md"))
    for name in MARKDOWN_DIRECTORIES:
        directory = root / name
        if directory.is_dir():
            candidates.extend(directory.rglob("*.md"))

    selected: set[Path] = set()
    for path in candidates:
        relative = path.relative_to(root)
        if path.name in EXCLUDED_NAMES or "third_party" in relative.parts:
            continue
        selected.add(path)
    return sorted(selected)


def _without_markdown_code(text: str) -> str:
    """Blank fenced and inline code while preserving source line numbers."""

    output: list[str] = []
    fence_character: str | None = None
    for line in text.splitlines(keepends=True):
        fence = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
        if fence:
            character = fence.group(1)[0]
            if fence_character is None:
                fence_character = character
            elif fence_character == character:
                fence_character = None
            output.append("\n" if line.endswith("\n") else "")
            continue
        if fence_character is not None:
            output.append("\n" if line.endswith("\n") else "")
            continue

        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        body = re.sub(r"`+[^`\n]*`+", "", body)
        output.append(body + newline)
    return "".join(output)


def _link_destination(raw: str) -> str:
    destination = raw.strip()
    if destination.startswith("<"):
        closing = destination.find(">")
        return destination[1:closing] if closing >= 0 else destination[1:]
    return destination.split(maxsplit=1)[0]


def _heading_slug(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"`+([^`]*)`+", r"\1", text)
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s", "-", text).strip("-")


def _markdown_anchors(text: str) -> set[str]:
    anchors = {unquote(match.group(1)) for match in EXPLICIT_ANCHOR.finditer(text)}
    occurrences: dict[str, int] = {}
    for match in HEADING.finditer(text):
        base = _heading_slug(match.group(1))
        if not base:
            continue
        duplicate = occurrences.get(base, 0)
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
        occurrences[base] = duplicate + 1
    return anchors


def _check_links(root: Path, path: Path, text: str, issues: list[Issue]) -> None:
    searchable = _without_markdown_code(text)
    for pattern in (INLINE_LINK, REFERENCE_LINK):
        for match in pattern.finditer(searchable):
            destination = unquote(_link_destination(match.group("destination")))
            line = _line_number(searchable, match.start("destination"))
            if not destination or destination.startswith("//"):
                continue
            try:
                parsed = urlsplit(destination)
            except ValueError as exc:
                issues.append(Issue(_relative(root, path), line, f"invalid link: {exc}"))
                continue
            if parsed.scheme or parsed.netloc:
                continue

            if parsed.path:
                candidate = (path.parent / parsed.path).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    issues.append(
                        Issue(_relative(root, path), line, "local link escapes repository")
                    )
                    continue
            else:
                candidate = path

            if not candidate.exists():
                issues.append(
                    Issue(
                        _relative(root, path),
                        line,
                        f"local link target does not exist: {destination!r}",
                    )
                )
                continue

            fragment = unquote(parsed.fragment)
            if fragment and candidate.is_file() and candidate.suffix.lower() == ".md":
                target_text = _read(candidate, root, issues)
                if target_text is not None and fragment not in _markdown_anchors(target_text):
                    issues.append(
                        Issue(
                            _relative(root, path),
                            line,
                            f"Markdown fragment does not exist: {destination!r}",
                        )
                    )


def _check_public_text(root: Path, path: Path, text: str, issues: list[Issue]) -> None:
    relative = _relative(root, path)
    prose = LEGACY_IDENTIFIER.sub("", _without_markdown_code(text))

    for match in re.finditer(r"\bnvrc\b", text, re.IGNORECASE):
        issues.append(
            Issue(relative, _line_number(text, match.start()), "obsolete repository name 'nvrc'")
        )
    for label, pattern in PROHIBITED_PROSE:
        for match in pattern.finditer(prose):
            issues.append(
                Issue(relative, _line_number(prose, match.start()), f"prohibited {label}")
            )
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            issues.append(
                Issue(relative, _line_number(text, match.start()), f"possible {label}")
            )
    for match in LOCAL_USER_PATH.finditer(text):
        user = match.group("unix") or match.group("windows")
        if user.lower() not in GENERIC_USER_NAMES:
            issues.append(
                Issue(
                    relative,
                    _line_number(text, match.start()),
                    f"machine-local user path for {user!r}",
                )
            )


def _check_onboarding_versions(
    root: Path, path: Path, text: str, issues: list[Issue]
) -> None:
    relative = _relative(root, path)
    if relative not in ACTIVE_ONBOARDING_FILES and not (
        relative.startswith("docs/docker") and relative.endswith(".md")
    ):
        return

    checks = (
        (PINNED_NVCR_IMAGE, "pinned semantic-version NVCR image tag"),
        (UNQUALIFIED_LATEST_IMAGE, "unqualified NVCR latest image tag"),
        (HARDCODED_LATEST_STABLE, "hardcoded latest-stable version claim"),
    )
    for pattern, message in checks:
        searchable = _without_markdown_code(text) if pattern is HARDCODED_LATEST_STABLE else text
        for match in pattern.finditer(searchable):
            issues.append(Issue(relative, _line_number(searchable, match.start()), message))


def _cff_field(text: str, name: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(name)}:[ \t]*([^#\n]+)", text)
    return match.group(1).strip().strip("\"'") if match else None


def _preferred_citation_version(text: str) -> tuple[int, str] | None:
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == "preferred-citation:"),
        None,
    )
    if start is None:
        return None
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line[0].isspace():
            break
        match = re.match(r"^[ \t]+version:[ \t]*([^#\n]+)", line)
        if match:
            return index + 1, match.group(1).strip().strip("\"'")
    return None


def _check_citation(root: Path, issues: list[Issue]) -> str:
    version_path = root / "version.txt"
    citation_path = root / "CITATION.cff"
    version_text = _read(version_path, root, issues)
    citation = _read(citation_path, root, issues)
    version = version_text.strip() if version_text else ""
    if not SEMVER.fullmatch(version):
        issues.append(Issue("version.txt", 1, f"invalid semantic version {version!r}"))
    if citation is None:
        return version

    top = _cff_field(citation, "version")
    if top != version:
        issues.append(
            Issue("CITATION.cff", 1, f"version {top!r} does not match version.txt {version!r}")
        )
    preferred = _preferred_citation_version(citation)
    if preferred is None:
        issues.append(Issue("CITATION.cff", 1, "preferred-citation version is missing"))
    elif preferred[1] != version:
        issues.append(
            Issue(
                "CITATION.cff",
                preferred[0],
                f"preferred-citation version {preferred[1]!r} does not match "
                f"version.txt {version!r}",
            )
        )
    return version


def check_repository(root: Path | str) -> tuple[str, list[Path], list[Issue]]:
    root_path = Path(root).expanduser().resolve()
    issues: list[Issue] = []
    version = _check_citation(root_path, issues)
    markdown_files = _markdown_files(root_path)
    for path in markdown_files:
        text = _read(path, root_path, issues)
        if text is None:
            continue
        _check_links(root_path, path, text, issues)
        _check_public_text(root_path, path, text, issues)
        _check_onboarding_versions(root_path, path, text, issues)
    return version, markdown_files, sorted(set(issues))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check NVCR citation metadata, local links, and public documentation."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    args = parser.parse_args(argv)
    version, markdown_files, issues = check_repository(args.root)
    if issues:
        for issue in issues:
            print(issue.render(), file=sys.stderr)
        print(f"documentation consistency failed: {len(issues)} issue(s)", file=sys.stderr)
        return 1
    print(
        f"documentation consistency passed: version {version}; "
        f"{len(markdown_files)} Markdown files checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
