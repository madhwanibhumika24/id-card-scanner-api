
import re
from datetime import date


DATE_PATTERN = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])[-/.]"
    r"(0?[1-9]|1[0-2])[-/.]"
    r"(\d{4})\b"
    r"|\b(\d{4})[-/.]"
    r"(0?[1-9]|1[0-2])[-/.]"
    r"(0?[1-9]|[12]\d|3[01])\b"
)

DOCUMENT_PATTERN = re.compile(
    r"\b(?:DL|ID|DOCUMENT|LICENSE|LICENCE)"
    r"\s*(?:NO\.?|NUMBER|#)?\s*[:#-]?\s*"
    r"([A-Z0-9][A-Z0-9/-]{3,})\b",
    re.IGNORECASE,
)

NAME_LABEL_PATTERN = re.compile(
    r"\b(?:FULL\s*NAME|NAME)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)

IGNORED_NAME_TERMS = (
    "faculty", "facultv", "engineering", "enginfering",
    "technology", "technoiogy", "university", "college",
    "department", "government", "authority", "license",
    "licence", "driver", "class", "address", "sex", "gender",
    "dob", "birth", "expiry", "valid", "emergency", "urgency",
    "group", "student", "identity",
)


def normalize_date(match: re.Match) -> str | None:
    """Convert supported date formats to YYYY-MM-DD."""
    groups = match.groups()

    try:
        if groups[0] is not None:
            day, month, year = map(int, groups[:3])
        else:
            year, month, day = map(int, groups[3:6])

        return date(year, month, day).isoformat()
    except (ValueError, TypeError):
        return None


def extract_fields(full_text: str) -> dict:
    """Extract candidate identity fields from OCR text."""
    lines = [
        re.sub(r"\s+", " ", line).strip(" :-")
        for line in full_text.splitlines()
        if line.strip()
    ]

    full_name = None
    document_number = None
    date_of_birth = None
    expiry_date = None

    # Prefer explicit name labels.
    for line in lines:
        match = NAME_LABEL_PATTERN.search(line)
        if match:
            candidate = match.group(1).strip(" :-")
            lower = candidate.lower()

            if (
                candidate
                and not re.search(r"\d", candidate)
                and not any(term in lower for term in IGNORED_NAME_TERMS)
            ):
                full_name = candidate
                break

    # Prefer clearly labelled document numbers.
    for line in lines:
        match = DOCUMENT_PATTERN.search(line)
        if match:
            document_number = match.group(1).upper()
            break

    # Extract dates only when their labels are identifiable.
    for index, line in enumerate(lines):
        for match in DATE_PATTERN.finditer(line):
            normalized = normalize_date(match)
            if normalized is None:
                continue

            current = line.lower()
            previous = lines[index - 1].lower() if index > 0 else ""

            if date_of_birth is None and (
                re.search(r"\b(dob|date of birth|birth|born)\b", current)
                or re.search(r"\b(dob|date of birth|birth|born)\b", previous)
            ):
                date_of_birth = normalized

            if expiry_date is None and (
                re.search(
                    r"\b(expiry|expires|expiration|valid until|exp)\b",
                    current,
                )
                or re.search(
                    r"\b(expiry|expires|expiration|valid until|exp)\b",
                    previous,
                )
            ):
                expiry_date = normalized

    # Fallback: prefer plausible personal-name lines, not headings.
    if full_name is None:
        candidates = []

        for index, line in enumerate(lines):
            candidate = line.strip(" :-")
            lower = candidate.lower()
            words = candidate.split()

            if (
                len(words) < 2
                or len(candidate) > 50
                or re.search(r"\d", candidate)
                or any(term in lower for term in IGNORED_NAME_TERMS)
            ):
                continue

            # Prefer lines that look like a person's name.
            if all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word) for word in words):
                score = 0

                # Name-like lines near a course/qualification label
                # are more plausible than institution headings.
                if index + 1 < len(lines):
                    if lines[index + 1].strip().upper() in {
                        "MCA", "BCA", "MBA", "BTECH", "B.TECH",
                    }:
                        score += 3

                if candidate.istitle():
                    score += 1

                candidates.append((score, index, candidate))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            full_name = candidates[0][2]

    return {
        "full_name": full_name,
        "document_number": document_number,
        "date_of_birth": date_of_birth,
        "expiry_date": expiry_date,
    }
