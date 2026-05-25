from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    label: str
    phrases: tuple[str, ...]


_OPERATIONAL_HEADINGS = {
    "date filter",
    "delivery",
    "execution instructions",
    "format",
    "information sources",
    "objective",
    "response",
    "source policy",
    "evaluation goal",
    "question",
}

_TASK_HEADINGS = {
    "core task",
    "task",
}

_TOPIC_HINTS = {
    "topic",
    "topics",
    "priority order",
}

_LEADING_VERBS = {
    "compile",
    "create",
    "deliver",
    "generate",
    "produce",
    "write",
}


def missing_requirements(topic: str, report_markdown: str) -> list[str]:
    report_text = _normalize_text(report_markdown)
    missing: list[str] = []
    for requirement in extract_requirements(topic):
        if not any(phrase in report_text for phrase in requirement.phrases):
            missing.append(requirement.label)
    return missing


def extract_requirements(topic: str) -> list[Requirement]:
    request = extract_user_request(topic)
    requirements: list[Requirement] = _inline_requirements(request)
    in_topics = False

    for line in request.splitlines():
        heading = _section_label(line)
        if heading is not None:
            in_topics = _is_topic_heading(heading)
            section = _section_requirement_from_heading(heading)
            if section is not None:
                requirements.append(section)
            continue

        if in_topics:
            bullet = _bullet_text(line)
            if bullet:
                topic_requirement = _topic_requirement_from_bullet(bullet)
                if topic_requirement is not None:
                    requirements.append(topic_requirement)

    return _dedupe_requirements(requirements)


def extract_user_request(topic: str) -> str:
    match = re.search(r"^##\s+Question\s*$", topic, flags=re.MULTILINE)
    if match is None:
        return topic.strip()
    start = match.end()
    end_match = re.search(r"^##\s+Source Policy\s*$", topic[start:], flags=re.MULTILINE)
    end = start + end_match.start() if end_match is not None else len(topic)
    return topic[start:end].strip()


def _section_requirement_from_heading(heading: str) -> Requirement | None:
    cleaned = _clean_label(heading)
    normalized = _normalize_text(cleaned)
    if not normalized:
        return None
    if _is_topic_heading(cleaned):
        return None
    if _is_operational_heading(normalized):
        return None
    if ":" in cleaned:
        prefix, value = cleaned.split(":", 1)
        if _normalize_text(prefix) in _TASK_HEADINGS:
            cleaned = _drop_leading_action(value)
            normalized = _normalize_text(cleaned)
    if normalized in _TASK_HEADINGS or _is_operational_heading(normalized):
        return None
    phrases = _phrases_for(cleaned)
    if not phrases:
        return None
    return Requirement(label=cleaned, phrases=tuple(phrases))


def _topic_requirement_from_bullet(text: str) -> Requirement | None:
    cleaned = _clean_label(text)
    if not cleaned:
        return None
    if ":" in cleaned:
        prefix, value = cleaned.split(":", 1)
        if _is_instruction_prefix(prefix):
            cleaned = value.strip()
    cleaned = _first_clause(cleaned)
    phrases = _phrases_for(cleaned)
    if not phrases:
        return None
    return Requirement(label=cleaned, phrases=tuple(phrases))


def _inline_requirements(request: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    pattern = r"\b(?:with|include|including|containing)\s+(.{1,180}?)(?:[.\n]|$)"
    for match in re.finditer(pattern, request, flags=re.IGNORECASE):
        for part in re.split(r"\s+and\s+|,\s*", match.group(1)):
            label = _clean_label(part)
            if not _looks_like_section_name(label):
                continue
            phrases = _phrases_for(label)
            if phrases:
                requirements.append(Requirement(label=label, phrases=tuple(phrases)))
    return requirements


def _markdown_heading(line: str) -> str | None:
    match = re.match(r"^\s{0,3}#{2,6}\s+(.+?)\s*$", line)
    if match is None:
        return None
    return _clean_label(match.group(1))


def _section_label(line: str) -> str | None:
    heading = _markdown_heading(line)
    if heading is not None:
        return heading
    match = re.match(r"^\s*(?:\d+[.)]\s+)?\*\*(.+?)\*\*\s*$", line)
    if match is not None:
        return _clean_label(match.group(1))
    return None


def _bullet_text(line: str) -> str:
    match = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$", line)
    return match.group(1).strip() if match is not None else ""


def _is_topic_heading(value: str) -> bool:
    normalized = _normalize_text(value)
    return any(hint in normalized.split() or hint in normalized for hint in _TOPIC_HINTS)


def _is_operational_heading(normalized: str) -> bool:
    return any(normalized == heading or normalized.startswith(heading + " ") for heading in _OPERATIONAL_HEADINGS)


def _is_instruction_prefix(value: str) -> bool:
    normalized = _normalize_text(value)
    if len(normalized.split()) > 4:
        return False
    return value.strip().isupper() or normalized in {"primary focus", "focus", "priority"}


def _first_clause(value: str) -> str:
    without_detail = re.split(r"\s+\(", value, maxsplit=1)[0]
    return without_detail.strip(" .:-")


def _drop_leading_action(value: str) -> str:
    words = _clean_label(value).split()
    while words and _normalize_text(words[0]) in _LEADING_VERBS:
        words = words[1:]
    return " ".join(words).strip()


def _phrases_for(label: str) -> list[str]:
    cleaned = _clean_label(label)
    normalized = _normalize_text(cleaned)
    if not normalized:
        return []
    phrases = [normalized]
    for part in re.split(r"[,/&]|\band\b|\bor\b", cleaned, flags=re.IGNORECASE):
        phrase = _normalize_text(part)
        if len(phrase) >= 3 and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _looks_like_section_name(label: str) -> bool:
    if len(label.split()) < 2:
        return False
    if _is_operational_heading(_normalize_text(label)):
        return False
    if "&" in label:
        return True
    words = [word.strip("()[]{}") for word in label.split()]
    titled = [word for word in words if word[:1].isupper() and any(char.islower() for char in word[1:])]
    return len(titled) >= 2


def _dedupe_requirements(requirements: list[Requirement]) -> list[Requirement]:
    seen: set[str] = set()
    deduped: list[Requirement] = []
    for requirement in requirements:
        key = _normalize_text(requirement.label)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(requirement)
    return deduped


def _clean_label(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .:-")


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
