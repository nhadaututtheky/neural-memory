"""Prepare arbitrary markdown into a trainer-friendly structure.

The doc trainer in NeuralMemory builds graph structure from heading hierarchy and
skips very short sections. This module reshapes existing markdown so the content
is more likely to survive chunking and produce richer cross-links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from neural_memory.extraction.keywords import extract_weighted_keywords
from neural_memory.utils.tag_normalizer import TagNormalizer

_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")

CANONICAL_SECTIONS: tuple[str, ...] = (
    "Overview",
    "Concepts",
    "Relationships",
    "Procedures",
    "Examples",
    "Reference",
)

_HEADING_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Overview",
        (
            "about",
            "abstract",
            "background",
            "goal",
            "goals",
            "introduction",
            "motivation",
            "overview",
            "purpose",
            "summary",
            "why",
        ),
    ),
    (
        "Concepts",
        (
            "architecture",
            "component",
            "components",
            "concept",
            "concepts",
            "data model",
            "design",
            "entity",
            "entities",
            "glossary",
            "model",
            "models",
            "terminology",
        ),
    ),
    (
        "Relationships",
        (
            "causal",
            "causality",
            "dependency",
            "dependencies",
            "flow",
            "flows",
            "how it works",
            "interaction",
            "interactions",
            "lifecycle",
            "relationship",
            "relationships",
            "sequence",
            "topology",
        ),
    ),
    (
        "Procedures",
        (
            "deploy",
            "deployment",
            "guide",
            "how to",
            "installation",
            "operations",
            "procedure",
            "procedures",
            "quickstart",
            "runbook",
            "setup",
            "steps",
            "tutorial",
            "usage",
            "workflow",
        ),
    ),
    (
        "Examples",
        (
            "case study",
            "cases",
            "demo",
            "demos",
            "example",
            "examples",
            "recipe",
            "recipes",
            "sample",
            "samples",
            "scenario",
            "scenarios",
            "snippet",
            "snippets",
        ),
    ),
    (
        "Reference",
        (
            "api",
            "appendix",
            "cli",
            "config",
            "configuration",
            "faq",
            "flags",
            "option",
            "options",
            "reference",
            "schema",
            "schemas",
            "table",
            "tables",
            "troubleshooting",
        ),
    ),
)

_RELATIONAL_CUES: tuple[str, ...] = (
    "after",
    "before",
    "because",
    "causes",
    "connects",
    "creates",
    "depends on",
    "drives",
    "enables",
    "feeds",
    "follows",
    "if ",
    "leads to",
    "links",
    "maps to",
    "reads",
    "requires",
    "results in",
    "stores",
    "then",
    "updates",
    "uses",
    "when ",
    "writes",
)

MAX_PREP_KEYWORDS = 40
MAX_TOPIC_TAGS = 50


@dataclass(frozen=True)
class ParsedSection:
    """A markdown section before canonicalization."""

    heading: str
    level: int
    body: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSection:
    """A section assigned to a canonical training bucket."""

    bucket: str
    original_heading: str
    body: str


@dataclass(frozen=True)
class MarkdownPreparationConfig:
    """Configuration for markdown preparation."""

    min_section_words: int = 20
    include_relationship_summary: bool = True
    include_frontmatter_metadata: bool = True


@dataclass(frozen=True)
class MarkdownPreparationResult:
    """Prepared markdown plus transformation metadata."""

    title: str
    output_text: str
    source_sections: int
    prepared_sections: int
    synthesized_sections: tuple[str, ...]
    relational_sentences: tuple[str, ...]
    topic_tags: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TrainingAnnotations:
    """Structured training annotations carried in markdown frontmatter."""

    title: str
    topic_tags: tuple[str, ...]
    labels: tuple[str, ...]


def prepare_markdown_for_training(
    text: str,
    *,
    source_name: str = "",
    config: MarkdownPreparationConfig | None = None,
) -> MarkdownPreparationResult:
    """Convert markdown text into a trainer-friendly canonical template."""
    prep_config = config or MarkdownPreparationConfig()
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    frontmatter, body_text = _split_frontmatter(normalized_text)
    parsed_sections = _parse_sections(body_text)
    title = _derive_title(parsed_sections, frontmatter, source_name)

    prepared_sections: list[PreparedSection] = []
    for section in parsed_sections:
        body = section.body.strip()
        if not body:
            continue
        bucket = _bucket_for_heading(section.heading, section.heading_path)
        if section.level == 1 and section.heading == title and bucket == "Reference":
            bucket = "Overview"
        prepared_sections.append(
            PreparedSection(
                bucket=bucket,
                original_heading=section.heading,
                body=body,
            )
        )

    synthesized_sections: list[str] = []
    relational_sentences = _extract_relational_sentences(body_text)
    if prep_config.include_relationship_summary and relational_sentences:
        relationship_summary = _build_relationship_summary(
            title=title,
            sections=parsed_sections,
            relational_sentences=relational_sentences,
        )
        prepared_sections.append(
            PreparedSection(
                bucket="Relationships",
                original_heading="Derived Link Summary",
                body=relationship_summary,
            )
        )
        synthesized_sections.append("Relationships")

    metadata_block = _frontmatter_metadata_block(frontmatter)
    if prep_config.include_frontmatter_metadata and metadata_block:
        prepared_sections.append(
            PreparedSection(
                bucket="Reference",
                original_heading="Frontmatter Metadata",
                body=metadata_block,
            )
        )
        synthesized_sections.append("Reference")

    overview_needed = not any(section.bucket == "Overview" for section in prepared_sections)
    if overview_needed:
        overview_body = _build_overview(title=title, parsed_sections=parsed_sections, source_name=source_name)
        prepared_sections.append(
            PreparedSection(
                bucket="Overview",
                original_heading="Document Focus",
                body=overview_body,
            )
        )
        synthesized_sections.append("Overview")

    labels = _derive_labels(prepared_sections)
    topic_tags = _derive_topic_tags(
        title=title,
        parsed_sections=parsed_sections,
        source_name=source_name,
    )
    output_text = _render_prepared_markdown(
        title=title,
        sections=prepared_sections,
        min_section_words=prep_config.min_section_words,
        topic_tags=topic_tags,
        labels=labels,
    )
    return MarkdownPreparationResult(
        title=title,
        output_text=output_text,
        source_sections=len(parsed_sections),
        prepared_sections=len(prepared_sections),
        synthesized_sections=tuple(dict.fromkeys(synthesized_sections)),
        relational_sentences=tuple(relational_sentences),
        topic_tags=topic_tags,
        labels=labels,
    )


def extract_training_annotations(text: str, *, source_name: str = "") -> TrainingAnnotations:
    """Read training frontmatter from markdown or derive a minimal fallback."""
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    frontmatter, body_text = _split_frontmatter(normalized_text)
    parsed_sections = _parse_sections(body_text)
    title = _derive_title(parsed_sections, frontmatter, source_name)

    frontmatter_tags = _parse_csv_values(frontmatter.get("nm_tags", ""))
    frontmatter_labels = _parse_csv_values(frontmatter.get("nm_labels", ""))
    if frontmatter_tags or frontmatter_labels:
        return TrainingAnnotations(
            title=title,
            topic_tags=tuple(frontmatter_tags),
            labels=tuple(frontmatter_labels),
        )

    fallback_labels = tuple(
        label.lower()
        for label in _derive_labels(
            [
                PreparedSection(
                    bucket=_bucket_for_heading(section.heading, section.heading_path),
                    original_heading=section.heading,
                    body=section.body,
                )
                for section in parsed_sections
                if section.body.strip()
            ]
        )
    )
    fallback_tags = _derive_topic_tags(
        title=title,
        parsed_sections=parsed_sections,
        source_name=source_name,
    )
    return TrainingAnnotations(
        title=title,
        topic_tags=fallback_tags,
        labels=fallback_labels,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract simple YAML frontmatter key-value pairs."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata, text[match.end() :]


def _parse_csv_values(value: str) -> list[str]:
    """Parse a comma-separated frontmatter value into normalized strings."""
    if not value.strip():
        return []
    items = [part.strip() for part in value.split(",")]
    return [item for item in items if item]


def _derive_title(
    sections: list[ParsedSection],
    frontmatter: dict[str, str],
    source_name: str,
) -> str:
    """Pick a stable title from headings, metadata, or file name."""
    for section in sections:
        if section.level == 1 and section.heading:
            return section.heading
    if "title" in frontmatter and frontmatter["title"]:
        return frontmatter["title"]
    if source_name:
        stem = Path(source_name).stem
        if stem:
            return stem.replace("-", " ").replace("_", " ").title()
    return "Prepared Training Document"


def _bucket_for_heading(heading: str, heading_path: tuple[str, ...]) -> str:
    """Map a heading to a canonical trainer bucket."""
    if not heading:
        return "Overview"

    heading_key = _normalize_heading_key(heading)
    for bucket, keywords in _HEADING_BUCKETS:
        if any(keyword in heading_key for keyword in keywords):
            return bucket

    if len(heading_path) >= 2:
        parent_key = _normalize_heading_key(heading_path[-2])
        for bucket, keywords in _HEADING_BUCKETS:
            if any(keyword in parent_key for keyword in keywords):
                return bucket

    return "Reference"


def _normalize_heading_key(heading: str) -> str:
    """Normalize heading text for keyword matching."""
    collapsed = _WHITESPACE_RE.sub(" ", heading.lower()).strip()
    return collapsed


def _parse_sections(text: str) -> list[ParsedSection]:
    """Parse markdown into sections while respecting fenced code blocks."""
    lines = text.split("\n")
    sections: list[tuple[int, str, str, tuple[str, ...]]] = []
    body_lines: list[str] = []
    current_heading = ""
    current_level = 0
    heading_stack: list[tuple[int, str]] = []
    in_fence = False
    fence_marker = ""

    def finalize_current() -> None:
        nonlocal body_lines
        body = "\n".join(body_lines).strip()
        if current_heading or body:
            heading_path = _build_heading_path(current_level, current_heading, heading_stack)
            sections.append((current_level, current_heading, body, heading_path))
        body_lines = []

    index = 0
    while index < len(lines):
        line = lines[index]
        next_in_fence, next_fence_marker = _toggle_fence(line, in_fence, fence_marker)
        if next_in_fence != in_fence or next_fence_marker != fence_marker:
            in_fence, fence_marker = next_in_fence, next_fence_marker
            body_lines.append(line)
            index += 1
            continue

        if not in_fence:
            atx_match = _ATX_HEADING_RE.match(line)
            if atx_match:
                finalize_current()
                current_level = len(atx_match.group(1))
                current_heading = atx_match.group(2).strip()
                heading_stack = _updated_heading_stack(
                    heading_stack, current_level, current_heading
                )
                index += 1
                continue

            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if next_line is not None and line.strip():
                if _SETEXT_H1_RE.match(next_line):
                    finalize_current()
                    current_level = 1
                    current_heading = line.strip()
                    heading_stack = _updated_heading_stack(
                        heading_stack, current_level, current_heading
                    )
                    index += 2
                    continue
                if _SETEXT_H2_RE.match(next_line):
                    finalize_current()
                    current_level = 2
                    current_heading = line.strip()
                    heading_stack = _updated_heading_stack(
                        heading_stack, current_level, current_heading
                    )
                    index += 2
                    continue

        body_lines.append(line)
        index += 1

    finalize_current()

    parsed: list[ParsedSection] = []
    for level, heading, body, heading_path in sections:
        parsed.append(
            ParsedSection(
                heading=heading,
                level=level,
                body=body,
                heading_path=heading_path,
            )
        )
    return parsed


def _toggle_fence(line: str, in_fence: bool, fence_marker: str) -> tuple[bool, str]:
    """Return updated fence state after reading a line."""
    match = _FENCE_OPEN_RE.match(line)
    if not match:
        return in_fence, fence_marker

    marker = match.group(1)
    if not in_fence:
        return True, marker[0]
    if line.lstrip().startswith(fence_marker * 3):
        return False, ""
    return in_fence, fence_marker


def _updated_heading_stack(
    heading_stack: list[tuple[int, str]],
    level: int,
    heading: str,
) -> list[tuple[int, str]]:
    """Return a new heading stack with the current heading applied."""
    new_stack = list(heading_stack)
    while new_stack and new_stack[-1][0] >= level:
        new_stack.pop()
    new_stack.append((level, heading))
    return new_stack


def _build_heading_path(
    level: int,
    heading: str,
    heading_stack: list[tuple[int, str]],
) -> tuple[str, ...]:
    """Derive heading path for the current section."""
    if level == 0 or not heading:
        return ()
    return tuple(item_heading for _, item_heading in heading_stack)


def _extract_relational_sentences(text: str, *, limit: int = 6) -> list[str]:
    """Collect sentences that explicitly express relationships or causality."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    matches: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        cleaned = _WHITESPACE_RE.sub(" ", sentence).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(cue in lowered for cue in _RELATIONAL_CUES):
            if cleaned not in seen:
                matches.append(cleaned)
                seen.add(cleaned)
        if len(matches) >= limit:
            break
    return matches


def _build_relationship_summary(
    *,
    title: str,
    sections: list[ParsedSection],
    relational_sentences: list[str],
) -> str:
    """Render a compact relationship summary that survives chunk filtering."""
    lines = [
        f"This section extracts the strongest links, dependencies, and sequences described in {title}.",
        "",
    ]
    for sentence in relational_sentences:
        lines.append(f"- {sentence}")

    hierarchy_notes = _hierarchy_relationship_notes(sections)
    for note in hierarchy_notes:
        lines.append(f"- {note}")
    return "\n".join(lines).strip()


def _hierarchy_relationship_notes(
    sections: list[ParsedSection],
    *,
    limit: int = 4,
) -> list[str]:
    """Translate heading hierarchy into explicit relationship sentences."""
    notes: list[str] = []
    for section in sections:
        if len(section.heading_path) >= 2 and section.heading:
            parent = section.heading_path[-2]
            child = section.heading_path[-1]
            notes.append(f"{child} is organized under {parent} in the source document.")
        if len(notes) >= limit:
            break
    return notes


def _frontmatter_metadata_block(frontmatter: dict[str, str]) -> str:
    """Convert frontmatter metadata into visible markdown content."""
    if not frontmatter:
        return ""
    lines = [
        "This metadata was lifted from frontmatter so it remains visible after markdown chunking.",
        "",
    ]
    for key, value in sorted(frontmatter.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _build_overview(
    *,
    title: str,
    parsed_sections: list[ParsedSection],
    source_name: str,
) -> str:
    """Synthesize an overview when the source file does not have one."""
    candidate_parts: list[str] = []
    for section in parsed_sections:
        body = _collapse_whitespace(section.body)
        if body:
            candidate_parts.append(body)
        if sum(len(part.split()) for part in candidate_parts) >= 50:
            break

    lead_text = " ".join(candidate_parts[:2]).strip()
    if lead_text:
        clipped = _clip_words(lead_text, 55)
        return (
            f"This prepared training document focuses on {title}. "
            f"The source material emphasizes the following context: {clipped}"
        ).strip()

    if source_name:
        return (
            f"This prepared training document focuses on {title} and was converted from "
            f"{source_name} so the trainer receives explicit structure and enough context "
            "to build durable graph links."
        )

    return (
        f"This prepared training document focuses on {title} and adds an explicit overview "
        "so the trainer has a stable starting point for chunking and link creation."
    )


def _derive_labels(sections: list[PreparedSection]) -> tuple[str, ...]:
    """Create coarse-grained labels from canonical sections."""
    labels = {"trainer-ready"}
    for section in sections:
        labels.add(section.bucket.lower())
    return tuple(sorted(labels))


def _derive_topic_tags(
    *,
    title: str,
    parsed_sections: list[ParsedSection],
    source_name: str,
) -> tuple[str, ...]:
    """Generate normalized topic tags from title, headings, and keywords."""
    tag_normalizer = TagNormalizer()
    candidates: set[str] = set()

    candidates.update(_heading_to_tags(title))
    for section in parsed_sections:
        if section.heading:
            candidates.update(_heading_to_tags(section.heading))

    keyword_source = "\n".join(
        [title]
        + [section.heading for section in parsed_sections if section.heading]
        + [section.body for section in parsed_sections if section.body.strip()]
    )
    for weighted_keyword in extract_weighted_keywords(keyword_source, language="auto")[
        :MAX_PREP_KEYWORDS
    ]:
        candidates.update(_heading_to_tags(weighted_keyword.text))

    if source_name:
        candidates.update(_heading_to_tags(Path(source_name).stem))

    normalized = tag_normalizer.normalize_set({candidate for candidate in candidates if candidate})
    filtered = [tag for tag in sorted(normalized) if len(tag) >= 2]
    return tuple(filtered[:MAX_TOPIC_TAGS])


def _heading_to_tags(text: str) -> set[str]:
    """Convert heading-like text into candidate tag tokens."""
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ỹ\s_-]+", " ", text.lower())
    parts = [part.strip("-_ ") for part in re.split(r"[\s/_-]+", cleaned) if part.strip("-_ ")]
    candidates: set[str] = set()
    if not parts:
        return candidates

    candidates.add("-".join(parts))
    for part in parts:
        if len(part) >= 2:
            candidates.add(part)
    if len(parts) >= 2:
        candidates.add(" ".join(parts[:2]))
    return candidates


def _render_prepared_markdown(
    *,
    title: str,
    sections: list[PreparedSection],
    min_section_words: int,
    topic_tags: tuple[str, ...],
    labels: tuple[str, ...],
) -> str:
    """Render canonical markdown with merged short sections."""
    grouped: dict[str, list[PreparedSection]] = {bucket: [] for bucket in CANONICAL_SECTIONS}
    for section in sections:
        grouped.setdefault(section.bucket, []).append(section)

    lines = _render_training_frontmatter(title=title, topic_tags=topic_tags, labels=labels)
    lines.extend([f"# {title}", ""])
    for bucket in CANONICAL_SECTIONS:
        bucket_sections = grouped.get(bucket, [])
        if not bucket_sections:
            continue

        lines.append(f"## {bucket}")
        lines.append("")
        short_notes: list[str] = []
        for section in bucket_sections:
            body = section.body.strip()
            if not body:
                continue

            if _should_merge_short_section(body, min_section_words):
                label = section.original_heading or "Note"
                short_notes.append(f"- {label}: {_collapse_whitespace(body)}")
                continue

            if section.original_heading and section.original_heading not in {bucket, title}:
                lines.append(f"### {section.original_heading}")
                lines.append("")
            lines.append(body)
            lines.append("")

        if short_notes:
            lines.append("### Collected Notes")
            lines.append("")
            lines.append(
                "These shorter notes were grouped together so they remain above the "
                "trainer's minimum chunk threshold and still contribute useful links."
            )
            lines.append("")
            lines.extend(short_notes)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_training_frontmatter(
    *,
    title: str,
    topic_tags: tuple[str, ...],
    labels: tuple[str, ...],
) -> list[str]:
    """Render deterministic frontmatter for downstream training."""
    lines = ["---", f"title: {title}"]
    if topic_tags:
        lines.append(f"nm_tags: {', '.join(topic_tags)}")
    if labels:
        lines.append(f"nm_labels: {', '.join(labels)}")
    lines.extend(["---", ""])
    return lines


def _should_merge_short_section(body: str, min_section_words: int) -> bool:
    """Decide whether a section is too short to stand alone."""
    if "```" in body or "~~~" in body:
        return False
    return len(body.split()) < min_section_words


def _collapse_whitespace(text: str) -> str:
    """Collapse internal whitespace without stripping markdown punctuation."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _clip_words(text: str, limit: int) -> str:
    """Limit text to a maximum number of words."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(".,;:") + "..."
