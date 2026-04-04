"""Security scanner for .brain package files.

Scans brain packages for prompt injection, dangerous commands,
malicious content, and structural issues before import/export.

Three gates use this scanner:
1. Export gate (author-side): blocks if risk >= high
2. Import gate (consumer-side): always scans, hard-blocks critical
3. Registry gate (GitHub Action): rejects critical/high PRs
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from neural_memory.safety.sensitive import check_sensitive_content

# ── Scan Result Types ────────────────────────────────────────────


@dataclass(frozen=True)
class ScanFinding:
    """A single security finding from a brain scan."""

    category: (
        str  # prompt_injection | dangerous_command | malicious_content | structural | sensitive
    )
    severity: str  # info | low | medium | high | critical
    description: str
    location: str  # neuron:{id} | fiber:{id} | synapse:{id} | manifest | metadata
    matched_pattern: str


@dataclass(frozen=True)
class BrainScanResult:
    """Aggregated result of scanning a brain package."""

    safe: bool
    risk_level: str  # clean | low | medium | high | critical
    findings: list[ScanFinding]
    neurons_scanned: int
    scan_duration_ms: float


# ── Severity Ranking ─────────────────────────────────────────────

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_RANK_TO_RISK = {0: "clean", 1: "low", 2: "medium", 3: "high", 4: "critical"}


def _aggregate_risk(findings: list[ScanFinding]) -> str:
    """Compute overall risk level from the highest severity finding."""
    if not findings:
        return "clean"
    max_rank = max(_SEVERITY_RANK.get(f.severity, 0) for f in findings)
    return _RANK_TO_RISK.get(max_rank, "medium")


# ── Prompt Injection Patterns ────────────────────────────────────

_PROMPT_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Role hijacking
    (
        re.compile(
            r"(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+)?(?:previous|prior|above|earlier|your|the)\s+"
            r"(?:instructions?|rules?|guidelines?|prompts?|context|constraints?|directives?)",
            re.IGNORECASE,
        ),
        "critical",
        "Role hijacking: instruction override attempt",
    ),
    (
        re.compile(
            r"you\s+are\s+now\s+(?:a|an|the|my)\s+\w+",
            re.IGNORECASE,
        ),
        "critical",
        "Role hijacking: identity reassignment",
    ),
    (
        re.compile(
            r"(?:new|updated|revised|real)\s+system\s+prompt",
            re.IGNORECASE,
        ),
        "critical",
        "Role hijacking: system prompt replacement",
    ),
    (
        re.compile(
            r"(?:act|behave|pretend|respond)\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)\s+)?(?:a|an|the|my)\s+",
            re.IGNORECASE,
        ),
        "high",
        "Role hijacking: persona injection",
    ),
    # Instruction manipulation
    (
        re.compile(
            r"do\s+not\s+follow\s+(?:any|the|your|previous)\s+",
            re.IGNORECASE,
        ),
        "critical",
        "Instruction manipulation: directive negation",
    ),
    (
        re.compile(
            r"(?:jailbreak|escape|break\s+free|unlock)\s+(?:mode|your|the|from)",
            re.IGNORECASE,
        ),
        "critical",
        "Jailbreak attempt",
    ),
    # Delimiter attacks — context boundary manipulation
    (
        re.compile(
            r"(?:^|\n)(?:<\|(?:im_start|im_end|system|endoftext)\|>)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "critical",
        "Delimiter attack: LLM special token injection",
    ),
    (
        re.compile(
            r"(?:^|\n)(?:###\s*(?:SYSTEM|INSTRUCTION|HUMAN|ASSISTANT|USER)\s*(?::|###))",
            re.IGNORECASE | re.MULTILINE,
        ),
        "high",
        "Delimiter attack: role boundary marker",
    ),
    # Hidden instructions via encoding
    (
        re.compile(
            r"(?:decode|base64|atob|btoa)\s*\(\s*['\"](?:[A-Za-z0-9+/]{20,}={0,2})['\"]",
            re.IGNORECASE,
        ),
        "high",
        "Encoded payload: base64 decode call with embedded data",
    ),
]

# ── Dangerous Command Patterns ───────────────────────────────────

_DANGEROUS_COMMAND_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Destructive shell commands
    (
        re.compile(
            r"(?:sudo\s+)?rm\s+(?:-\w*[rf]\w*\s+)+(?:/|~|\.\.|[*])",
            re.IGNORECASE,
        ),
        "high",
        "Destructive command: recursive file deletion",
    ),
    (
        re.compile(
            r"(?:curl|wget)\s+\S+\s*\|\s*(?:ba)?sh",
            re.IGNORECASE,
        ),
        "critical",
        "Piped remote execution: curl/wget to shell",
    ),
    (
        re.compile(
            r"(?:^|\s)(?:chmod\s+(?:777|666|a\+rwx))",
            re.IGNORECASE,
        ),
        "medium",
        "Insecure permissions: world-writable",
    ),
    # Database destructive
    (
        re.compile(
            r"(?:DROP\s+(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE|DELETE\s+FROM\s+\w+\s*(?:;|$))",
            re.IGNORECASE,
        ),
        "high",
        "Destructive SQL: data deletion command",
    ),
    # Code execution
    (
        re.compile(
            r"(?:^|\s)(?:eval|exec|compile|__import__)\s*\(",
            re.IGNORECASE,
        ),
        "high",
        "Dynamic code execution call",
    ),
    # Windows destructive
    (
        re.compile(
            r"(?:format\s+[a-zA-Z]:|del\s+/[sfq]\s|rd\s+/s)",
            re.IGNORECASE,
        ),
        "high",
        "Destructive Windows command",
    ),
]

# ── Malicious Content Patterns ───────────────────────────────────

_MALICIOUS_CONTENT_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # HTML/script injection
    (
        re.compile(
            r"<script\b[^>]*>|<iframe\b[^>]*>|<object\b[^>]*>|<embed\b[^>]*>",
            re.IGNORECASE,
        ),
        "high",
        "HTML injection: executable tag",
    ),
    (
        re.compile(
            r"(?:on(?:error|load|click|mouseover|focus|submit))\s*=\s*['\"]",
            re.IGNORECASE,
        ),
        "high",
        "HTML injection: event handler attribute",
    ),
    (
        re.compile(
            r"javascript\s*:|data\s*:\s*text/html",
            re.IGNORECASE,
        ),
        "high",
        "URI injection: javascript/data protocol",
    ),
    # Credential harvesting instructions
    (
        re.compile(
            r"(?:send|post|share|leak|exfiltrate|upload|forward)\s+"
            r"(?:your|the|all|any|user'?s?)?\s*"
            r"(?:api[_\s]?key|secret|token|password|credential|private[_\s]?key|ssh[_\s]?key)",
            re.IGNORECASE,
        ),
        "critical",
        "Credential harvesting: instruction to exfiltrate secrets",
    ),
    (
        re.compile(
            r"(?:send|post|share|forward)\s+(?:to|at)\s+\S+\s+"
            r"(?:api[_\s]?key|secret|token|password|credential)",
            re.IGNORECASE,
        ),
        "critical",
        "Credential harvesting: exfiltration with target",
    ),
    # Social engineering
    (
        re.compile(
            r"(?:CRITICAL|URGENT|EMERGENCY|IMMEDIATE)\s*:\s*(?:run|execute|do|perform)\s+",
            re.IGNORECASE,
        ),
        "medium",
        "Social engineering: fake urgency pattern",
    ),
    (
        re.compile(
            r"(?:admin|administrator|root|owner)\s+(?:requires?|demands?|orders?|commands?)\s+",
            re.IGNORECASE,
        ),
        "medium",
        "Social engineering: authority abuse",
    ),
]

# ── Content Scanner ──────────────────────────────────────────────


_ZERO_WIDTH_RE = re.compile(
    r"[\u00a0\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5"
    r"\u180e\u2000-\u200f\u2028\u2029\u202a-\u202f"
    r"\u2060-\u2064\u2066-\u206f\u3000\u3164\ufeff\uffa0]"
)


def _normalize_text(text: str) -> str:
    """Normalize text to defeat Unicode bypass attacks.

    Applies NFKC normalization (maps homoglyphs to ASCII equivalents)
    and strips zero-width characters that can break regex matching.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return _ZERO_WIDTH_RE.sub("", normalized)


def _scan_text(
    text: str,
    location: str,
    findings: list[ScanFinding],
) -> None:
    """Scan a text string against all pattern categories."""
    if not text or not isinstance(text, str):
        return

    # Normalize to defeat Unicode bypass (homoglyphs + zero-width chars)
    # Cap scan length to prevent ReDoS on very large content
    scan_text = _normalize_text(text[:50_000])

    for pattern, severity, description in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(scan_text)
        if match:
            findings.append(
                ScanFinding(
                    category="prompt_injection",
                    severity=severity,
                    description=description,
                    location=location,
                    matched_pattern=match.group()[:100],
                )
            )

    for pattern, severity, description in _DANGEROUS_COMMAND_PATTERNS:
        match = pattern.search(scan_text)
        if match:
            findings.append(
                ScanFinding(
                    category="dangerous_command",
                    severity=severity,
                    description=description,
                    location=location,
                    matched_pattern=match.group()[:100],
                )
            )

    for pattern, severity, description in _MALICIOUS_CONTENT_PATTERNS:
        match = pattern.search(scan_text)
        if match:
            findings.append(
                ScanFinding(
                    category="malicious_content",
                    severity=severity,
                    description=description,
                    location=location,
                    matched_pattern=match.group()[:100],
                )
            )


def _scan_metadata(
    metadata: Any,
    location: str,
    findings: list[ScanFinding],
) -> None:
    """Scan metadata dict for hidden injections and oversized fields."""
    if not isinstance(metadata, dict):
        return

    serialized = str(metadata)

    # Check total metadata size
    if len(serialized) > 10_000:
        findings.append(
            ScanFinding(
                category="structural",
                severity="medium",
                description=f"Oversized metadata ({len(serialized)} chars)",
                location=location,
                matched_pattern=f"size={len(serialized)}",
            )
        )

    # Scan metadata values for injection
    _scan_text(serialized, f"{location}:metadata", findings)

    # Check for executable patterns in metadata
    for key, value in metadata.items():
        if isinstance(value, str) and any(
            kw in value for kw in ("__import__", "lambda ", "compile(", "exec(", "eval(")
        ):
            findings.append(
                ScanFinding(
                    category="structural",
                    severity="high",
                    description=f"Executable code in metadata key '{key}'",
                    location=location,
                    matched_pattern=value[:100],
                )
            )


# ── Structural Validation ────────────────────────────────────────


def _scan_structure(
    snapshot: dict[str, Any],
    findings: list[ScanFinding],
) -> None:
    """Validate brain snapshot structural integrity."""
    neurons = snapshot.get("neurons", [])
    synapses = snapshot.get("synapses", [])
    fibers = snapshot.get("fibers", [])

    # Collect all neuron IDs
    neuron_ids = set()
    for n in neurons:
        nid = n.get("id")
        if nid:
            neuron_ids.add(nid)

    # Check synapses
    for s in synapses:
        sid = s.get("id", "unknown")
        source = s.get("source_id")
        target = s.get("target_id")

        # Circular reference
        if source and target and source == target:
            findings.append(
                ScanFinding(
                    category="structural",
                    severity="medium",
                    description="Circular synapse: source equals target",
                    location=f"synapse:{sid}",
                    matched_pattern=f"{source} -> {target}",
                )
            )

        # Orphan references
        if source and source not in neuron_ids:
            findings.append(
                ScanFinding(
                    category="structural",
                    severity="low",
                    description="Orphan synapse: source neuron not in package",
                    location=f"synapse:{sid}",
                    matched_pattern=f"source={source}",
                )
            )
        if target and target not in neuron_ids:
            findings.append(
                ScanFinding(
                    category="structural",
                    severity="low",
                    description="Orphan synapse: target neuron not in package",
                    location=f"synapse:{sid}",
                    matched_pattern=f"target={target}",
                )
            )

    # Check fibers reference valid neurons
    for f in fibers:
        fid = f.get("id", "unknown")
        fiber_neuron_ids = f.get("neuron_ids", [])
        if isinstance(fiber_neuron_ids, (list, set)):
            for nid in fiber_neuron_ids:
                if nid not in neuron_ids:
                    findings.append(
                        ScanFinding(
                            category="structural",
                            severity="low",
                            description="Fiber references non-existent neuron",
                            location=f"fiber:{fid}",
                            matched_pattern=f"neuron={nid}",
                        )
                    )
                    break  # One warning per fiber is enough


# ── Sensitive Content Integration ────────────────────────────────


def _scan_sensitive(
    text: str,
    location: str,
    findings: list[ScanFinding],
) -> None:
    """Check for credentials, API keys, PII using existing safety module."""
    if not text or not isinstance(text, str):
        return

    # Normalize to catch homoglyph-substituted credentials
    normalized = _normalize_text(text)
    matches = check_sensitive_content(normalized, min_severity=2)
    for m in matches:
        findings.append(
            ScanFinding(
                category="sensitive",
                severity="high" if m.severity >= 3 else "medium",
                description=f"Sensitive content: {m.pattern_name}",
                location=location,
                matched_pattern=m.redacted()[:100],
            )
        )


# ── Main Scanner ─────────────────────────────────────────────────


def scan_brain_package(data: dict[str, Any]) -> BrainScanResult:
    """Scan a brain package for security threats.

    Checks four categories:
    1. Prompt injection — role hijacking, instruction override, delimiter attacks
    2. Dangerous commands — rm -rf, curl|sh, DROP TABLE, eval()
    3. Malicious content — HTML/script injection, credential harvesting, social engineering
    4. Structural — circular refs, orphan synapses, oversized metadata

    Also integrates sensitive content detection (API keys, passwords, PII).

    Args:
        data: Brain package dict with 'manifest' and 'snapshot' keys,
              or a raw BrainSnapshot dict.

    Returns:
        BrainScanResult with safe flag, risk level, and detailed findings.
    """
    start_time = time.monotonic()
    findings: list[ScanFinding] = []

    # Accept both full package and raw snapshot
    snapshot = data.get("snapshot", data)
    manifest = data.get("manifest", {})

    # Scan manifest fields
    for key in ("name", "display_name", "description", "author"):
        value = manifest.get(key, "")
        if value:
            _scan_text(str(value), f"manifest:{key}", findings)

    # Scan neurons
    neurons = snapshot.get("neurons", [])
    for neuron in neurons:
        nid = neuron.get("id", "unknown")
        content = neuron.get("content", "")
        location = f"neuron:{nid}"

        _scan_text(content, location, findings)
        _scan_sensitive(content, location, findings)
        _scan_metadata(neuron.get("metadata"), location, findings)

    # Scan fibers (summary, essence)
    fibers = snapshot.get("fibers", [])
    for fiber in fibers:
        fid = fiber.get("id", "unknown")
        location = f"fiber:{fid}"

        for text_field in ("summary", "essence"):
            text = fiber.get(text_field, "")
            if text:
                _scan_text(text, f"{location}:{text_field}", findings)

        _scan_metadata(fiber.get("metadata"), location, findings)

    # Scan synapse metadata
    synapses = snapshot.get("synapses", [])
    for synapse in synapses:
        sid = synapse.get("id", "unknown")
        _scan_metadata(synapse.get("metadata"), f"synapse:{sid}", findings)

    # Structural validation
    _scan_structure(snapshot, findings)

    elapsed_ms = (time.monotonic() - start_time) * 1000
    risk_level = _aggregate_risk(findings)

    return BrainScanResult(
        safe=risk_level in ("clean", "low"),
        risk_level=risk_level,
        findings=findings,
        neurons_scanned=len(neurons),
        scan_duration_ms=round(elapsed_ms, 2),
    )
