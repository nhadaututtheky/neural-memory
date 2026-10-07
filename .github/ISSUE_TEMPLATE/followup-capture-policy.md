---
name: CapturePolicy Base Class
about: Extract language-specific capture logic for extensibility (JP/KR support)
title: "[Architecture] CapturePolicy base class for multi-language auto-capture"
labels: architecture, mcp, i18n
assignees: ''
---

## Problem
PR #209 added Chinese support by adding CJK-specific logic directly in `auto_capture.py`:
- Separate `_detect_patterns` branch for CJK
- CJK-specific thresholds (8 chars vs 20 Latin)
- No "too short" penalty for CJK
- Conservative triggers (explicit markers required)

Vietnamese already had `_vi_quality_gate` with its own tokenizer. When Japanese/Korean support lands, we'll copy-paste this pattern → technical debt.

## Solution: CapturePolicy Base Class

```python
# src/neural_memory/mcp/capture_policy.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PatternMatch:
    type: str           # decision, error, todo, fact, preference, insight
    content: str
    confidence: float
    start: int
    end: int
    metadata: dict

class CapturePolicy(ABC):
    """Language-specific capture policy."""
    
    @property
    @abstractmethod
    def language_code(self) -> str: ...
    
    @property
    @abstractmethod
    def min_input_length(self) -> int: ...      # Latin: 20, CJK: 8
    
    @property
    @abstractmethod
    def min_capture_length(self) -> int: ...    # Latin: 5-15, CJK: 4
    
    @abstractmethod
    def detect_patterns(self, text: str) -> list[PatternMatch]: ...
    
    @abstractmethod
    def quality_gate(self, match: PatternMatch) -> float: ...
    
    def empty_hint(self, text: str) -> str | None:
        """Return hint for empty results, or None if not applicable."""
        return None

# Registry
POLICY_REGISTRY: dict[str, CapturePolicy] = {}

def register_policy(policy: CapturePolicy) -> None:
    POLICY_REGISTRY[policy.language_code] = policy

def detect_language(text: str) -> str:
    """Detect primary script: 'en', 'vi', 'zh', 'ja', 'ko'."""
    ...

def get_policy(text: str) -> CapturePolicy:
    lang = detect_language(text)
    return POLICY_REGISTRY.get(lang, POLICY_REGISTRY['en'])
```

## Implementation Plan
1. **Create `capture_policy.py`** with base class + registry
2. **Extract existing logic**:
   - `LatinPolicy` (current English default)
   - `VietnamesePolicy` (existing `_vi_quality_gate` + tokenizer)
   - `CJKPolicy` (PR #209 logic)
3. **Refactor `auto_capture.py`** to use `get_policy(text).detect_patterns(text)`
4. **Add tests** for each policy + registry auto-detection

## Japanese/Korean Ready
```python
class JapanesePolicy(CJKPolicy):
    language_code = "ja"
    # Override patterns: 判断/決定/TODO/エラー/気づき/事実/好み
    # Particle trimming: です/ます/だ/である → strip
    
class KoreanPolicy(CJKPolicy):
    language_code = "ko"
    # Override patterns: 결정/오류/할일/사실/선호/통찰
    # Particle trimming: ~다/~이다/~함 → strip
```

## Acceptance Criteria
- [ ] `auto_capture.py` uses `CapturePolicy` registry
- [ ] English/Vietnamese/Chinese behavior **unchanged** (regression tests pass)
- [ ] Adding new language = 1 new class + register, no core changes
- [ ] `pre_ship.py` passes
- [ ] Documentation: `docs/guides/adding-language-support.md`