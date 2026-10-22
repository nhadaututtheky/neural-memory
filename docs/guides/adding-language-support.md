# Adding a capture language

Auto-capture stores decisions, errors, todos, facts, insights, and
preferences. English, Vietnamese, and Chinese each have a
`CapturePolicy` in `src/neural_memory/mcp/capture_policy.py`. Japanese
and Korean are recognized and reported as unsupported. They have no
patterns yet.

`analyze_text_for_memories` asks `get_policy(text)` for the input floor
(20 characters for Latin and Vietnamese, 8 for Chinese). It then runs
**every registered policy**, category by category: decision, error,
todo, fact, insight, preference. Inside a category the order is the
registry order (`en`, `vi`, `zh`). A sentence that mixes languages
keeps a match from each policy. Dedup still happens once, after the
scan.

`get_policy(text).detect_patterns(text)` scans only that language. Use
it when a caller already knows the script. The auto-capture entry point
does not, because dropping the other policies would change mixed text.

Auto-save triggers in `core/trigger_engine.py` are a separate detector.
A capture policy does not fire them.

## Add a language

Subclass `CapturePolicy`, supply `specs()`, and register the instance.
No edit to `auto_capture.py` is required.

```python
from neural_memory.mcp.capture_policy import (
    ChinesePolicy,
    register_policy,
)

class JapanesePolicy(ChinesePolicy):
    language_code = "ja"

    def specs(self):
        return _JAPANESE_SPECS  # same six groups, Japanese markers


register_policy(JapanesePolicy())
```

`ChinesePolicy` already strips trailing particles and uses the short
CJK length floor. A Latin-like language subclasses `LatinPolicy`
instead and inherits the 20-character input floor. Override
`quality_gate` to return `0.0` when a capture should be dropped, or a
multiplier such as Vietnamese `0.55`. Override `empty_hint` when an
empty result needs a language-specific explanation.

Patterns should require an explicit marker. A descriptive verb in
ordinary prose will be stored as a memory at high confidence.

Until a Japanese or Korean policy is registered, `detect_language`
still returns `ja` or `ko`, `get_policy` falls back to Latin for the
length floor, and the empty-result hint says the script is not
supported.
