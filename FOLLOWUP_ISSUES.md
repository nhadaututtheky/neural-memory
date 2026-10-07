# Follow-up Issues Created

| Issue | Template | Priority | Description |
|-------|----------|----------|-------------|
| Dialect-aware write strategy | `followup-sqlite-dialect-strategy.md` | **High** | Only serialize on SQLite; use pool on Postgres |
| CapturePolicy base class | `followup-capture-policy.md` | **Medium** | Extract language logic for JP/KR extensibility |
| Windows native CLI | `followup-windows-native-cli.md` | **Medium** | Rust/Go binary for proper UTF-8 pipe handling |
| Metrics & observability | `followup-metrics-observability.md` | **Medium** | Prometheus counters for silent failures |

## Files Created

```
.github/ISSUE_TEMPLATE/
├── followup-sqlite-dialect-strategy.md
├── followup-capture-policy.md
├── followup-windows-native-cli.md
└── followup-metrics-observability.md

scripts/benchmark/
└── serialize_impact.py
```

## Next Actions

1. **Create GitHub Issues** from these templates:
   ```bash
   gh issue create --title "[Strategy] Dialect-aware write serialization" --body-file .github/ISSUE_TEMPLATE/followup-sqlite-dialect-strategy.md
   gh issue create --title "[Architecture] CapturePolicy base class" --body-file .github/ISSUE_TEMPLATE/followup-capture-policy.md
   gh issue create --title "[Windows] Native CLI binary" --body-file .github/ISSUE_TEMPLATE/followup-windows-native-cli.md
   gh issue create --title "[Observability] Metrics for silent failures" --body-file .github/ISSUE_TEMPLATE/followup-metrics-observability.md
   ```

2. **Run benchmark** to quantify Postgres impact:
   ```bash
   POSTGRES_DSN=postgresql://... python scripts/benchmark/serialize_impact.py
   ```

3. **Assign to sprint** based on priority