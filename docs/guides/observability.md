# Observability

`GET /metrics` reports the silent failures fixed by the SQLite write queue,
the CLI encoding guard, and empty auto-capture hints. Install the server
extra, which includes `prometheus-client`:

```bash
pip install "neural-memory[server]"
nmem serve
```

`nmem serve` binds `127.0.0.1` by default. Scrape `http://127.0.0.1:8000/metrics`.
The path is public, like `/health`, so a local scraper works while an API key
is set. The body is counters and a latency histogram. It has no brain names
and no memory text. A process without `prometheus-client` still answers the
route with a one-line comment, and every recorder is a no-op.

## Series

| Name | Labels | When it moves |
|------|--------|----------------|
| `neural_memory_sqlite_write_latency_seconds` | `dialect="sqlite"` | Every SQLite write. The sample includes queue wait and execution. |
| `neural_memory_sqlite_write_collision_total` | `dialect="sqlite"` | The driver reports `SQL statements in progress`. The single-writer queue keeps this at zero. |
| `neural_memory_cli_encoding_damage_rejected_total` | `command`, `damage_type` | `nmem remember` refuses damaged text. `damage_type` is `question_mark_flood`, `unicode_escape`, or `replacement_char`. `--force` stores the text and does not increment. |
| `neural_memory_auto_capture_empty_total` | `reason`, `language` | Detection found nothing. `reason` is `too_short`, `cjk_unsupported`, or `no_trigger`. `language` is `zh`, `ja`, `ko`, `vi`, or `en`. |

`command` is limited to `remember`, `forget`, and `list`. A value outside a
fixed set is recorded as `other`. The client also emits a `_created` gauge
per series. Chart the names in the table.

Forbidden-pattern checks stay in `scripts/pre_ship.py`. That run fails the
build in CI. It is not a runtime series.

## Grafana

Import `docs/grafana/neural-memory-silent-failures.json` and point it at the
Prometheus datasource. Load `docs/grafana/neural-memory-alerts.yml` as a
Prometheus rule file.

The collision counter is cumulative, so a single historical event would leave
`neural_memory_sqlite_write_collision_total > 0` forever. The shipped rule
fires when the counter increases inside five minutes:

```promql
increase(neural_memory_sqlite_write_collision_total[5m]) > 0
```

Severity is warning. An increase means a commit collision reached the driver
again.
