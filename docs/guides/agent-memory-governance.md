# Agent Memory Governance

NeuralMemory is most useful when agents store durable context, not every passing detail. Use these guidelines to keep memory useful, auditable, and safe.

## Store Durable Signals

Good memories help a future agent make a better decision.

Store:

- Project decisions and the reason they were made
- User preferences that should change future behavior
- Repeated workflow constraints
- Confirmed facts about the current project
- Resolutions to errors that are likely to recur
- Handoffs, open questions, and next actions

Avoid storing:

- Raw command output unless the exact output is needed later
- Temporary status updates that will be stale in hours
- Speculation without a clear uncertainty marker
- Large copied documents when a source reference is enough
- Secrets, tokens, credentials, private keys, or regulated personal data

## Mark Confidence And Source

When a memory may influence later work, include enough context to judge it.

Prefer:

```bash
nmem remember "Decision: keep update notices out of JSON output because agents parse CLI stdout as structured context. Source: issue triage on 2026-04-30." --type decision
```

Over:

```bash
nmem remember "JSON output is fixed"
```

Useful details include:

- Source: issue, PR, commit, file, meeting, or user instruction
- Date: when the statement was observed or decided
- Confidence: confirmed, inferred, tentative, or needs-review
- Scope: project, branch, user, environment, or tool

## Keep Machine Output Clean

Agents often feed CLI and MCP output back into prompts. Human-facing banners, advisory hints, and package update notices can pollute summaries.

For automation:

```bash
nmem context --json
nmem status --json
nmem stats --json
nmem recall "deployment constraints" --json
```

For MCP clients, use compact responses when the client supports it:

```json
{
  "query": "deployment constraints",
  "compact": true
}
```

By default, MCP response hint fields such as `maintenance_hint`, `update_hint`, and `onboarding` are controlled by:

```toml
[response]
strip_hints = true
```

Set `strip_hints = false` only when an interactive client should surface those advisory fields directly to the user.

## Review Before Writing

Before saving a memory, ask:

- Will this still matter next week?
- Is the source or reason clear?
- Could this be harmful if recalled out of context?
- Does it contain sensitive or private data?
- Should this be attached to a specific project brain instead of a global brain?

Use `nmem check` for sensitive content before storing uncertain text:

```bash
nmem check "candidate memory text"
```

## Correct And Retire Stale Memories

Treat memory as a maintained project artifact. When facts change, add a correcting memory with the new source and date. For stale or harmful content, edit, forget, or use lifecycle controls rather than relying on future agents to infer that it is obsolete.

Useful review commands:

```bash
nmem status
nmem stats
nmem context --fresh-only
nmem recall "old decision" --show-age
```

## Suggested Agent Policy

For coding agents, a conservative default policy is:

- Store decisions, constraints, and confirmed user preferences.
- Store fixes only when the cause and resolution are clear.
- Do not store secrets or raw logs.
- Do not store every command result.
- Prefer project-specific brains for project facts.
- Use JSON or compact MCP output when piping memory back into prompts.
