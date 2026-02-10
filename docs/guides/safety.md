# Safety & Security

Best practices and known risks when using NeuralMemory.

## Known Risks

### 1. Memory Poisoning

!!! danger "Risk Level: HIGH"
    Incorrect or malicious information stored in memory can propagate to future recalls.

**Examples:**

- Wrong API endpoint: future code may use wrong URL
- Incorrect attribution: "Bob said X" when Alice said it
- Outdated information: deprecated APIs, old contacts

**Mitigations:**

```bash
# Verify before storing
nmem check "Bob's phone: 555-1234"

# Use tags to mark confidence
nmem remember "Bob's phone: 555-1234" -t unverified -t needs-confirmation

# Include source attribution
nmem remember "Per email from Bob on 2024-02-04: API endpoint is /v2" --type fact

# Review health periodically
nmem brain health
```

**Best Practices:**

- Verify critical information before storing
- Use tags: `verified`, `unverified`, `needs-review`
- Include source attribution
- Periodically review and clean old memories

### 2. Stale Memory

!!! warning "Risk Level: MEDIUM"
    Old memories become outdated but still returned in queries.

**Examples:**

- Project decisions that have changed
- Former employee contacts
- Deprecated API versions
- Changed credentials

**Mitigations:**

```bash
# Check freshness
nmem stats

# Get only fresh context
nmem context --fresh-only

# Review health
nmem brain health

# Recall shows age warnings
nmem recall "API endpoint"
# ⚠️ STALE: This memory is 180 days old
```

**Freshness Levels:**

| Level | Age | Indicator | Action |
|-------|-----|-----------|--------|
| Fresh | < 7 days | 🟢 | Safe to use |
| Recent | 7-30 days | 🟢 | Generally safe |
| Aging | 30-90 days | 🟡 | Consider verifying |
| Stale | 90-365 days | 🟠 | Verify before using |
| Ancient | > 365 days | 🔴 | Likely outdated |

### 3. Privacy Leak

!!! danger "Risk Level: HIGH"
    Sensitive information can be accidentally exposed through exports, sharing, or logs.

**Examples:**

- API keys stored in memory
- Database credentials
- Personal information (SSN, credit cards)
- Private keys and tokens

**Mitigations:**

```bash
# Check content before storing
nmem check "AWS_SECRET_KEY=xxx"
# ⚠️ SENSITIVE CONTENT DETECTED

# Auto-redact sensitive content
nmem remember "Config: API_KEY=sk-xxx" --redact
# Stores: "Config: API_KEY=[REDACTED]"

# Export without sensitive content
nmem brain export --exclude-sensitive -o safe.json

# Scan imports
nmem brain import untrusted.json --scan
```

**Detected Patterns:**

- API keys and secrets
- Passwords
- AWS/Azure/GCP credentials
- Database URLs with credentials
- Private keys (PEM format)
- JWT tokens
- Credit card numbers
- Social Security Numbers

### 4. Over-reliance

!!! warning "Risk Level: MEDIUM"
    Blindly trusting memory output without verification.

**Mitigations:**

```bash
# Check confidence scores
nmem recall "critical config" --json | jq '.confidence'

# Set minimum threshold
nmem recall "important decision" --min-confidence 0.7

# Check memory age
nmem recall "api endpoint" --show-age
```

**Best Practices:**

- Treat memory as "hints" not "facts" for critical operations
- Always verify security-sensitive information
- Use confidence thresholds for automation
- Cross-reference with authoritative sources

## Security Best Practices

### 1. Data Classification

**DO Store:**

- Project decisions and rationale
- Meeting notes (non-confidential)
- Code patterns and solutions
- Error resolutions
- Workflow documentation

**DON'T Store:**

- Passwords and API keys
- Personal identification numbers
- Credit card or financial data
- Private encryption keys
- Medical or legal information

### 2. Brain Isolation

Use separate brains for different security contexts:

```bash
nmem brain create work-public      # Safe to share
nmem brain create work-internal    # Internal only
nmem brain create personal         # Never share
```

### 3. Export Safety

Always use `--exclude-sensitive` when sharing:

```bash
nmem brain export --exclude-sensitive -o shareable.json
nmem brain health --name work-public
```

### 4. Regular Audits

```bash
# Weekly health check
nmem brain health

# Review statistics
nmem stats

# Clean up old memories
nmem cleanup --expired
```

## System Limitations

### No Encryption at Rest

**Current State:** Memory data stored as plain files.

**Workaround:**

- Use encrypted file system (BitLocker, FileVault, LUKS)
- Store data directory on encrypted volume

### No Access Control

**Current State:** No authentication system.

**Workaround:**

- Use file system permissions
- Separate brains per user
- Use server mode with API keys (production)

### No Automatic Cleanup

**Current State:** Memories don't auto-delete.

**Workaround:**

- Manual cleanup: `nmem cleanup --expired`
- Use `--fresh-only` for context
- Apply decay: `nmem decay`

### Database Training Security

**Current State:** DB-to-Brain training uses read-only connections and validates paths.

**Built-in protections:**

- SQLite databases opened in read-only mode (`?mode=ro`)
- Absolute paths rejected in connection strings
- Path traversal (`../`) rejected
- SQL identifiers sanitized (regex validation)
- Error messages sanitized (no raw exceptions exposed)
- Only PRAGMA metadata queried (no data rows accessed)

### ~~No Contradiction Detection~~

**Resolved in v1.5.0:** The `nmem_conflicts` tool detects factual contradictions and decision reversals at encode time, with manual resolution via `keep_existing`, `keep_new`, `keep_both` strategies.

## Incident Response

### Sensitive Data Stored

1. **Identify:** `nmem brain health`
2. **Remove:** `nmem brain delete compromised-brain --force`
3. **Rotate:** Change exposed credentials immediately
4. **Audit:** Check exports and shared data

### Accidental Brain Share

1. **Revoke:** Remove shared files immediately
2. **Assess:** Determine what was exposed
3. **Rotate:** Change potentially exposed credentials
4. **Notify:** Inform affected parties if PII involved

## Recommendations by Use Case

### Personal Use

- Use default local storage
- Enable sensitive content warnings
- Regular backups with `--exclude-sensitive`

### Team Use

- Separate brains per project/sensitivity
- Never share brains with sensitive content
- Use `brain health` before sharing

### Automated/CI Use

- Always set `--min-confidence` threshold
- Never store credentials in memory
- Use environment variables for secrets
- Log confidence scores for audit

### Production/Enterprise

- Use server mode with authentication
- Use encrypted storage backend
- Implement access logging
- Regular security audits
