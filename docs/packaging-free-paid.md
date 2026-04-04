# Neural Memory — Free / Pro / Enterprise Packaging Plan

> Status: **Planning** | Author: criznguyen | Created: 2026-03-22

## Overview

Tách Neural Memory thành 3 pack (Free, Pro, Enterprise) sử dụng **license-gated approach** — một package duy nhất, gate tools theo license key tại MCP server layer.

Lý do chọn hướng này:
- Đã có sẵn `TOOL_TIERS` system trong `tool_schemas.py`
- Mixin-based server architecture cho phép gate từng nhóm handler
- Không cần tách repo, không cần quản lý nhiều PyPI packages
- User chỉ cần `pip install neural-memory` + nhập license key để unlock

---

## Tool Distribution

### Free (Community) — 10 tools

Core memory CRUD, đủ dùng cho cá nhân.

| Tool | Mô tả |
|------|--------|
| `nmem_remember` | Lưu memory |
| `nmem_recall` | Tìm memory |
| `nmem_context` | Lấy context gần đây |
| `nmem_recap` | Load project context |
| `nmem_session` | Session state |
| `nmem_show` | Xem chi tiết memory |
| `nmem_edit` | Sửa memory |
| `nmem_forget` | Xóa memory |
| `nmem_todo` | TODOs 30 ngày |
| `nmem_stats` | Brain statistics |

**Limits:**
- Max 1 brain
- Max 5,000 neurons
- SQLite only
- Local embeddings only (no OpenAI/Gemini)

### Pro — 30 tools (Free + 20)

Intelligence layer cho power users.

| Tool | Mô tả |
|------|--------|
| `nmem_remember_batch` | Batch store |
| `nmem_auto` | Auto-capture |
| `nmem_eternal` | Eternal context |
| `nmem_consolidate` | Consolidation (prune/merge/summarize/dedup) |
| `nmem_narrative` | Timeline/topic/causal narratives |
| `nmem_hypothesize` | Hypotheses với Bayesian confidence |
| `nmem_evidence` | Evidence tracking |
| `nmem_predict` | Predictions |
| `nmem_verify` | Verify predictions |
| `nmem_cognitive` | Cognitive overview |
| `nmem_gaps` | Knowledge gap detection |
| `nmem_review` | Spaced repetition |
| `nmem_health` | Health diagnostics |
| `nmem_lifecycle` | Memory lifecycle management |
| `nmem_surface` | Knowledge surface |
| `nmem_explain` | Path explanation giữa entities |
| `nmem_habits` | Workflow habit suggestions |
| `nmem_suggest` | Autocomplete/suggestions |
| `nmem_budget` | Token budget analysis |
| `nmem_drift` | Semantic drift detection |

**Limits:**
- Max 5 brains
- Max 50,000 neurons
- SQLite + PostgreSQL
- Cloud embeddings (OpenAI, Gemini)
- All consolidation modes

### Enterprise — 50 tools (All)

Team, sync, integrations, compliance.

| Tool | Mô tả |
|------|--------|
| `nmem_sync` | Multi-device sync (push/pull/full) |
| `nmem_sync_status` | Sync status |
| `nmem_sync_config` | Sync configuration |
| `nmem_telegram_backup` | Telegram backup |
| `nmem_train` | Train from documents (PDF, DOCX, etc.) |
| `nmem_train_db` | Train from database schema |
| `nmem_import` | Import from ChromaDB, Mem0, Cognee, etc. |
| `nmem_index` | Codebase indexing |
| `nmem_transplant` | Move memories between brains |
| `nmem_schema` | Schema evolution |
| `nmem_conflicts` | Conflict detection/resolution |
| `nmem_version` | Brain version control (snapshot/rollback) |
| `nmem_provenance` | Memory provenance tracking |
| `nmem_source` | Source management |
| `nmem_pin` | Pin/unpin memories |
| `nmem_alert` | Alert management |
| `nmem_tool_stats` | Tool usage analytics |
| `nmem_refine` | Instruction refinement |
| `nmem_report_outcome` | Instruction outcome tracking |

**Limits:**
- Unlimited brains
- Unlimited neurons
- All storage backends (SQLite, PostgreSQL, FalkorDB)
- All embedding providers
- All import/export integrations
- Multi-device sync
- Priority support

---

## Implementation Plan

### Phase 1: License Infrastructure

**Files to create/modify:**

#### 1.1 `src/neural_memory/licensing/__init__.py` (new module)

```
licensing/
  __init__.py        — Public exports
  license_key.py     — JWT decode, validation, caching
  plans.py           — Plan definitions (FREE, PRO, ENTERPRISE)
  gate.py            — Tool gating logic
  limits.py          — Resource limit enforcement
```

#### 1.2 `license_key.py` — License validation

```python
@dataclass(frozen=True)
class License:
    plan: Plan            # FREE | PRO | ENTERPRISE
    email: str
    issued_at: datetime
    expires_at: datetime | None  # None = lifetime
    machine_id: str | None       # None = any machine

class LicenseValidator:
    async def validate(self, key: str) -> License:
        """Decode JWT, check expiry, check machine binding."""

    async def validate_cached(self, key: str) -> License:
        """Validate with 24h local cache (offline-friendly)."""
```

**License format:** Signed JWT (RS256)
- Payload: `{plan, email, iat, exp, machine_id}`
- Public key embedded in package for offline validation
- Optional online check for revocation (fail-open if offline)

#### 1.3 `plans.py` — Plan definitions

```python
class Plan(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

# Tool → minimum required plan
TOOL_PLANS: dict[str, Plan] = {
    "nmem_remember": Plan.FREE,
    "nmem_recall": Plan.FREE,
    # ... all 50 tools mapped
    "nmem_sync": Plan.ENTERPRISE,
}

# Resource limits per plan
PLAN_LIMITS: dict[Plan, PlanLimits] = {
    Plan.FREE: PlanLimits(max_brains=1, max_neurons=5000, ...),
    Plan.PRO: PlanLimits(max_brains=5, max_neurons=50000, ...),
    Plan.ENTERPRISE: PlanLimits(max_brains=None, max_neurons=None, ...),
}
```

#### 1.4 `gate.py` — Tool gating

```python
class ToolGate:
    def __init__(self, license: License):
        self.license = license

    def is_allowed(self, tool_name: str) -> bool:
        required = TOOL_PLANS.get(tool_name, Plan.ENTERPRISE)
        return self.license.plan >= required

    def gate_message(self, tool_name: str) -> str:
        required = TOOL_PLANS[tool_name]
        return f"🔒 {tool_name} requires {required.value} plan. Upgrade at https://neuralmemory.dev/pricing"
```

### Phase 2: Config Integration

**Modify:** `src/neural_memory/unified_config.py`

```python
@dataclass(frozen=True)
class LicenseConfig:
    key: str = ""          # License key (JWT)
    plan: str = "free"     # Resolved plan (cached)
    expires: str = ""      # Expiry date (cached)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "plan": self.plan, "expires": self.expires}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LicenseConfig:
        return cls(
            key=str(data.get("key", "")),
            plan=str(data.get("plan", "free")),
            expires=str(data.get("expires", "")),
        )
```

**Config file** (`~/.neuralmemory/config.toml`):

```toml
[license]
key = "eyJhbGciOiJS..."
```

### Phase 3: MCP Server Gate

**Modify:** `src/neural_memory/mcp/server.py`

```python
def get_tools(self) -> list[dict[str, Any]]:
    """Return tools filtered by license plan."""
    all_schemas = get_tool_schemas()
    gate = ToolGate(self.license)
    return [t for t in all_schemas if gate.is_allowed(t["name"])]

async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    gate = ToolGate(self.license)
    if not gate.is_allowed(name):
        return {"error": gate.gate_message(name)}
    # ... existing dispatch logic
```

### Phase 4: Resource Limits

**Modify:** `src/neural_memory/storage/memory_store.py`

```python
async def add_neuron(self, neuron: Neuron) -> str:
    # Check neuron count limit
    if self.limits.max_neurons is not None:
        count = await self.storage.count_neurons()
        if count >= self.limits.max_neurons:
            raise LimitExceeded(
                f"Free plan limit: {self.limits.max_neurons} neurons. "
                "Upgrade to Pro for 50,000."
            )
    # ... existing logic
```

### Phase 5: CLI Commands

**Add:** `src/neural_memory/cli/commands/license_cmd.py`

```
nmem license activate <key>    — Activate license
nmem license status            — Show current plan + expiry
nmem license deactivate        — Remove license
```

### Phase 6: Upgrade Prompts (UX)

Khi user gọi tool bị gate, trả về message hữu ích:

```
🔒 nmem_sync requires Enterprise plan.

You're on the Free plan. Upgrade to unlock:
  Pro ($9/mo)       — 30 tools, cognitive layer, narratives
  Enterprise ($29/mo) — 50 tools, sync, team, integrations

→ https://neuralmemory.dev/pricing
→ Or run: nmem license activate <key>
```

---

## Migration Plan (Existing Users)

1. **Existing users without key** → default to Free plan
2. **Grace period** (30 days after release) → all existing users get Pro features free
3. **After grace period** → Free plan enforced, prompt to upgrade
4. **Existing data preserved** — no data loss, just tool access restricted
5. **`nmem_recall` always works** — never gate read access to existing memories

---

## Pricing Model (Draft)

| Plan | Price | Target |
|------|-------|--------|
| **Free** | $0 | Cá nhân, thử nghiệm |
| **Pro** | $9/month hoặc $90/year | Power users, developers |
| **Enterprise** | $29/month hoặc $290/year | Teams, companies |
| **Lifetime Pro** | $199 one-time | Early adopters |

---

## License Key Infrastructure

### Option A: Self-hosted (MVP)

- Generate JWT keys locally with a CLI script
- Public key embedded in package
- No server needed
- Manual key distribution

### Option B: Stripe + License API

```
neuralmemory.dev/api/license
  POST /activate    — Stripe checkout → generate JWT
  POST /validate    — Check revocation
  POST /deactivate  — Revoke key
```

- Stripe for payments
- Simple FastAPI server for key management
- Webhook: Stripe payment → generate + email key

### Option C: Polar.sh / Lemon Squeezy

- Third-party license management
- Webhook on purchase → deliver key
- Built-in license validation API
- Lower maintenance

**Recommendation:** Start with **Option C** (Polar.sh) for MVP, migrate to self-hosted later if needed.

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/neural_memory/licensing/` | **New** | License module (key, plans, gate, limits) |
| `src/neural_memory/unified_config.py` | Modify | Add `LicenseConfig` |
| `src/neural_memory/mcp/server.py` | Modify | Gate `get_tools()` + `call_tool()` |
| `src/neural_memory/mcp/tool_schemas.py` | Modify | Add `TOOL_PLANS` mapping |
| `src/neural_memory/storage/memory_store.py` | Modify | Resource limit checks |
| `src/neural_memory/cli/commands/license_cmd.py` | **New** | CLI license commands |
| `tests/unit/test_licensing.py` | **New** | License validation tests |
| `tests/unit/test_tool_gate.py` | **New** | Tool gating tests |

---

## Open Questions

- [ ] Nên gate ở `get_tools()` (ẩn tool) hay `call_tool()` (hiện tool nhưng block khi gọi)?
  - Đề xuất: **cả hai** — ẩn khỏi schema + block nếu gọi trực tiếp
- [ ] Offline-first hay online-first validation?
  - Đề xuất: **offline-first** (JWT với embedded public key), online check optional
- [ ] Free plan có giới hạn thời gian không?
  - Đề xuất: **không** — free forever, chỉ giới hạn features + resources
- [ ] Team/org licenses cho Enterprise?
  - Để sau, MVP chỉ cần per-user license
- [ ] Có cần separate PyPI package cho enterprise-only integrations (FalkorDB, Mem0 sync)?
  - Đề xuất: giữ trong cùng package, gate bằng license
