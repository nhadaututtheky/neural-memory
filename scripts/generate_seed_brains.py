"""Generate 3 seed .brain packages for the Brain Store.

Seeds: Python Best Practices, Git Workflows, Docker Essentials.
Each brain contains curated neurons with rich semantic connections.

Usage:
    python scripts/generate_seed_brains.py
    # Outputs: seeds/python-best-practices.brain
    #          seeds/git-workflows.brain
    #          seeds/docker-essentials.brain
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_memory.core.brain import BrainSnapshot
from neural_memory.engine.brain_package import create_brain_package

UTC = timezone.utc


def _neuron(
    type_: str,
    content: str,
    tags: list[str] | None = None,
    neuron_id: str | None = None,
) -> dict[str, Any]:
    nid = neuron_id or str(uuid4())
    now = datetime.now(UTC).isoformat()
    return {
        "id": nid,
        "type": type_,
        "content": content,
        "created_at": now,
        "updated_at": now,
        "brain_id": "seed",
        "state": "active",
        "activation": 0.5,
        "access_count": 1,
        "salience": 0.6,
        "tags": tags or [],
    }


def _synapse(
    source_id: str,
    target_id: str,
    type_: str = "related_to",
    weight: float = 0.7,
) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "source_id": source_id,
        "target_id": target_id,
        "type": type_,
        "weight": weight,
        "direction": "unidirectional",
        "reinforced_count": 1,
        "created_at": datetime.now(UTC).isoformat(),
    }


from typing import Any


# ── Python Best Practices ──────────────────────────────────────


def build_python_brain() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    neurons: list[dict[str, Any]] = []
    synapses: list[dict[str, Any]] = []

    # --- Decisions ---
    d1 = _neuron(
        "decision",
        "Chose dataclasses over plain dicts for domain models because they provide type safety, immutability (frozen=True), and IDE autocompletion with zero runtime overhead",
        ["python", "architecture"],
        "py-d1",
    )
    d2 = _neuron(
        "decision",
        "Chose Pydantic for API boundary validation and dataclasses for internal domain models — Pydantic handles untrusted input, dataclasses handle trusted internal data",
        ["python", "validation"],
        "py-d2",
    )
    d3 = _neuron(
        "decision",
        "Chose pathlib over os.path because it provides an object-oriented API, handles cross-platform paths correctly, and chains operations fluently",
        ["python", "filesystem"],
        "py-d3",
    )
    d4 = _neuron(
        "decision",
        "Chose f-strings over .format() and % formatting because they are faster, more readable, and support expressions inline since Python 3.6",
        ["python", "strings"],
        "py-d4",
    )
    d5 = _neuron(
        "decision",
        "Chose asyncio + aiohttp over threading for I/O-bound concurrency because cooperative scheduling avoids lock contention and scales to thousands of connections",
        ["python", "async"],
        "py-d5",
    )

    # --- Insights ---
    i1 = _neuron(
        "insight",
        "Type hints don't slow Python at runtime — they're stripped by the interpreter. The cost is zero but the gain in IDE support and mypy catches real bugs before production",
        ["python", "typing"],
        "py-i1",
    )
    i2 = _neuron(
        "insight",
        "Mutable default arguments are shared across all calls: `def f(items=[])` creates ONE list. Always use `None` as default and create inside the function body",
        ["python", "gotchas"],
        "py-i2",
    )
    i3 = _neuron(
        "insight",
        "Context managers (with statement) guarantee cleanup even on exceptions. Use them for files, locks, database connections — anything that needs deterministic release",
        ["python", "patterns"],
        "py-i3",
    )
    i4 = _neuron(
        "insight",
        "List comprehensions are 30-40% faster than equivalent for-loops because the iteration happens in C code. But readability beats speed — switch to loops if the comprehension exceeds one line",
        ["python", "performance"],
        "py-i4",
    )
    i5 = _neuron(
        "insight",
        "Python's GIL means CPU-bound threads don't parallelize. Use multiprocessing for CPU work, asyncio for I/O work, and threading only for C extensions that release the GIL",
        ["python", "concurrency"],
        "py-i5",
    )

    # --- Errors ---
    e1 = _neuron(
        "error",
        "Root cause: bare `except:` catches SystemExit and KeyboardInterrupt, making scripts unkillable. Fix: always catch specific exceptions like `except ValueError as e:`",
        ["python", "error-handling"],
        "py-e1",
    )
    e2 = _neuron(
        "error",
        "Root cause: `import *` silently overwrites local names. A library update can break your code without changing your files. Fix: always import explicitly",
        ["python", "imports"],
        "py-e2",
    )
    e3 = _neuron(
        "error",
        "Root cause: comparing with `==` to None/True/False fails for objects that override __eq__. Fix: use `is None`, `is True` for identity checks",
        ["python", "gotchas"],
        "py-e3",
    )

    # --- Facts ---
    f1 = _neuron(
        "fact",
        "PEP 8 naming: snake_case for functions/variables, PascalCase for classes, SCREAMING_SNAKE for constants. Private: _single_leading_underscore. Name-mangling: __double_leading",
        ["python", "style"],
        "py-f1",
    )
    f2 = _neuron(
        "fact",
        "Virtual environments isolate project dependencies. Always use `python -m venv .venv` or `uv venv` — never install packages globally. Pin versions in requirements.txt or pyproject.toml",
        ["python", "packaging"],
        "py-f2",
    )
    f3 = _neuron(
        "fact",
        "The walrus operator `:=` (Python 3.8+) assigns inside expressions: `if (n := len(items)) > 10:` — useful in while-loops and comprehension filters but hurts readability if overused",
        ["python", "syntax"],
        "py-f3",
    )

    # --- Workflows ---
    w1 = _neuron(
        "workflow",
        "Python project setup: 1) pyproject.toml with [project] metadata 2) src/ layout 3) uv or pip for deps 4) ruff for lint+format 5) mypy for types 6) pytest for tests 7) pre-commit hooks",
        ["python", "tooling"],
        "py-w1",
    )
    w2 = _neuron(
        "workflow",
        "Debugging workflow: 1) Reproduce with minimal case 2) Add breakpoint() 3) Step through with pdb commands (n/s/c/p) 4) Check locals/type 5) Fix and add regression test",
        ["python", "debugging"],
        "py-w2",
    )

    # --- Concepts ---
    c1 = _neuron(
        "concept",
        "EAFP (Easier to Ask Forgiveness than Permission) is Pythonic: try the operation, catch the exception. LBYL (Look Before You Leap) uses if-checks first. EAFP is preferred in Python because exceptions are cheap and race-condition safe",
        ["python", "philosophy"],
        "py-c1",
    )
    c2 = _neuron(
        "concept",
        "Generators yield values lazily — they produce one item at a time and suspend between yields. This means processing a 10GB file uses constant memory. Use generators for any pipeline that processes items sequentially",
        ["python", "patterns"],
        "py-c2",
    )

    neurons = [d1, d2, d3, d4, d5, i1, i2, i3, i4, i5, e1, e2, e3, f1, f2, f3, w1, w2, c1, c2]

    # Semantic connections
    synapses = [
        _synapse("py-d2", "py-i1", "related_to", 0.8),  # pydantic/dataclass → type hints
        _synapse("py-i2", "py-d1", "caused_by", 0.7),  # mutable default → use dataclasses
        _synapse("py-e1", "py-i3", "related_to", 0.6),  # bare except → context managers
        _synapse("py-i5", "py-d5", "caused_by", 0.9),  # GIL insight → asyncio decision
        _synapse("py-i4", "py-c2", "related_to", 0.7),  # comprehension perf → generators
        _synapse("py-f1", "py-w1", "related_to", 0.5),  # naming → project setup
        _synapse("py-e2", "py-f2", "related_to", 0.6),  # import * → virtual envs
        _synapse("py-c1", "py-e1", "related_to", 0.7),  # EAFP → exception handling
        _synapse("py-d3", "py-w2", "related_to", 0.4),  # pathlib → debugging
        _synapse("py-e3", "py-i1", "related_to", 0.5),  # identity check → type hints
    ]

    return neurons, synapses


# ── Git Workflows ──────────────────────────────────────────────


def build_git_brain() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    neurons: list[dict[str, Any]] = []
    synapses: list[dict[str, Any]] = []

    # --- Decisions ---
    d1 = _neuron(
        "decision",
        "Chose conventional commits (feat/fix/refactor) over free-form messages because they enable automated changelogs, semantic versioning, and make git log scannable",
        ["git", "workflow"],
        "git-d1",
    )
    d2 = _neuron(
        "decision",
        "Chose rebase for feature branches over merge because it produces a linear history that's easier to bisect and revert. Only merge for long-lived branches with multiple contributors",
        ["git", "branching"],
        "git-d2",
    )
    d3 = _neuron(
        "decision",
        "Chose squash-merge for PRs to main because it keeps main history clean (one commit per feature) while preserving detailed commits in the PR for review context",
        ["git", "workflow"],
        "git-d3",
    )

    # --- Insights ---
    i1 = _neuron(
        "insight",
        "git reflog is the safety net most developers don't know about. It tracks every HEAD movement for 90 days — even after hard reset, you can recover commits with `git reflog` + `git cherry-pick`",
        ["git", "recovery"],
        "git-i1",
    )
    i2 = _neuron(
        "insight",
        "Interactive rebase (`git rebase -i`) is the most powerful git tool: reorder, squash, split, edit, or drop commits before pushing. Use it to craft clean commit history from messy development",
        ["git", "rebase"],
        "git-i2",
    )
    i3 = _neuron(
        "insight",
        "git stash is a stack, not a single slot. Use `git stash push -m 'description'` to label stashes, `git stash list` to see them, `git stash pop stash@{2}` to apply a specific one",
        ["git", "workflow"],
        "git-i3",
    )
    i4 = _neuron(
        "insight",
        "Atomic commits make bisect, revert, and cherry-pick trivial. Each commit should compile, pass tests, and do ONE thing. If your commit message needs 'and', split it",
        ["git", "best-practices"],
        "git-i4",
    )
    i5 = _neuron(
        "insight",
        "git blame with -w ignores whitespace, -M detects moved lines, -C detects copied lines from other files. Combined: `git blame -wMC` gives accurate authorship even after refactoring",
        ["git", "debugging"],
        "git-i5",
    )

    # --- Errors ---
    e1 = _neuron(
        "error",
        "Root cause: force-pushing to a shared branch overwrites others' work. Fix: use `--force-with-lease` which fails if the remote has commits you haven't fetched yet",
        ["git", "safety"],
        "git-e1",
    )
    e2 = _neuron(
        "error",
        "Root cause: committing secrets (.env, API keys) to git means they're in history forever, even after deletion. Fix: use .gitignore from day 1, git-secrets for pre-commit scanning, BFG to rewrite history if leaked",
        ["git", "security"],
        "git-e2",
    )
    e3 = _neuron(
        "error",
        "Root cause: large merge conflicts from long-lived branches. Prevention: rebase onto main frequently (daily for active branches), keep PRs small (<400 lines), split features into incremental slices",
        ["git", "conflicts"],
        "git-e3",
    )

    # --- Facts ---
    f1 = _neuron(
        "fact",
        "Git stores snapshots, not diffs. Each commit points to a tree of blob objects. Identical files across commits share the same blob (content-addressable). Pack files compress old objects for storage efficiency",
        ["git", "internals"],
        "git-f1",
    )
    f2 = _neuron(
        "fact",
        "Branch naming conventions: feature/ticket-description, fix/ticket-description, release/v1.2.0, hotfix/critical-issue. Slash-separated prefixes enable tab completion and log filtering",
        ["git", "conventions"],
        "git-f2",
    )
    f3 = _neuron(
        "fact",
        ".gitignore patterns: `*.log` ignores all logs, `build/` ignores directory, `!important.log` negates a pattern, `**/temp` matches in any subdirectory. Check with `git check-ignore -v file`",
        ["git", "config"],
        "git-f3",
    )

    # --- Workflows ---
    w1 = _neuron(
        "workflow",
        "PR review workflow: 1) Self-review diff first 2) Run CI locally 3) Write description with context and test plan 4) Request review from domain expert 5) Address feedback in fixup commits 6) Squash-merge after approval",
        ["git", "code-review"],
        "git-w1",
    )
    w2 = _neuron(
        "workflow",
        "Git bisect for bug hunting: 1) `git bisect start` 2) `git bisect bad` (current) 3) `git bisect good v1.0` (known good) 4) Test each checkout 5) Git narrows to the exact breaking commit in O(log n) steps",
        ["git", "debugging"],
        "git-w2",
    )
    w3 = _neuron(
        "workflow",
        "Release workflow: 1) Create release branch from main 2) Bump version + update changelog 3) Tag with `git tag -a v1.2.0 -m 'Release 1.2.0'` 4) Push tag triggers CI pipeline 5) Merge back to main",
        ["git", "release"],
        "git-w3",
    )

    # --- Concepts ---
    c1 = _neuron(
        "concept",
        "Trunk-based development: everyone commits to main (or short-lived branches). No long-lived feature branches. Feature flags control visibility. This reduces merge conflicts and enables continuous deployment",
        ["git", "branching-strategy"],
        "git-c1",
    )
    c2 = _neuron(
        "concept",
        "The three trees of git: Working Directory (your files), Staging Area/Index (next commit preview), HEAD (last commit). `git add` moves to staging, `git commit` moves to HEAD, `git checkout` moves from HEAD to working",
        ["git", "mental-model"],
        "git-c2",
    )

    neurons = [d1, d2, d3, i1, i2, i3, i4, i5, e1, e2, e3, f1, f2, f3, w1, w2, w3, c1, c2]

    synapses = [
        _synapse("git-d1", "git-w3", "related_to", 0.8),  # conventional commits → release
        _synapse("git-d2", "git-i2", "related_to", 0.9),  # rebase decision → interactive rebase
        _synapse("git-d3", "git-i4", "related_to", 0.7),  # squash-merge → atomic commits
        _synapse("git-e1", "git-d2", "caused_by", 0.8),  # force-push danger → rebase decision
        _synapse("git-e3", "git-c1", "related_to", 0.8),  # merge conflicts → trunk-based
        _synapse("git-i1", "git-e1", "related_to", 0.7),  # reflog → force-push recovery
        _synapse("git-i5", "git-w2", "related_to", 0.6),  # blame → bisect debugging
        _synapse("git-f1", "git-c2", "related_to", 0.9),  # internals → three trees
        _synapse("git-e2", "git-f3", "caused_by", 0.7),  # leaked secrets → gitignore
        _synapse("git-w1", "git-d3", "related_to", 0.6),  # PR review → squash-merge
    ]

    return neurons, synapses


# ── Docker Essentials ──���─────────────��─────────────────────────


def build_docker_brain() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    neurons: list[dict[str, Any]] = []
    synapses: list[dict[str, Any]] = []

    # --- Decisions ---
    d1 = _neuron(
        "decision",
        "Chose multi-stage builds over single-stage because they reduce final image size by 60-80%. Build dependencies stay in the builder stage, only runtime artifacts copy to the slim final image",
        ["docker", "optimization"],
        "dk-d1",
    )
    d2 = _neuron(
        "decision",
        "Chose Alpine-based images for production over Ubuntu because they're 5-10MB vs 70-100MB. Trade-off: musl libc can cause subtle compatibility issues with some Python packages (use python:slim as alternative)",
        ["docker", "images"],
        "dk-d2",
    )
    d3 = _neuron(
        "decision",
        "Chose docker compose for local dev over raw docker commands because it declaratively defines multi-service stacks (app + db + cache + worker) and provides consistent networking between containers",
        ["docker", "local-dev"],
        "dk-d3",
    )

    # --- Insights ---
    i1 = _neuron(
        "insight",
        "Each Dockerfile instruction creates a layer. Docker caches layers until a line changes — then invalidates ALL subsequent layers. Put rarely-changing instructions (apt install) before frequently-changing ones (COPY source code) to maximize cache hits",
        ["docker", "caching"],
        "dk-i1",
    )
    i2 = _neuron(
        "insight",
        "Running containers as root is a security risk. Always add `RUN adduser --disabled-password appuser` and `USER appuser` in your Dockerfile. If the app is compromised, the attacker has limited permissions",
        ["docker", "security"],
        "dk-i2",
    )
    i3 = _neuron(
        "insight",
        "Docker volumes persist data beyond container lifecycle. Named volumes (docker volume create) are managed by Docker. Bind mounts (-v /host:/container) map host directories. Use named volumes for databases, bind mounts for development hot-reload",
        ["docker", "storage"],
        "dk-i3",
    )
    i4 = _neuron(
        "insight",
        "Container health checks (`HEALTHCHECK CMD curl -f http://localhost:8000/health`) let Docker and orchestrators know when a container is truly ready, not just running. Critical for zero-downtime deployments",
        ["docker", "reliability"],
        "dk-i4",
    )
    i5 = _neuron(
        "insight",
        ".dockerignore is as important as .gitignore. Without it, `COPY . .` sends node_modules, .git, .env, and build artifacts to the daemon — slowing builds by minutes and leaking secrets into images",
        ["docker", "build"],
        "dk-i5",
    )

    # --- Errors ---
    e1 = _neuron(
        "error",
        "Root cause: using `latest` tag in production means deploys are non-reproducible. A new push to latest changes what runs without any code change. Fix: always pin specific version tags or SHA digests",
        ["docker", "versioning"],
        "dk-e1",
    )
    e2 = _neuron(
        "error",
        "Root cause: COPY before dependency install invalidates cache on every code change. Fix: COPY requirements.txt first, RUN pip install, THEN COPY source. Dependencies cache survives code edits",
        ["docker", "caching"],
        "dk-e2",
    )
    e3 = _neuron(
        "error",
        "Root cause: containers sharing the default bridge network can reach each other by IP. Fix: create custom networks per stack (`docker network create myapp`), containers resolve by service name and are isolated from other stacks",
        ["docker", "networking"],
        "dk-e3",
    )

    # --- Facts ---
    f1 = _neuron(
        "fact",
        "Docker image layers are read-only and shared. Running a container adds a thin writable layer on top (copy-on-write). This is why 10 containers from the same image use almost no extra disk",
        ["docker", "internals"],
        "dk-f1",
    )
    f2 = _neuron(
        "fact",
        "Compose file key directives: `build` (Dockerfile path), `image` (pre-built), `ports` (host:container), `volumes` (persistent storage), `environment` (env vars), `depends_on` (startup order), `healthcheck`, `restart: unless-stopped`",
        ["docker", "compose"],
        "dk-f2",
    )
    f3 = _neuron(
        "fact",
        "Docker resource limits prevent container from consuming all host resources. Set `--memory=512m --cpus=1.5` in run or `mem_limit` / `cpus` in compose. Without limits, one container can OOM-kill the host",
        ["docker", "resources"],
        "dk-f3",
    )

    # --- Workflows ---
    w1 = _neuron(
        "workflow",
        "Dockerfile best practices: 1) FROM specific-version 2) RUN apt-get update && apt-get install -y --no-install-recommends 3) COPY deps first, install, then copy source 4) Multi-stage build 5) Non-root USER 6) HEALTHCHECK 7) LABEL for metadata",
        ["docker", "dockerfile"],
        "dk-w1",
    )
    w2 = _neuron(
        "workflow",
        "Container debugging: 1) `docker logs -f container` for stdout 2) `docker exec -it container sh` for shell access 3) `docker inspect container` for config/network 4) `docker stats` for CPU/memory 5) `docker compose logs --tail=100`",
        ["docker", "debugging"],
        "dk-w2",
    )
    w3 = _neuron(
        "workflow",
        "CI/CD with Docker: 1) Build image in CI 2) Run tests inside container (consistent env) 3) Push to registry with git SHA tag 4) Deploy by updating image reference 5) Rollback = point to previous SHA",
        ["docker", "ci-cd"],
        "dk-w3",
    )

    # --- Concepts ---
    c1 = _neuron(
        "concept",
        "Containers are process isolation, not VMs. They share the host kernel and use Linux namespaces (PID, network, mount) + cgroups (resource limits) for isolation. This is why containers start in milliseconds vs minutes for VMs",
        ["docker", "fundamentals"],
        "dk-c1",
    )
    c2 = _neuron(
        "concept",
        "12-Factor App for containers: config via environment variables (not files baked into image), logs to stdout (not files), stateless processes (persist to external volumes/services), port binding via ENV, disposable (fast start/graceful stop)",
        ["docker", "architecture"],
        "dk-c2",
    )

    neurons = [d1, d2, d3, i1, i2, i3, i4, i5, e1, e2, e3, f1, f2, f3, w1, w2, w3, c1, c2]

    synapses = [
        _synapse("dk-d1", "dk-i1", "related_to", 0.9),  # multi-stage → layer caching
        _synapse("dk-d2", "dk-d1", "related_to", 0.7),  # alpine → multi-stage
        _synapse("dk-e2", "dk-i1", "caused_by", 0.9),  # copy order bug → cache insight
        _synapse("dk-i2", "dk-c1", "related_to", 0.6),  # non-root → process isolation
        _synapse("dk-e1", "dk-w3", "caused_by", 0.8),  # latest tag → CI/CD workflow
        _synapse("dk-i3", "dk-c2", "related_to", 0.7),  # volumes → 12-factor
        _synapse("dk-i5", "dk-e2", "related_to", 0.6),  # dockerignore → copy order
        _synapse("dk-f1", "dk-c1", "related_to", 0.9),  # layers → containers concept
        _synapse("dk-w1", "dk-d1", "related_to", 0.8),  # best practices → multi-stage
        _synapse("dk-i4", "dk-f2", "related_to", 0.5),  # healthcheck → compose directives
    ]

    return neurons, synapses


# ── Package Builder ────────────────���───────────────────────────


def build_package(
    name: str,
    display_name: str,
    description: str,
    tags: list[str],
    category: str,
    neurons: list[dict[str, Any]],
    synapses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a .brain package from raw neurons and synapses."""
    snapshot = BrainSnapshot(
        brain_id=name,
        brain_name=name,
        exported_at=datetime.now(UTC),
        version="1",
        neurons=neurons,
        synapses=synapses,
        fibers=[],
        config={},
        metadata={"source": "seed-generator", "quality": "curated"},
    )

    return create_brain_package(
        snapshot,
        display_name=display_name,
        description=description,
        author="neural-memory",
        name=name,
        tags=tags,
        category=category,
        nmem_version="4.33.0",
    )


def main() -> None:
    output_dir = Path(__file__).parent.parent / "seeds"
    output_dir.mkdir(exist_ok=True)

    brains = [
        {
            "name": "python-best-practices",
            "display_name": "Python Best Practices",
            "description": "Curated knowledge on Pythonic patterns, common gotchas, performance tips, and project setup workflows. Covers type hints, async, error handling, naming conventions, and tooling.",
            "tags": ["python", "best-practices", "patterns", "typing", "async"],
            "category": "programming",
            "builder": build_python_brain,
        },
        {
            "name": "git-workflows",
            "display_name": "Git Workflows",
            "description": "Essential git knowledge: branching strategies, conventional commits, interactive rebase, conflict resolution, security practices, and debugging with bisect and blame.",
            "tags": ["git", "version-control", "workflow", "branching", "code-review"],
            "category": "devops",
            "builder": build_git_brain,
        },
        {
            "name": "docker-essentials",
            "display_name": "Docker Essentials",
            "description": "Docker fundamentals and production practices: multi-stage builds, layer caching, security hardening, compose workflows, CI/CD integration, and the 12-factor methodology.",
            "tags": ["docker", "containers", "devops", "ci-cd", "deployment"],
            "category": "devops",
            "builder": build_docker_brain,
        },
    ]

    for brain_def in brains:
        builder = brain_def.pop("builder")
        neurons, synapses = builder()

        print(f"Building {brain_def['name']}: {len(neurons)} neurons, {len(synapses)} synapses")

        package = build_package(neurons=neurons, synapses=synapses, **brain_def)
        manifest = package["manifest"]
        print(f"  Size: {manifest['size_bytes']} bytes ({manifest['size_tier']})")
        print(
            f"  Scan: {manifest['scan_summary']['risk_level']} (safe={manifest['scan_summary']['safe']})"
        )

        out_path = output_dir / f"{brain_def['name']}.brain"
        out_path.write_text(
            json.dumps(package, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Written: {out_path}")

    print(f"\nDone! {len(brains)} seed brains generated in {output_dir}/")


if __name__ == "__main__":
    main()
