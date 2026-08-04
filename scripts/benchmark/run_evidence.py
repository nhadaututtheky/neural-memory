"""Run Neural Memory and offline baselines under one evidence contract.

Examples:
    python scripts/benchmark/run_evidence.py --variant s --limit 5 \
        --methods naive,fts5,recency --retrieval-only
    python scripts/benchmark/run_evidence.py --variant s \
        --instance-ids scripts/benchmark/mini_bench_ids.json \
        --methods nm,naive,fts5,embedding --require-embedding --retrieval-only
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark.baselines import BASELINES  # noqa: E402
from scripts.benchmark.compare_baselines import run_baseline_method  # noqa: E402
from scripts.benchmark.config import BenchmarkConfig  # noqa: E402
from scripts.benchmark.data_loader import _VARIANT_FILES, LMEInstance, load_dataset  # noqa: E402
from scripts.benchmark.evidence import (  # noqa: E402
    PINNED_BACKEND,
    PINNED_EMBEDDING_MODEL,
    PINNED_EMBEDDING_REVISION,
    PINNED_RETRIEVAL_PROFILE,
    build_evidence_manifest,
    canonical_json_sha256,
    evaluate_evidence,
)
from scripts.benchmark.metrics import QuestionResult  # noqa: E402
from scripts.benchmark.report import build_evidence_report, save_evidence_report  # noqa: E402

logger = logging.getLogger("benchmark_evidence")

CANONICAL_METHODS = frozenset({"nm", "naive", "fts5", "embedding"})


class MissingCapabilityError(RuntimeError):
    """Raised when a required optional benchmark capability is unavailable."""


def _nm_environment(seed: int, temporary_dir: Path, backend: str) -> dict[str, str]:
    """Build isolated child settings without leaking SQLite-only strict mode."""
    environment = {
        **os.environ,
        "PYTHONHASHSEED": str(seed),
        "NMEM_BENCHMARK_SEED": str(seed),
        "NMEM_BENCHMARK_WORK_DIR": str(temporary_dir),
        "NMEM_SENTENCE_TRANSFORMER_REVISION": PINNED_EMBEDDING_REVISION,
    }
    if backend == PINNED_BACKEND:
        environment["NMEM_REQUIRE_EMBEDDING"] = "1"
    else:
        environment.pop("NMEM_REQUIRE_EMBEDDING", None)
    return environment


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = [method for method in methods if method != "nm" and method not in BASELINES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; available: {['nm', *BASELINES.keys()]}"
        )
    if not methods:
        raise argparse.ArgumentTypeError("at least one method is required")
    return methods


def _load_instance_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    instance_ids = data.get("instance_ids")
    if not isinstance(instance_ids, list) or not all(
        isinstance(question_id, str) for question_id in instance_ids
    ):
        raise ValueError(f"{path} must contain a string instance_ids array")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"{path} contains duplicate instance IDs")
    return instance_ids


def _select_instances(
    instances: list[LMEInstance],
    *,
    instance_ids_path: Path | None,
    limit: int | None,
) -> list[LMEInstance]:
    if instance_ids_path is None:
        return instances[:limit] if limit is not None else instances

    requested_ids = _load_instance_ids(instance_ids_path)
    by_id = {instance.question_id: instance for instance in instances}
    missing_ids = [question_id for question_id in requested_ids if question_id not in by_id]
    if missing_ids:
        raise ValueError(f"instance IDs not present in dataset: {missing_ids}")
    selected = [by_id[question_id] for question_id in requested_ids]
    return selected[:limit] if limit is not None else selected


async def _run_nm_method(
    instances: list[LMEInstance],
    *,
    variant: str,
    backend: str,
    output_dir: Path,
    timeout_sec: float,
    seed: int,
) -> tuple[list[QuestionResult], float]:
    """Run each NM instance in the existing isolated subprocess harness."""
    run_instance_path = Path(__file__).resolve().parent / "run_instance.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[QuestionResult] = []
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="nmem-evidence-", dir=output_dir) as temp_name:
        temporary_dir = Path(temp_name).resolve()
        for index, instance in enumerate(instances, start=1):
            result_path = _safe_result_path(temporary_dir, instance.question_id)
            command = [
                sys.executable,
                str(run_instance_path),
                variant,
                instance.question_id,
                str(result_path),
                "--backend",
                backend,
            ]
            environment = _nm_environment(seed, temporary_dir, backend)
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=_REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            if completed.returncode != 0 or not result_path.exists():
                error_tail = completed.stderr[-500:].strip()
                raise RuntimeError(
                    f"NM failed for {instance.question_id} with exit code "
                    f"{completed.returncode}: {error_tail or 'no result artifact'}"
                )
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
            result = QuestionResult.from_dict(result_data)
            if result.question_id != instance.question_id:
                raise RuntimeError(
                    f"NM result ID mismatch: expected {instance.question_id}, got {result.question_id}"
                )
            results.append(result)
            print(f"  [nm] {index}/{len(instances)}", flush=True)

    return results, time.perf_counter() - started


def _embedding_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _safe_result_path(temporary_dir: Path, question_id: str) -> Path:
    """Derive an opaque result filename and prove it remains under the temp root."""
    root = temporary_dir.resolve()
    filename = f"{hashlib.sha256(question_id.encode('utf-8')).hexdigest()}.json"
    result_path = (root / filename).resolve()
    if not result_path.is_relative_to(root):
        raise ValueError("benchmark result path escaped the temporary directory")
    return result_path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)


def _load_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _limit_result(result: QuestionResult, top_k: int) -> QuestionResult:
    retrieved_ids = result.retrieved_session_ids[:top_k]
    return QuestionResult(
        question_id=result.question_id,
        question_type=result.question_type,
        hypothesis=result.hypothesis,
        correct=result.correct,
        retrieved_session_ids=retrieved_ids,
        answer_session_ids=list(result.answer_session_ids),
        retrieval_hit=any(session_id in retrieved_ids for session_id in result.answer_session_ids),
        elapsed_sec=result.elapsed_sec,
    )


async def _run_selected_method(
    method: str,
    instances: list[LMEInstance],
    *,
    config: BenchmarkConfig,
    top_k: int,
    nm_timeout_sec: float,
    seed: int,
) -> tuple[list[QuestionResult], float]:
    _seed_everything(seed)
    if method == "nm":
        results, elapsed = await _run_nm_method(
            instances,
            variant=config.variant,
            backend=config.backend,
            output_dir=config.output_dir,
            timeout_sec=nm_timeout_sec,
            seed=seed,
        )
        return [_limit_result(result, top_k) for result in results], elapsed
    return await run_baseline_method(method, instances, top_k)


async def run_evidence(
    config: BenchmarkConfig,
    *,
    methods: tuple[str, ...],
    top_k: int = 10,
    seed: int = 42,
    warmup_runs: int = 0,
    require_embedding: bool = False,
    nm_timeout_sec: float = 600.0,
    baseline_path: Path | None = None,
) -> tuple[dict[str, object], Path, Path]:
    """Run selected methods on one ID set and persist a versioned evidence artifact."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if "nm" in methods and top_k > 10:
        raise ValueError("NM isolated runner supports top_k up to 10")
    all_instances = load_dataset(config.variant, config.data_dir)
    instances = _select_instances(
        all_instances,
        instance_ids_path=config.instance_ids_file,
        limit=config.limit,
    )
    if not instances:
        raise ValueError("no benchmark instances selected")

    corpus_path = config.data_dir / _VARIANT_FILES[config.variant]
    manifest = build_evidence_manifest(
        config,
        instances,
        corpus_path=corpus_path,
        seed=seed,
        warmup_runs=warmup_runs,
        run_config={
            "methods": list(methods),
            "top_k": top_k,
            "require_embedding": require_embedding,
            "nm_timeout_sec": nm_timeout_sec,
            "embedding_model": PINNED_EMBEDDING_MODEL,
            "embedding_revision": PINNED_EMBEDDING_REVISION,
            "retrieval_profile": PINNED_RETRIEVAL_PROFILE,
        },
    )

    non_canonical_reasons: list[str] = []
    pinned_baseline_path = Path(__file__).parent / "evidence_baseline.json"
    selected_baseline_path = baseline_path or pinned_baseline_path
    if selected_baseline_path.resolve() != pinned_baseline_path.resolve():
        non_canonical_reasons.append("custom_baseline")
    if config.backend != PINNED_BACKEND:
        non_canonical_reasons.append(f"backend_not_pinned:{config.backend}")
    selected_methods = list(methods)
    if "embedding" in selected_methods and not _embedding_available():
        if require_embedding:
            raise MissingCapabilityError(
                "embedding baseline requires `pip install neural-memory[embeddings]`"
            )
        selected_methods.remove("embedding")
        non_canonical_reasons.append("embedding_unavailable")

    missing_canonical_methods = sorted(CANONICAL_METHODS - set(selected_methods))
    non_canonical_reasons.extend(f"method_not_run:{method}" for method in missing_canonical_methods)
    if manifest.git_dirty:
        non_canonical_reasons.append("git_worktree_dirty")
    if manifest.git_sha == "unknown":
        non_canonical_reasons.append("git_metadata_unknown")

    method_results: dict[str, list[QuestionResult]] = {}
    total_elapsed: dict[str, float] = {}
    for method in selected_methods:
        print(f"\n=== Running {method} ===", flush=True)
        try:
            for warmup_index in range(warmup_runs):
                print(f"  [{method}] warmup {warmup_index + 1}/{warmup_runs}", flush=True)
                await _run_selected_method(
                    method,
                    instances[:1],
                    config=config,
                    top_k=top_k,
                    nm_timeout_sec=nm_timeout_sec,
                    seed=seed,
                )
            results, elapsed = await _run_selected_method(
                method,
                instances,
                config=config,
                top_k=top_k,
                nm_timeout_sec=nm_timeout_sec,
                seed=seed,
            )
        except Exception as exc:
            if method != "embedding":
                raise
            if require_embedding:
                raise MissingCapabilityError(
                    "pinned embedding model revision is unavailable"
                ) from exc
            logger.warning("Embedding baseline unavailable at runtime: %s", exc)
            non_canonical_reasons.extend(
                ["embedding_runtime_unavailable", "method_not_run:embedding"]
            )
            continue
        method_results[method] = results
        total_elapsed[method] = elapsed

    canonical = not non_canonical_reasons and set(selected_methods) >= CANONICAL_METHODS
    report = build_evidence_report(
        manifest,
        method_results,
        total_elapsed=total_elapsed,
        canonical=canonical,
        non_canonical_reasons=tuple(non_canonical_reasons),
    )
    baseline = _load_json_object(selected_baseline_path)
    _attach_evidence_gate(report, baseline)
    json_path, markdown_path = save_evidence_report(report, config.output_dir)
    return report, json_path, markdown_path


def _attach_evidence_gate(
    report: dict[str, object],
    baseline: dict[str, object],
) -> None:
    """Attach the gate result and prevent a failed gate from remaining canonical."""
    report["baseline_sha256"] = canonical_json_sha256(baseline)
    gate_result = evaluate_evidence(report, baseline)
    if not gate_result.passed:
        report["canonical"] = False
        reasons = report.get("non_canonical_reasons")
        if not isinstance(reasons, list):
            reasons = []
            report["non_canonical_reasons"] = reasons
        if "evidence_gate_failed" not in reasons:
            reasons.append("evidence_gate_failed")
        gate_result = evaluate_evidence(report, baseline)
    report["gate"] = {
        "passed": gate_result.passed,
        "failures": list(gate_result.failures),
    }


def _parse_args() -> tuple[BenchmarkConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Canonical LongMemEval evidence runner")
    parser.add_argument("--variant", choices=["oracle", "s", "m"], default="s")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-ids", type=Path, default=None)
    parser.add_argument("--backend", choices=["sqlite", "infinitydb"], default="sqlite")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument(
        "--methods", type=_parse_methods, default=_parse_methods("nm,naive,fts5,embedding")
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--require-embedding", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).parent / "evidence_baseline.json",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit nonzero after saving the artifact when its evidence gate fails",
    )
    parser.add_argument("--nm-timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Required declaration: evidence currently measures retrieval only",
    )
    args = parser.parse_args()
    if not args.retrieval_only:
        parser.error("canonical evidence currently requires --retrieval-only")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    config = BenchmarkConfig(
        variant=args.variant,
        limit=args.limit,
        backend=args.backend,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        retrieval_only=True,
        instance_ids_file=args.instance_ids,
    )
    return config, args


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config, args = _parse_args()
    report, json_path, markdown_path = await run_evidence(
        config,
        methods=args.methods,
        top_k=args.top_k,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        require_embedding=args.require_embedding,
        nm_timeout_sec=args.nm_timeout_sec,
        baseline_path=args.baseline,
    )
    print(f"\nCanonical: {report['canonical']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    gate = report["gate"]
    if not isinstance(gate, dict):
        raise RuntimeError("evidence report gate is malformed")
    print(f"Gate passed: {gate['passed']}")
    if args.enforce_gate and gate["passed"] is not True:
        failures = gate.get("failures", [])
        for failure in failures if isinstance(failures, (list, tuple)) else []:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
