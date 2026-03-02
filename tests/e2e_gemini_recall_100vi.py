"""E2E test round 2: 100 Vietnamese queries against existing huskyAI brain via Gemini embeddings."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_gemini_100vi")

for name in ["neural_memory.engine", "neural_memory.storage", "neural_memory.safety"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# Reuse existing trained brain from round 1
EXISTING_DB = "/var/folders/l_/97h__v4n3p39qyz6qqw74kbm0000gn/T/nmem_e2e_r7rtdx5k/test_brain.db"
BRAIN_ID = "huskyAI"

# 100 Vietnamese queries covering motorcycle manual topics
VI_QUERIES: list[tuple[str, str]] = [
    # --- Oil & Lubrication ---
    ("oil", "cách thay dầu nhớt xe KTM"),
    ("oil", "loại dầu nhớt nào phù hợp cho xe KTM"),
    ("oil", "khi nào cần thay dầu động cơ"),
    ("oil", "dung tích dầu nhớt xe là bao nhiêu"),
    ("oil", "bước kiểm tra mức dầu nhớt"),
    ("oil", "dầu nhớt tổng hợp hay khoáng cho xe mô tô"),
    ("oil", "thay lọc dầu nhớt như thế nào"),
    ("oil", "độ nhớt dầu khuyến nghị SAE bao nhiêu"),
    ("oil", "dầu nhớt hộp số và dầu động cơ khác nhau không"),
    ("oil", "tần suất thay dầu nhớt xe máy"),
    # --- Tires & Pressure ---
    ("tire", "áp suất lốp xe bao nhiêu"),
    ("tire", "cách kiểm tra áp suất lốp"),
    ("tire", "khi nào cần thay lốp xe"),
    ("tire", "loại lốp nào phù hợp cho xe KTM"),
    ("tire", "kích thước lốp trước và lốp sau"),
    ("tire", "hệ thống cảm biến áp suất lốp TPMS hoạt động thế nào"),
    ("tire", "áp suất lốp khi chở hai người"),
    ("tire", "cách thay lốp xe mô tô"),
    ("tire", "lốp xe mòn một bên là do nguyên nhân gì"),
    ("tire", "nhiệt độ ảnh hưởng áp suất lốp thế nào"),
    # --- Brakes ---
    ("brake", "hệ thống phanh xe hoạt động thế nào"),
    ("brake", "cách kiểm tra má phanh"),
    ("brake", "khi nào cần thay dầu phanh"),
    ("brake", "loại dầu phanh nào phù hợp DOT mấy"),
    ("brake", "hệ thống ABS trên xe hoạt động ra sao"),
    ("brake", "cách xả gió hệ thống phanh"),
    ("brake", "phanh trước và phanh sau khác nhau thế nào"),
    ("brake", "đèn cảnh báo phanh sáng là sao"),
    ("brake", "độ dày má phanh tối thiểu bao nhiêu"),
    ("brake", "cách điều chỉnh tay phanh"),
    # --- Engine ---
    ("engine", "thông số kỹ thuật động cơ xe KTM"),
    ("engine", "nhiệt độ hoạt động bình thường của động cơ"),
    ("engine", "cách khởi động xe khi trời lạnh"),
    ("engine", "tiếng kêu lạ từ động cơ là do đâu"),
    ("engine", "hệ thống làm mát động cơ hoạt động thế nào"),
    ("engine", "nước làm mát loại nào phù hợp"),
    ("engine", "công suất tối đa của xe là bao nhiêu"),
    ("engine", "momen xoắn cực đại bao nhiêu"),
    ("engine", "tốc độ vòng tua máy tối đa"),
    ("engine", "hệ thống phun nhiên liệu EFI"),
    # --- Electrical ---
    ("elec", "cách thay ắc quy xe"),
    ("elec", "loại ắc quy phù hợp cho xe"),
    ("elec", "kiểm tra hệ thống sạc ắc quy"),
    ("elec", "đèn báo trên bảng đồng hồ nghĩa là gì"),
    ("elec", "cách thay bóng đèn pha"),
    ("elec", "hệ thống đánh lửa hoạt động thế nào"),
    ("elec", "cầu chì xe nằm ở đâu"),
    ("elec", "cách kiểm tra bugi xe máy"),
    ("elec", "khoảng cách khe hở bugi bao nhiêu"),
    ("elec", "sạc ắc quy bằng cách nào"),
    # --- Chain & Drivetrain ---
    ("chain", "cách căng xích xe"),
    ("chain", "khi nào cần thay xích xe"),
    ("chain", "loại dầu bôi trơn xích nào tốt"),
    ("chain", "độ chùng xích tiêu chuẩn bao nhiêu"),
    ("chain", "cách vệ sinh xích xe"),
    ("chain", "nhông sên đĩa thay cùng lúc không"),
    ("chain", "số răng nhông trước và nhông sau"),
    ("chain", "cách kiểm tra độ mòn xích"),
    ("chain", "xích DID hay RK tốt hơn"),
    ("chain", "tần suất bôi trơn xích bao lâu"),
    # --- Suspension & Frame ---
    ("susp", "cách điều chỉnh giảm xóc trước"),
    ("susp", "độ cứng giảm xóc phù hợp bao nhiêu"),
    ("susp", "giảm xóc sau có điều chỉnh được không"),
    ("susp", "áp suất nitơ giảm xóc bao nhiêu"),
    ("susp", "khi nào cần thay dầu giảm xóc"),
    ("susp", "chiều cao yên xe bao nhiêu"),
    ("susp", "trọng lượng tối đa xe chịu được"),
    ("susp", "hành trình giảm xóc trước bao nhiêu mm"),
    ("susp", "cách kiểm tra giảm xóc bị hỏng"),
    ("susp", "giảm xóc nguyên bản có tốt không"),
    # --- General Maintenance ---
    ("maint", "lịch bảo dưỡng định kỳ xe KTM"),
    ("maint", "những mục kiểm tra trước khi chạy xe"),
    ("maint", "cách thay lọc gió"),
    ("maint", "khi nào cần vệ sinh kim phun"),
    ("maint", "dung tích bình xăng bao nhiêu lít"),
    ("maint", "loại xăng phù hợp RON bao nhiêu"),
    ("maint", "cách rửa xe đúng cách"),
    ("maint", "bảo quản xe khi không sử dụng lâu"),
    ("maint", "momen siết bu lông bánh xe bao nhiêu"),
    ("maint", "cách tháo và lắp bánh xe"),
    # --- Safety & Operation ---
    ("safety", "trang bị an toàn cần thiết khi lái xe"),
    ("safety", "cách sử dụng chế độ lái khác nhau"),
    ("safety", "hệ thống kiểm soát lực kéo TCS"),
    ("safety", "cách lái xe an toàn trong mưa"),
    ("safety", "tốc độ tối đa xe chạy được"),
    ("safety", "cách vận hành xe đúng cách cho người mới"),
    ("safety", "chế độ lái sport và street khác nhau thế nào"),
    ("safety", "hệ thống chống trượt hoạt động ra sao"),
    ("safety", "cách đỗ xe an toàn dùng chân chống"),
    ("safety", "quy trình tắt máy đúng cách"),
    # --- Clutch & Gearbox ---
    ("clutch", "cách điều chỉnh tay côn"),
    ("clutch", "ly hợp trượt là gì"),
    ("clutch", "khi nào cần thay lá côn"),
    ("clutch", "hành trình tự do tay côn bao nhiêu"),
    ("clutch", "hộp số quickshifter hoạt động thế nào"),
    ("clutch", "cách sang số mượt mà nhất"),
    ("clutch", "xe có mấy số"),
    ("clutch", "dầu ly hợp và dầu phanh dùng chung không"),
    ("clutch", "tiếng kêu khi nhả côn là bình thường không"),
    ("clutch", "cách kiểm tra cáp côn"),
]


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY env var")
        sys.exit(1)

    # --- Step 1: Load existing trained brain ---
    logger.info("Step 1: Loading existing huskyAI brain from %s", EXISTING_DB)

    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.sqlite_store import SQLiteStorage

    db_path = Path(EXISTING_DB)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    storage = SQLiteStorage(db_path)
    await storage.initialize()

    loaded_brain = await storage.get_brain(BRAIN_ID)
    assert loaded_brain is not None, f"Brain '{BRAIN_ID}' not found in DB!"

    brain_config = loaded_brain.config
    storage.set_brain(loaded_brain.id)
    logger.info("  Brain: %s, embedding_enabled=%s, provider=%s",
                loaded_brain.name, brain_config.embedding_enabled, brain_config.embedding_provider)

    all_neurons = await storage.find_neurons(limit=1000)
    emb_count = sum(1 for n in all_neurons if n.metadata.get("_embedding"))
    logger.info("  Neurons: %d total, %d with embeddings", len(all_neurons), emb_count)

    # --- Step 2: Recall 100 Vietnamese queries ---
    logger.info("Step 2: Testing 100 Vietnamese queries")

    from neural_memory.engine.retrieval import ReflexPipeline

    pipeline = ReflexPipeline(storage, brain_config)

    results_data: list[dict] = []
    recall_start = time.time()

    for i, (category, query) in enumerate(VI_QUERIES):
        try:
            r = await pipeline.query(query, max_tokens=3000)
            n_fibers = len(r.fibers_matched)
            conf = r.confidence
            answer_preview = (r.answer or "(no answer)")[:150]
            status = "OK" if n_fibers > 0 else "FAIL"
        except Exception as e:
            n_fibers = 0
            conf = 0.0
            answer_preview = f"ERROR: {e}"
            status = "ERROR"

        results_data.append({
            "index": i + 1,
            "category": category,
            "query": query,
            "status": status,
            "fibers": n_fibers,
            "confidence": round(conf, 2),
            "answer_preview": answer_preview,
        })

        # Progress every 10 queries
        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/100 queries done", i + 1)

    recall_elapsed = time.time() - recall_start

    # --- Stats ---
    total = len(results_data)
    ok_count = sum(1 for r in results_data if r["status"] == "OK")
    fail_count = sum(1 for r in results_data if r["status"] == "FAIL")
    error_count = sum(1 for r in results_data if r["status"] == "ERROR")
    avg_conf = sum(r["confidence"] for r in results_data if r["status"] == "OK") / max(ok_count, 1)
    avg_fibers = sum(r["fibers"] for r in results_data if r["status"] == "OK") / max(ok_count, 1)

    # Per-category stats
    categories = sorted(set(r["category"] for r in results_data))
    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in results_data if r["category"] == cat]
        cat_ok = sum(1 for r in cat_results if r["status"] == "OK")
        cat_total = len(cat_results)
        cat_avg_conf = sum(r["confidence"] for r in cat_results if r["status"] == "OK") / max(cat_ok, 1)
        cat_stats[cat] = {"ok": cat_ok, "total": cat_total, "avg_conf": round(cat_avg_conf, 2)}

    # --- Print results ---
    print("\n" + "=" * 90)
    print("E2E GEMINI RECALL — 100 VIETNAMESE QUERIES (Round 2)")
    print("=" * 90)
    print(f"DB: {db_path}")
    print(f"Neurons: {len(all_neurons)} total, {emb_count} with embeddings")
    print(f"Provider: {brain_config.embedding_provider} / {brain_config.embedding_model}")
    print(f"Threshold: {brain_config.embedding_similarity_threshold}")
    print(f"Recall time: {recall_elapsed:.1f}s ({recall_elapsed/total:.2f}s/query)")
    print("-" * 90)
    print(f"RESULTS: {ok_count}/{total} OK | {fail_count} FAIL | {error_count} ERROR")
    print(f"Avg confidence (OK): {avg_conf:.2f}")
    print(f"Avg fibers (OK): {avg_fibers:.1f}")
    print("-" * 90)

    print("\nPer-Category Breakdown:")
    for cat in categories:
        s = cat_stats[cat]
        pct = s["ok"] / s["total"] * 100
        print(f"  {cat:8s}: {s['ok']}/{s['total']} ({pct:5.1f}%) avg_conf={s['avg_conf']:.2f}")

    print("\nDetailed Results:")
    print("-" * 90)
    for r in results_data:
        marker = "✅" if r["status"] == "OK" else "❌"
        print(f"  {r['index']:3d}. {marker} [{r['category']:6s}] fibers={r['fibers']:2d} conf={r['confidence']:.2f} | {r['query']}")
        if r["status"] == "OK" and r["fibers"] > 0:
            print(f"       → {r['answer_preview'][:100]}")

    # Failed queries detail
    failed = [r for r in results_data if r["status"] != "OK"]
    if failed:
        print(f"\nFailed Queries ({len(failed)}):")
        for r in failed:
            print(f"  {r['index']}. {r['query']}")
            print(f"     {r['answer_preview']}")

    print("-" * 90)

    # Save JSON results next to test file
    json_path = Path(__file__).resolve().parent / "results_100vi.json"
    json_path.write_text(json.dumps({
        "summary": {
            "total": total,
            "ok": ok_count,
            "fail": fail_count,
            "error": error_count,
            "success_rate": round(ok_count / total * 100, 1),
            "avg_confidence": round(avg_conf, 2),
            "avg_fibers": round(avg_fibers, 1),
            "existing_brain": BRAIN_ID,
            "recall_time_s": round(recall_elapsed, 1),
            "recall_per_query_s": round(recall_elapsed / total, 2),
            "neurons_total": len(all_neurons),
            "neurons_with_embeddings": emb_count,
        },
        "category_stats": cat_stats,
        "results": results_data,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON results saved: {json_path}")

    await storage.close()
    print(f"DB: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
