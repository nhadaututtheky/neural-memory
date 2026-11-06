"""Chinese recall through the real SQLStorage + FTS5 pipeline.

Verifies that CJK short-word queries hit via the cjk_spaced() index
(utils/cjk.py) end-to-end: insert, update, delete, fiber summaries,
and the v41 -> v42 migration backfill.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sql.sql_storage import SQLStorage
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def store(tmp_path: Path):
    s = SQLStorage(SQLiteDialect(tmp_path / "zh.db"))
    await s.initialize()
    brain = Brain.create(name="zh")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    yield s
    await s.close()


async def test_chinese_neuron_short_word_hits(store: SQLStorage) -> None:
    await store.add_neuron(
        Neuron.create(type=NeuronType.CONCEPT, content="起源世界与太古文明和太空电梯")
    )
    hits = await store.find_neurons(content_contains="起源", limit=10)
    assert any("起源" in n.content for n in hits)
    hits2 = await store.find_neurons(content_contains="电梯", limit=10)
    assert any("电梯" in n.content for n in hits2)


async def test_chinese_mixed_latin_neuron(store: SQLStorage) -> None:
    await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="服务器配置 v2rayN 代理"))
    hits = await store.find_neurons(content_contains="代理", limit=10)
    assert any("代理" in n.content for n in hits)
    hits2 = await store.find_neurons(content_contains="v2rayN代理", limit=10)
    assert any("v2rayN" in n.content for n in hits2)


async def test_chinese_fiber_summary_hits(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="anchor fiber")
    await store.add_neuron(n)
    fiber = Fiber.create(
        neuron_ids={n.id},
        synapse_ids=set(),
        anchor_neuron_id=n.id,
        summary="起源设定共工怒触不周山",
    )
    await store.add_fiber(fiber)
    hits = await store.search_fiber_summaries("不周山", limit=10)
    assert any(f.id == fiber.id for f in hits)


async def test_chinese_update_replaces_fts_terms(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="旧词太阳")
    await store.add_neuron(n)
    await store.update_neuron(replace(n, content="新词月亮"))
    old = await store.find_neurons(content_contains="太阳", limit=10)
    new = await store.find_neurons(content_contains="月亮", limit=10)
    assert all(h.id != n.id for h in old)
    assert any(h.id == n.id for h in new)


async def test_chinese_delete_removes_fts(store: SQLStorage) -> None:
    n = Neuron.create(type=NeuronType.CONCEPT, content="待删彩虹")
    await store.add_neuron(n)
    assert await store.delete_neuron(n.id)
    hits = await store.find_neurons(content_contains="彩虹", limit=10)
    assert all(h.id != n.id for h in hits)


async def test_chinese_query_with_quote_does_not_crash(store: SQLStorage) -> None:
    await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="起源世界与太古文明"))
    # A quote mixed into a CJK query must be escaped, not break FTS5 parsing.
    hits = await store.find_neurons(content_contains='起源" OR *', limit=10)
    assert isinstance(hits, list)


async def test_migration_v41_v42_backfills_cjk_index(tmp_path: Path) -> None:
    """A v41 database with raw (unspaced) FTS data is reindexed on upgrade."""
    db = tmp_path / "mig.db"

    # 1) Build a current-schema database, then rewind it to v41 and strip
    #    the FTS layer to simulate a pre-fix database.
    s = SQLStorage(SQLiteDialect(db))
    await s.initialize()
    brain = Brain.create(name="mig")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    # Insert a Chinese neuron while the new triggers are still active, then
    # drop the whole FTS layer so the old DB state has no index at all.
    n = Neuron.create(type=NeuronType.CONCEPT, content="迁移验证海底两万里")
    await store_add(s, n)
    await s.close()

    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("DROP TRIGGER IF EXISTS neurons_au")
    conn.execute("DROP TRIGGER IF EXISTS neurons_ad")
    conn.execute("DROP TRIGGER IF EXISTS neurons_ai")
    conn.execute("DROP TRIGGER IF EXISTS fibers_au")
    conn.execute("DROP TRIGGER IF EXISTS fibers_ad")
    conn.execute("DROP TRIGGER IF EXISTS fibers_ai")
    conn.execute("DROP TABLE IF EXISTS neurons_fts")
    conn.execute("DROP TABLE IF EXISTS fibers_fts")
    conn.execute("UPDATE schema_version SET version = 41")
    conn.commit()
    conn.close()

    # 2) Reopen: the 41 -> 42 migration must rebuild the FTS tables with
    #    cjk_spaced() triggers and backfill existing rows.
    s2 = SQLStorage(SQLiteDialect(db))
    await s2.initialize()
    brain2 = await s2.get_brain(brain.id)
    s2.set_brain(brain2.id)
    hits = await s2.find_neurons(content_contains="海底", limit=10)
    assert any("海底" in x.content for x in hits)
    await s2.close()


async def store_add(store: SQLStorage, neuron: Neuron) -> None:
    await store.add_neuron(neuron)


async def test_chinese_query_survives_extraction_into_recall(store: SQLStorage) -> None:
    """A Chinese question must reach the FTS index, not be dropped before it.

    The index side was already correct (#207), but ``extract_keywords`` used a
    Latin-only character class and returned ``[]`` for Han text, so recall
    answered "no relevant memories" for memories that were indexed. This walks
    the whole chain: question -> keywords -> FTS phrase -> stored summary.
    """
    from neural_memory.extraction.keywords import extract_keywords

    n = Neuron.create(type=NeuronType.CONCEPT, content="anchor")
    await store.add_neuron(n)
    fiber = Fiber.create(
        neuron_ids={n.id},
        synapse_ids=set(),
        anchor_neuron_id=n.id,
        summary="媒体投稿界面增加了定时投放功能",
    )
    await store.add_fiber(fiber)

    for question in ("媒体投稿", "投稿"):
        keywords = extract_keywords(question)
        assert keywords, f"{question!r} produced no keywords, so FTS is never queried"
        hits = await store.search_fiber_summaries(" ".join(keywords), limit=10)
        assert any(f.id == fiber.id for f in hits), (
            f"{question!r} -> keywords {keywords} did not reach the indexed fiber"
        )


async def test_content_word_is_what_reaches_the_index(store: SQLStorage) -> None:
    """A word made of characters that are particles elsewhere still works.

    ``用户``/``更新``/``问题`` each contain a character that is a particle on
    its own. Treating those characters as separators destroyed the word and
    left recall with nothing to search, which is the bug this guards against.
    """
    from neural_memory.extraction.keywords import extract_keywords

    n = Neuron.create(type=NeuronType.CONCEPT, content="anchor2")
    await store.add_neuron(n)
    fiber = Fiber.create(
        neuron_ids={n.id},
        synapse_ids=set(),
        anchor_neuron_id=n.id,
        summary="用户反馈更新后问题消失",
    )
    await store.add_fiber(fiber)

    for word in ("用户", "更新", "问题"):
        keywords = extract_keywords(word)
        assert keywords == [word], f"{word!r} -> {keywords}"
        hits = await store.search_fiber_summaries(" ".join(keywords), limit=10)
        assert any(f.id == fiber.id for f in hits), f"{word!r} did not reach the fiber"
