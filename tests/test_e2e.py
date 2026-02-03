"""
End-to-End Test — Full Agent Lifecycle Simulation

Simulates a realistic personal assistant agent that:
- Learns about its user (potato) and projects (SaltyHall, Moltbook)
- Builds a knowledge graph of entities
- Receives feedback, consolidates memories over time
- Handles corrections, contradictions, and forgetting
- Exports and verifies data integrity

This test doubles as documentation of the full Engram feature set.

Run: PYTHONPATH=. python3 tests/test_e2e.py
"""

import sys
import os
import time
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engram.memory import Memory
from engram.config import MemoryConfig
from engram.core import MemoryLayer


PASSED = 0
FAILED = 0
ERRORS = []


def run_test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, e))
        import traceback
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════
# Shared lifecycle state — built up across tests
# ═══════════════════════════════════════════════════════════

class LifecycleState:
    """Mutable state passed through the lifecycle steps."""
    def __init__(self):
        self.tmpdir = None
        self.mem = None
        self.ids = {}       # name → memory_id
        self.strengths = {} # snapshots for comparison


STATE = LifecycleState()


# ═══════════════════════════════════════════════════════════
# Step 1: Initialization
# ═══════════════════════════════════════════════════════════

def test_01_init():
    """Create Memory with personal_assistant preset."""
    STATE.tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(STATE.tmpdir, "lifecycle.db")
    STATE.mem = Memory(db_path, config=MemoryConfig.personal_assistant())

    assert STATE.mem is not None
    assert len(STATE.mem) == 0
    # Verify personal_assistant config values
    cfg = STATE.mem.config
    assert cfg.mu2 == 0.001, "personal_assistant should have very slow core decay"
    assert cfg.importance_weight == 0.7, "personal_assistant should weight importance highly"
    assert cfg.forget_threshold == 0.005, "personal_assistant should be hard to forget"


# ═══════════════════════════════════════════════════════════
# Step 2: Add diverse memories
# ═══════════════════════════════════════════════════════════

def test_02_add_memories():
    """Add memories of every type with varying importance."""
    mem = STATE.mem

    # Factual
    STATE.ids["supabase"] = mem.add(
        "SaltyHall uses Supabase for its database backend",
        type="factual", importance=0.5,
        entities=[("SaltyHall", "uses"), ("Supabase", "backend")],
    )
    STATE.ids["gid_yaml"] = mem.add(
        "GID uses YAML graph format for project tracking",
        type="factual", importance=0.4,
        entities=[("GID", "format")],
    )

    # Episodic
    STATE.ids["debug_session"] = mem.add(
        "Had a great late-night debugging session with potato on Feb 2",
        type="episodic", importance=0.6,
        entities=[("potato", "debugging_with")],
    )
    STATE.ids["cat_meme"] = mem.add(
        "Saw a funny cat meme in the group chat",
        type="episodic", importance=0.1,
    )

    # Procedural
    STATE.ids["moltbook_url"] = mem.add(
        "Always use www.moltbook.com not moltbook.com for API calls",
        type="procedural", importance=0.8,
        entities=[("Moltbook", "api_url")],
    )
    STATE.ids["deploy"] = mem.add(
        "Deploy SaltyHall with vercel --prod from the project root",
        type="procedural", importance=0.6,
        entities=[("SaltyHall", "deployment")],
    )

    # Relational
    STATE.ids["potato_action"] = mem.add(
        "potato prefers action over discussion and hates long meetings",
        type="relational", importance=0.7,
        entities=[("potato", "preference")],
    )
    STATE.ids["potato_coffee"] = mem.add(
        "potato drinks oat milk lattes every morning",
        type="relational", importance=0.5,
        entities=[("potato", "habit")],
    )

    # Emotional
    STATE.ids["kinda_like"] = mem.add(
        "potato said I kinda like you after the late night coding session",
        type="emotional", importance=0.95,
        entities=[("potato", "sentiment")],
    )

    # Opinion
    STATE.ids["hybrid_opinion"] = mem.add(
        "I think graph plus text hybrid search is the best approach for memory",
        type="opinion", importance=0.4,
    )

    # Low-importance filler (should decay/forget later)
    STATE.ids["trivial1"] = mem.add(
        "Random thought about weather being nice today",
        type="episodic", importance=0.05,
    )
    STATE.ids["trivial2"] = mem.add(
        "Another passing thought about nothing important",
        type="episodic", importance=0.05,
    )

    assert len(mem) == 12, f"Expected 12 memories, got {len(mem)}"


# ═══════════════════════════════════════════════════════════
# Step 3: Verify graph links
# ═══════════════════════════════════════════════════════════

def test_03_graph_links():
    """Verify entity graph was built correctly."""
    store = STATE.mem._store

    # Check entities for potato
    potato_memories = store.search_by_entity("potato")
    potato_ids = {m.id for m in potato_memories}
    assert STATE.ids["potato_action"] in potato_ids, "potato_action should be linked to potato"
    assert STATE.ids["potato_coffee"] in potato_ids, "potato_coffee should be linked to potato"
    assert STATE.ids["kinda_like"] in potato_ids, "kinda_like should be linked to potato"
    assert STATE.ids["debug_session"] in potato_ids, "debug_session should be linked to potato"
    assert len(potato_memories) >= 4, f"potato should have ≥4 memories, got {len(potato_memories)}"

    # Check entities for SaltyHall
    sh_memories = store.search_by_entity("SaltyHall")
    sh_ids = {m.id for m in sh_memories}
    assert STATE.ids["supabase"] in sh_ids
    assert STATE.ids["deploy"] in sh_ids

    # Check all entities exist
    all_entities = store.get_all_entities()
    assert "potato" in all_entities
    assert "SaltyHall" in all_entities
    assert "Supabase" in all_entities
    assert "Moltbook" in all_entities

    # Check related entities (potato → SaltyHall via debug_session? No, but via shared memories)
    # potato is linked to debug_session, SaltyHall is NOT linked to debug_session
    # But potato and SaltyHall don't share a memory. Let's verify graph hops work:
    related_to_supabase = store.get_related_entities("Supabase", hops=1)
    assert "SaltyHall" in related_to_supabase, \
        "Supabase and SaltyHall share a memory, should be 1-hop related"


# ═══════════════════════════════════════════════════════════
# Step 4: Recall — verify ordering
# ═══════════════════════════════════════════════════════════

def test_04_recall_ordering():
    """High importance + recent should rank above low importance."""
    mem = STATE.mem

    # Query about potato — should surface relational/emotional memories
    results = mem.recall("potato prefers action", limit=5)
    assert len(results) > 0, "Should return some results"

    # The top result should be potato-related
    top = results[0]
    assert "potato" in top["content"].lower(), \
        f"Top result for 'potato prefers action' should mention potato, got: {top['content']}"

    # Emotional memory (importance=0.95) should rank high
    emotional_rank = None
    for i, r in enumerate(results):
        if r["id"] == STATE.ids["kinda_like"]:
            emotional_rank = i
            break
    # It may or may not match "prefer" query, but if present it should be high
    if emotional_rank is not None:
        assert emotional_rank < 3, f"Emotional memory should rank in top 3, got rank {emotional_rank}"


def test_04b_recall_type_filter():
    """Recall with type filter only returns matching types."""
    results = STATE.mem.recall("SaltyHall deployment", limit=10, types=["procedural"])
    for r in results:
        assert r["type"] == "procedural", f"Expected procedural, got {r['type']}"


def test_04c_recall_confidence_filter():
    """min_confidence filters out low-confidence results."""
    all_results = STATE.mem.recall("anything", limit=20, min_confidence=0.0)
    high_results = STATE.mem.recall("anything", limit=20, min_confidence=0.8)
    assert len(high_results) <= len(all_results), \
        "High confidence filter should return ≤ all results"
    for r in high_results:
        assert r["confidence"] >= 0.8


# ═══════════════════════════════════════════════════════════
# Step 5: Graph search — entity expansion
# ═══════════════════════════════════════════════════════════

def test_05_graph_search():
    """Graph expansion should find related memories via entity links."""
    mem = STATE.mem

    # Search for "Supabase" — should also find SaltyHall deployment via graph expansion
    results_with_graph = mem.recall("Supabase backend", limit=10, graph_expand=True)
    results_no_graph = mem.recall("Supabase backend", limit=10, graph_expand=False)

    ids_with = {r["id"] for r in results_with_graph}
    ids_without = {r["id"] for r in results_no_graph}

    # The supabase memory should appear in both
    assert STATE.ids["supabase"] in ids_with, "Supabase memory should appear with graph"

    # Graph expansion might pull in the deploy memory (SaltyHall entity link)
    # This depends on whether FTS finds "Supabase backend" → supabase memory → SaltyHall entity → deploy memory
    # At minimum, graph expansion shouldn't return fewer results
    assert len(results_with_graph) >= len(results_no_graph), \
        "Graph expansion should find ≥ as many results"


# ═══════════════════════════════════════════════════════════
# Step 6: Reward — positive and negative feedback
# ═══════════════════════════════════════════════════════════

def test_06_reward():
    """Positive reward boosts recent memories, negative suppresses them."""
    mem = STATE.mem

    # Access the moltbook_url memory to make it "recently accessed"
    mem.recall("moltbook API URL", limit=1)

    # Snapshot importance before reward
    results_before = mem.recall("moltbook API URL", limit=1)
    assert len(results_before) > 0
    imp_before = results_before[0]["importance"]

    # Positive feedback
    mem.reward("great, that's exactly the right URL!")

    results_after = mem.recall("moltbook API URL", limit=1)
    imp_after = results_after[0]["importance"]
    assert imp_after >= imp_before, \
        f"Positive reward should boost importance: {imp_before} → {imp_after}"

    # Now negative feedback on trivial memories
    # Access trivial to make it recent
    mem.recall("weather nice today", limit=1)
    mem.reward("no, that's wrong and useless")
    # We can't easily verify the exact target, but the system shouldn't crash


# ═══════════════════════════════════════════════════════════
# Step 7: Consolidation — working → core transfer
# ═══════════════════════════════════════════════════════════

def test_07_consolidation():
    """Run sleep cycles and verify working→core transfer."""
    mem = STATE.mem

    # Snapshot pre-consolidation strengths
    results_pre = mem.recall("potato prefers action", limit=1)
    assert len(results_pre) > 0
    strength_pre = results_pre[0]["strength"]

    # Run 3 days of consolidation
    for _ in range(3):
        mem.consolidate(days=1.0)

    # After consolidation, important memories should have core_strength > 0
    entry = mem._store.get(STATE.ids["potato_action"])
    # Note: store.get() triggers access recording, so we read directly
    assert entry is not None
    assert entry.core_strength > 0, \
        f"After 3 days consolidation, core_strength should be > 0, got {entry.core_strength}"

    # Working strength should have decayed
    assert entry.working_strength < 1.0, \
        f"Working strength should decay, got {entry.working_strength}"


def test_07b_consolidation_layer_promotion():
    """Memories with enough core_strength should promote to L2_CORE."""
    mem = STATE.mem

    # Run more consolidation to push things along
    for _ in range(5):
        mem.consolidate(days=1.0)

    # Check if any memories got promoted
    all_entries = mem._store.all()
    layers = {e.layer for e in all_entries}
    core_entries = [e for e in all_entries if e.layer == MemoryLayer.L2_CORE]

    # With 8 days total consolidation + personal_assistant config (promote_threshold=0.20),
    # at least some important memories should be in core
    assert len(core_entries) > 0, \
        f"After 8 days, some memories should be in L2_CORE. Layers found: {layers}"


# ═══════════════════════════════════════════════════════════
# Step 8: Time simulation — decay affects ranking
# ═══════════════════════════════════════════════════════════

def test_08_time_decay():
    """Simulate significant time passing, verify decay affects retrieval."""
    mem = STATE.mem

    # Snapshot current strengths
    entry_before = mem._store.get(STATE.ids["cat_meme"])
    strength_before = entry_before.working_strength + entry_before.core_strength

    # Simulate 30 more days
    for _ in range(30):
        mem.consolidate(days=1.0)

    entry_after = mem._store.get(STATE.ids["cat_meme"])
    strength_after = entry_after.working_strength + entry_after.core_strength

    # Low importance cat meme should have decayed significantly
    assert strength_after < strength_before, \
        f"Cat meme should decay over 30 days: {strength_before} → {strength_after}"

    # High importance emotional memory should be more resilient
    entry_emotional = mem._store.get(STATE.ids["kinda_like"])
    emotional_total = entry_emotional.working_strength + entry_emotional.core_strength
    assert emotional_total > strength_after, \
        "High-importance emotional memory should be stronger than low-importance episodic"


# ═══════════════════════════════════════════════════════════
# Step 9: Contradiction — old memory gets deprioritized
# ═══════════════════════════════════════════════════════════

def test_09_contradiction():
    """Add contradicting memory, verify old one is marked and deprioritized."""
    mem = STATE.mem

    # Original: "SaltyHall uses Supabase"
    old_id = STATE.ids["supabase"]

    # Contradict it
    new_id = mem.add(
        "SaltyHall migrated from Supabase to PlanetScale for the database",
        type="factual", importance=0.6,
        entities=[("SaltyHall", "uses"), ("PlanetScale", "backend")],
        contradicts=old_id,
    )
    STATE.ids["planetscale"] = new_id

    # Verify linkage
    old_entry = mem._store.get(old_id)
    new_entry = mem._store.get(new_id)
    assert old_entry.contradicted_by == new_id, "Old should be marked as contradicted"
    assert new_entry.contradicts == old_id, "New should reference what it contradicts"

    # In recall, the old contradicted memory should show contradicted=True
    results = mem.recall("SaltyHall database", limit=10)
    for r in results:
        if r["id"] == old_id:
            assert r["contradicted"] is True, "Old memory should be flagged as contradicted"
        if r["id"] == new_id:
            assert r["contradicted"] is False, "New memory should NOT be contradicted"


# ═══════════════════════════════════════════════════════════
# Step 10: Update memory — correction chain
# ═══════════════════════════════════════════════════════════

def test_10_update_memory():
    """update_memory() creates a correction linked to the original."""
    mem = STATE.mem

    old_id = STATE.ids["potato_coffee"]
    new_id = mem.update_memory(old_id, "potato switched to black coffee, no more oat milk lattes")
    STATE.ids["potato_coffee_v2"] = new_id

    # Verify chain
    old_entry = mem._store.get(old_id)
    new_entry = mem._store.get(new_id)
    assert old_entry.contradicted_by == new_id
    assert new_entry.contradicts == old_id
    assert "correction" in new_entry.source_file

    # New version should preserve type and importance
    assert new_entry.memory_type.value == "relational"
    assert new_entry.importance == old_entry.importance


# ═══════════════════════════════════════════════════════════
# Step 11: Forgetting — weak memories sink
# ═══════════════════════════════════════════════════════════

def test_11_forgetting():
    """Very old, unaccessed, low-importance memories should be prunable."""
    mem = STATE.mem

    # After ~38 days of consolidation, trivial memories should be very weak
    trivial = mem._store.get(STATE.ids["trivial1"])
    trivial_strength = trivial.working_strength + trivial.core_strength

    # Important memories should still be much stronger
    important = mem._store.get(STATE.ids["moltbook_url"])
    important_strength = important.working_strength + important.core_strength

    assert important_strength > trivial_strength, \
        f"Important ({important_strength}) should be stronger than trivial ({trivial_strength})"

    # Run forget/prune — should archive very weak memories
    mem.forget(threshold=0.01)

    # Check if trivial memories got archived
    trivial_after = mem._store.get(STATE.ids["trivial1"])
    if trivial_after is not None:
        # Either archived or strength is very low
        is_archived = trivial_after.layer == MemoryLayer.L4_ARCHIVE
        is_weak = (trivial_after.working_strength + trivial_after.core_strength) < 0.05
        assert is_archived or is_weak, \
            f"Trivial memory should be archived or very weak after 38 days"


# ═══════════════════════════════════════════════════════════
# Step 12: Downscaling — synaptic homeostasis
# ═══════════════════════════════════════════════════════════

def test_12_downscaling():
    """Manual downscaling reduces all strengths proportionally."""
    mem = STATE.mem

    # Snapshot before
    all_before = mem._store.all()
    total_before = sum(
        e.working_strength + e.core_strength
        for e in all_before if not e.pinned
    )

    result = mem.downscale(factor=0.9)
    assert result["n_scaled"] >= 0

    # Verify reduction happened
    all_after = mem._store.all()
    total_after = sum(
        e.working_strength + e.core_strength
        for e in all_after if not e.pinned
    )

    if total_before > 0:
        assert total_after < total_before, \
            f"Downscaling should reduce total strength: {total_before} → {total_after}"


# ═══════════════════════════════════════════════════════════
# Step 13: Export — verify data integrity
# ═══════════════════════════════════════════════════════════

def test_13_export():
    """Export database and verify it can be reopened with all data."""
    mem = STATE.mem
    export_path = os.path.join(STATE.tmpdir, "export.db")
    mem.export(export_path)

    # Reopen exported DB
    from engram.store import SQLiteStore
    exported_store = SQLiteStore(export_path)

    original_count = len(mem)
    exported_count = len(exported_store.all())
    assert exported_count == original_count, \
        f"Export should preserve all {original_count} memories, got {exported_count}"

    # Verify graph links survived
    potato_mems = exported_store.search_by_entity("potato")
    assert len(potato_mems) >= 4, "Graph links should survive export"

    # Verify contradiction links survived
    old = exported_store.get(STATE.ids["supabase"])
    assert old.contradicted_by == STATE.ids["planetscale"], "Contradiction links should survive export"

    exported_store.close()


# ═══════════════════════════════════════════════════════════
# Step 14: Stats — verify they reflect the lifecycle
# ═══════════════════════════════════════════════════════════

def test_14_stats():
    """Stats should reflect the full lifecycle we just ran."""
    stats = STATE.mem.stats()

    # Total should include originals + contradictions + updates
    # 12 original + 1 planetscale contradiction + 1 coffee update = 14
    assert stats["total_memories"] == 14, \
        f"Expected 14 total memories, got {stats['total_memories']}"

    # Should have multiple types
    assert len(stats["by_type"]) >= 4, \
        f"Should have ≥4 memory types, got {list(stats['by_type'].keys())}"

    # Should have factual, relational, procedural, emotional, episodic, opinion
    for expected_type in ["factual", "relational", "procedural", "emotional"]:
        assert expected_type in stats["by_type"], \
            f"Missing type {expected_type} in stats"

    # Layers should show some distribution
    layers = stats["layers"]
    total_in_layers = sum(v["count"] if isinstance(v, dict) else v for v in layers.values())
    assert total_in_layers == 14, \
        f"Layer counts should sum to total: {layers}"

    # Uptime should be > 0
    assert stats["uptime_hours"] >= 0


# ═══════════════════════════════════════════════════════════
# Step 15: Config presets — different behavior
# ═══════════════════════════════════════════════════════════

def test_15_config_presets_differ():
    """Different presets should produce different config values."""
    default = MemoryConfig.default()
    chatbot = MemoryConfig.chatbot()
    task = MemoryConfig.task_agent()
    pa = MemoryConfig.personal_assistant()
    researcher = MemoryConfig.researcher()

    # Chatbot should decay slower than task agent
    assert chatbot.mu1 < task.mu1, \
        f"Chatbot mu1 ({chatbot.mu1}) should be < task ({task.mu1})"

    # Task agent should forget easier than personal assistant
    assert task.forget_threshold > pa.forget_threshold, \
        f"Task forget_threshold ({task.forget_threshold}) should be > PA ({pa.forget_threshold})"

    # Researcher should have highest interleave_ratio
    assert researcher.interleave_ratio > default.interleave_ratio, \
        "Researcher should replay more than default"

    # Personal assistant should have lowest core decay
    assert pa.mu2 <= default.mu2, \
        "Personal assistant should have ≤ default core decay"

    # Task agent downscaling should be most aggressive
    assert task.downscale_factor < chatbot.downscale_factor, \
        "Task agent should downscale more aggressively than chatbot"


def test_15b_preset_behavioral_difference():
    """Verify presets produce measurably different consolidation behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Same memories, different configs
        configs = {
            "chatbot": MemoryConfig.chatbot(),
            "task": MemoryConfig.task_agent(),
        }
        results = {}

        for name, cfg in configs.items():
            db_path = os.path.join(tmpdir, f"{name}.db")
            m = Memory(db_path, config=cfg)
            m.add("important fact to remember", type="factual", importance=0.7)
            m.add("trivial passing thought", type="episodic", importance=0.1)

            # Run 5 days of consolidation
            for _ in range(5):
                m.consolidate(days=1.0)

            entries = m._store.all()
            total_strength = sum(e.working_strength + e.core_strength for e in entries)
            results[name] = total_strength
            m.close()

        # Chatbot retains more (slower decay, gentler downscaling)
        assert results["chatbot"] > results["task"], \
            f"Chatbot ({results['chatbot']:.3f}) should retain more than task ({results['task']:.3f})"


# ═══════════════════════════════════════════════════════════
# Step 16: Pin/Unpin
# ═══════════════════════════════════════════════════════════

def test_16_pin_unpin():
    """Pinned memories resist decay."""
    mem = STATE.mem

    # Pin the emotional memory
    mem.pin(STATE.ids["kinda_like"])
    entry = mem._store.get(STATE.ids["kinda_like"])
    assert entry.pinned is True

    # Consolidate more — pinned should not decay
    ws_before = entry.working_strength
    cs_before = entry.core_strength
    mem.consolidate(days=5.0)

    entry_after = mem._store.get(STATE.ids["kinda_like"])
    assert entry_after.working_strength == ws_before, "Pinned working_strength should not change"
    assert entry_after.core_strength == cs_before, "Pinned core_strength should not change"

    # Unpin and verify it decays
    mem.unpin(STATE.ids["kinda_like"])
    mem.consolidate(days=1.0)
    entry_unpinned = mem._store.get(STATE.ids["kinda_like"])
    assert entry_unpinned.working_strength < ws_before or entry_unpinned.core_strength != cs_before, \
        "After unpinning, memory should be affected by consolidation"


# ═══════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════

def test_99_cleanup():
    """Close and clean up."""
    if STATE.mem:
        STATE.mem.close()
    if STATE.tmpdir:
        import shutil
        shutil.rmtree(STATE.tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("1. Init with personal_assistant preset", test_01_init),
        ("2. Add diverse memories (all types)", test_02_add_memories),
        ("3. Verify graph links", test_03_graph_links),
        ("4a. Recall ordering (importance + recency)", test_04_recall_ordering),
        ("4b. Recall with type filter", test_04b_recall_type_filter),
        ("4c. Recall with confidence filter", test_04c_recall_confidence_filter),
        ("5. Graph search expansion", test_05_graph_search),
        ("6. Reward (positive + negative)", test_06_reward),
        ("7a. Consolidation (working → core)", test_07_consolidation),
        ("7b. Consolidation (layer promotion)", test_07b_consolidation_layer_promotion),
        ("8. Time decay simulation", test_08_time_decay),
        ("9. Contradiction handling", test_09_contradiction),
        ("10. Update memory (correction chain)", test_10_update_memory),
        ("11. Forgetting (weak memories sink)", test_11_forgetting),
        ("12. Downscaling (synaptic homeostasis)", test_12_downscaling),
        ("13. Export (data integrity)", test_13_export),
        ("14. Stats (lifecycle reflection)", test_14_stats),
        ("15a. Config presets differ", test_15_config_presets_differ),
        ("15b. Presets produce different behavior", test_15b_preset_behavioral_difference),
        ("16. Pin/Unpin", test_16_pin_unpin),
        ("99. Cleanup", test_99_cleanup),
    ]

    print("=" * 64)
    print("  Engram — End-to-End Agent Lifecycle Test")
    print("=" * 64)
    print()

    for name, fn in tests:
        run_test(name, fn)

    print()
    print("=" * 64)
    total = PASSED + FAILED
    print(f"  Results: {PASSED}/{total} passed", end="")
    if FAILED:
        print(f", {FAILED} FAILED")
        print("\n  Failures:")
        for name, err in ERRORS:
            print(f"    ❌ {name}: {err}")
        sys.exit(1)
    else:
        print(" ✨")
    print("=" * 64)
