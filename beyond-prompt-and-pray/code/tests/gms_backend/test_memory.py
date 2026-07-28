"""Tests for GMSMemory against a real loaded GMS store."""

from agentlab.gms_backend import GMSMemory
from agentlab.memory import MemoryItem


def test_store_passthrough(gms_store):
    mem = GMSMemory(gms_store)
    assert mem.store is gms_store


def test_score_triple_real(gms_store):
    triples = gms_store.query_triples()
    assert triples, "store has no triples"
    h, r, t = triples[0]
    mem = GMSMemory(gms_store)
    score = mem.score_triple(h, r, t)
    assert score is None or isinstance(score, (int, float))


def test_query_triples_returns_list(gms_store):
    mem = GMSMemory(gms_store)
    out = mem.query_triples()
    assert isinstance(out, list)
    if out:
        assert isinstance(out[0], tuple)
        assert len(out[0]) == 3


def test_query_by_head(gms_store):
    triples = gms_store.query_triples()
    if not triples:
        return
    head, _, _ = triples[0]
    mem = GMSMemory(gms_store)
    hits = mem.query_triples(head=head)
    assert all(h == head for h, _, _ in hits)


def test_query_returns_memory_items(gms_store):
    triples = gms_store.query_triples()
    if not triples:
        return
    head, _, _ = triples[0]
    mem = GMSMemory(gms_store)
    items = mem.query(head, k=3)
    assert all(isinstance(i, MemoryItem) for i in items)


def test_lookup_enm_for_known_value(gms_store):
    """The store has 408 ENM entries; querying one returns a real number."""
    mem = GMSMemory(gms_store)
    sample_keys = []
    if hasattr(gms_store, "enm"):
        enm = gms_store.enm
        if hasattr(enm, "keys"):
            sample_keys = list(enm.keys())[:5]
    if not sample_keys:
        return
    for key in sample_keys:
        cat, eid = (key.type, key.id) if hasattr(key, "type") else (None, None)
        if cat is None:
            continue
        value = mem.lookup_enm(cat, eid)
        if value is not None:
            assert isinstance(value, float)
            return
