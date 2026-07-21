from agentlab.memory import GraphMemory, Triple


def test_add_and_neighbors_one_hop():
    g = GraphMemory()
    g.add_triple(Triple("alice", "owes", "bank"))
    g.add_triple(Triple("bank", "charges", "fee"))
    nb = g.neighbors("alice", hops=1)
    assert any(t.object == "bank" for t in nb)


def test_neighbors_multi_hop():
    g = GraphMemory()
    g.add_triple(Triple("alice", "owes", "bank"))
    g.add_triple(Triple("bank", "charges", "fee"))
    nb = g.neighbors("alice", hops=2)
    objects = {t.object for t in nb} | {t.subject for t in nb}
    assert "fee" in objects


def test_query_substring():
    g = GraphMemory()
    g.add_triple(Triple("alice", "owes", "bank"))
    g.add_triple(Triple("bob", "owns", "shares"))
    hits = g.query("bank")
    assert len(hits) == 1
    assert "bank" in hits[0].content
