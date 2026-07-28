from agentlab.memory import MemoryItem, MemoryKind, ShortTermMemory, WorkingMemory


def _item(text: str) -> MemoryItem:
    return MemoryItem(content=text, kind=MemoryKind.SHORT_TERM)


def test_add_and_query():
    m = ShortTermMemory(maxlen=10)
    m.add(_item("the cat sat"))
    m.add(_item("the dog ran"))
    hits = m.query("cat")
    assert len(hits) == 1
    assert "cat" in hits[0].content


def test_maxlen_evicts_oldest():
    m = ShortTermMemory(maxlen=2)
    m.add(_item("a"))
    m.add(_item("b"))
    m.add(_item("c"))
    assert len(m) == 2
    hits = m.query("a")
    assert hits == []


def test_query_returns_most_recent_first():
    m = ShortTermMemory()
    m.add(_item("hello first"))
    m.add(_item("hello second"))
    hits = m.query("hello")
    assert hits[0].content == "hello second"


def test_working_memory_defaults():
    wm = WorkingMemory(task_id="t1")
    assert wm.facts == {}
    assert wm.open_questions == []
