from agentlab.multiagent import AgentMessage, MessageBus


def test_message_has_id_and_timestamp():
    m = AgentMessage(sender="a", receiver="b", message_type="hi", payload={})
    assert m.id
    assert m.timestamp > 0


def test_message_is_frozen():
    m = AgentMessage(sender="a", receiver="b", message_type="hi", payload={})
    import pytest
    with pytest.raises(Exception):
        m.sender = "c"


def test_bus_routes_by_receiver():
    bus = MessageBus()
    bus.send(AgentMessage(sender="a", receiver="b", message_type="x", payload={}))
    bus.send(AgentMessage(sender="a", receiver="c", message_type="x", payload={}))
    assert len(bus.messages_for("b")) == 1
    assert len(bus.messages_for("c")) == 1


def test_bus_thread_collects_replies():
    bus = MessageBus()
    root = AgentMessage(sender="a", receiver="b", message_type="x", payload={})
    bus.send(root)
    bus.send(AgentMessage(sender="b", receiver="a", message_type="r", payload={}, parent_id=root.id))
    thread = bus.thread(root.id)
    assert len(thread) == 2
