import time

from agentlab.audit import AuditEvent, AuditLogger


def _ev(step: int) -> AuditEvent:
    return AuditEvent(
        run_id="r1",
        step=step,
        timestamp=time.time(),
        state_hash=f"s-{step}",
        proposed_action={"kind": "tool_call"},
        observation={},
        final_state_status="running",
    )


def test_logger_appends_in_order():
    log = AuditLogger()
    s1 = log.log(_ev(0))
    s2 = log.log(_ev(1))
    assert log.events[0] == s1
    assert log.events[1] == s2


def test_logger_head_advances():
    log = AuditLogger()
    log.log(_ev(0))
    h1 = log.head()
    log.log(_ev(1))
    h2 = log.head()
    assert h1 != h2


def test_logger_verify_ok():
    log = AuditLogger()
    for i in range(3):
        log.log(_ev(i))
    assert log.verify() is True


def test_logger_replay_returns_events_in_order():
    log = AuditLogger()
    log.log(_ev(0))
    log.log(_ev(1))
    replayed = log.replay()
    assert [e.step for e in replayed] == [0, 1]
