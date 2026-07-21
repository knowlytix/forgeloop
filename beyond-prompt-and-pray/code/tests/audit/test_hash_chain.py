import time

from agentlab.audit import AuditEvent, HashChain, SealedEvent
from agentlab.audit.hash_chain import GENESIS, verify_chain


def _ev(step: int) -> AuditEvent:
    return AuditEvent(
        run_id="r1",
        step=step,
        timestamp=time.time(),
        state_hash=f"state-{step:02d}",
        proposed_action={"kind": "tool_call", "tool_name": "ping"},
        observation={"success": True},
        final_state_status="running",
    )


def test_chain_starts_at_genesis():
    chain = HashChain()
    s1 = chain.seal(_ev(0))
    assert s1.prev_hash == GENESIS


def test_chain_links_in_order():
    chain = HashChain()
    s1 = chain.seal(_ev(0))
    s2 = chain.seal(_ev(1))
    assert s2.prev_hash == s1.event_hash


def test_verify_clean_chain():
    chain = HashChain()
    sealed = [chain.seal(_ev(i)) for i in range(4)]
    assert verify_chain(sealed) is True


def test_verify_detects_tampered_middle():
    chain = HashChain()
    sealed = [chain.seal(_ev(i)) for i in range(4)]
    bad_event = AuditEvent(
        run_id="r1",
        step=99,
        timestamp=0.0,
        state_hash="x",
        proposed_action={},
        observation={},
        final_state_status="done",
    )
    tampered = list(sealed)
    tampered[2] = SealedEvent(
        event=bad_event,
        prev_hash=sealed[2].prev_hash,
        event_hash=sealed[2].event_hash,
    )
    assert verify_chain(tampered) is False


def test_verify_detects_reordering():
    chain = HashChain()
    sealed = [chain.seal(_ev(i)) for i in range(3)]
    reordered = [sealed[0], sealed[2], sealed[1]]
    assert verify_chain(reordered) is False
