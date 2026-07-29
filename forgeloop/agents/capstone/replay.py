"""Replay a governed run from its audit trail (Chapters 4 and 12).

The harness seals one ``AuditEvent`` per step: the proposed action as a JSON
dict (``Action.model_dump()``) and a hash of the state *before* the step. This
module reconstructs the run from that trail using the Chapter 4 primitives:

* ``parse_action`` turns each stored action dict back into a typed ``Action``
  (dispatching on the ``kind`` discriminator) -- the replay-from-JSON path
  Chapter 4 introduces and the audit log relies on.
* ``AgentState.from_dict`` reconstructs the pre-step state. Because that
  round-trip is lossless, re-hashing the reconstructed state reproduces the
  exact ``state_hash`` the audit chain sealed -- which is what lets the hash
  chain (Chapter 12) be verified against replayed state rather than trusted.

``replay_run`` checks both for every step and confirms the chain still verifies.
It replays the most recent run recorded on the audit logger -- i.e. the
trajectory just returned by ``harness.run`` -- selected by ``run_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forgeloop.agents.audit import AuditLogger
from forgeloop.agents.core import AgentState, parse_action
from forgeloop.agents.evaluation.trajectory import Trajectory

# Reuse the harness's exact state-hashing so a reconstructed state's hash is
# comparable to the one the audit chain sealed (same algorithm, no drift).
from forgeloop.agents.governance.harness import _hash_state


@dataclass(frozen=True)
class ReplayStep:
    """Per-step outcome of replaying one sealed audit event.

    Attributes:
        step: The step index.
        action_kind: The action discriminator recovered for the step.
        action_roundtrips: Whether parse_action reproduces the sealed typed action.
        state_hash_matches: Whether the round-tripped state reproduces the sealed hash.
    """

    step: int
    action_kind: str
    action_roundtrips: bool      # parse_action(JSON) reproduces the typed action
    state_hash_matches: bool     # from_dict round-trip reproduces the sealed hash


@dataclass
class ReplayReport:
    """Summary of a replayed run and whether it reconstructs exactly.

    Attributes:
        steps: Number of steps replayed.
        actions_reconstructed: Count of steps whose action round-tripped.
        state_hashes_matched: Count of steps whose state hash matched the seal.
        chain_verifies: Whether the audit hash chain still verifies.
        detail: Per-step ReplayStep records.
    """

    steps: int
    actions_reconstructed: int
    state_hashes_matched: int
    chain_verifies: bool
    detail: list[ReplayStep] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff every step replayed exactly and the chain still verifies."""
        return (
            self.chain_verifies
            and self.actions_reconstructed == self.steps
            and self.state_hashes_matched == self.steps
        )


def replay_run(trajectory: Trajectory, audit: AuditLogger) -> ReplayReport:
    """Replay ``trajectory`` from ``audit`` and verify the reconstruction.

    For each step the proposed action is rebuilt from the sealed JSON via
    :func:`parse_action`, and the pre-step state is round-tripped through
    :meth:`AgentState.from_dict`; the reconstructed state's hash is compared to
    the sealed ``state_hash``. Returns a :class:`ReplayReport`.

    The audit logger accumulates events across every run it has recorded, so
    the events for *this* run are selected by the ``run_id`` of the most recent
    event. Call this immediately after the ``harness.run`` that produced
    ``trajectory``.
    """
    sealed = audit.events
    if not sealed:
        raise ValueError("audit log is empty; nothing to replay")

    run_id = sealed[-1].event.run_id
    events = [s.event for s in sealed if s.event.run_id == run_id]
    if len(events) != len(trajectory.records):
        raise ValueError(
            f"audit run {run_id!r} has {len(events)} events but the trajectory has "
            f"{len(trajectory.records)} records; replay_run expects the trajectory "
            "returned by the most recent harness.run on this audit logger."
        )

    detail: list[ReplayStep] = []
    for record, event in zip(trajectory.records, events):
        # 1. Rebuild the typed action from the sealed JSON (Chapter 4 replay).
        action = parse_action(event.proposed_action)
        action_roundtrips = (
            action.kind == record.action.kind
            and action.model_dump() == event.proposed_action
        )
        # 2. Round-trip the pre-step state and confirm the lossless reconstruction
        #    reproduces the hash the chain sealed.
        restored = AgentState.from_dict(record.state_before.to_dict())
        state_hash_matches = _hash_state(restored) == event.state_hash

        detail.append(
            ReplayStep(
                step=event.step,
                action_kind=action.kind,
                action_roundtrips=action_roundtrips,
                state_hash_matches=state_hash_matches,
            )
        )

    return ReplayReport(
        steps=len(detail),
        actions_reconstructed=sum(d.action_roundtrips for d in detail),
        state_hashes_matched=sum(d.state_hash_matches for d in detail),
        chain_verifies=audit.verify(),
        detail=detail,
    )
