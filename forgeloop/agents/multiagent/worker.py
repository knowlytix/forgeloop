"""Worker: a named agent with its own GovernanceHarness.

Each worker has a capability label and handles delegate messages by
building a TaskSpec from the payload and running its own harness. The
result is wrapped in a response message.
"""

from __future__ import annotations

from forgeloop.agents.core.task import TaskSpec
from forgeloop.agents.governance.harness import GovernanceHarness
from forgeloop.agents.multiagent.message import AgentMessage


class Worker:
    """A named agent that handles delegate messages by running its governance harness."""

    def __init__(self, name: str, capability: str, harness: GovernanceHarness) -> None:
        self.name = name
        self.capability = capability
        self._harness = harness

    @property
    def harness(self) -> GovernanceHarness:
        return self._harness

    def handle(self, message: AgentMessage, max_steps: int = 16) -> AgentMessage:
        """Run the harness on a delegate message and return a response message.

        Builds a TaskSpec from the message payload, runs the harness and wraps
        the resulting status, final output and step count in a response. A
        non-delegate message yields a rejected message instead.

        Args:
            message: The incoming message; only "delegate" types are handled.
            max_steps: Maximum steps the harness may run.

        Returns:
            A "response" message on success, or a "rejected" message when the
            message type is not "delegate".
        """
        if message.message_type != "delegate":
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="rejected",
                payload={"reason": f"unsupported message_type {message.message_type!r}"},
                parent_id=message.id,
            )
        task = TaskSpec(
            goal=str(message.payload.get("goal", "")),
            inputs=dict(message.payload.get("inputs", {})),
            constraints=list(message.payload.get("constraints", [])),
        )
        traj = self._harness.run(task, max_steps=max_steps)
        return AgentMessage(
            sender=self.name,
            receiver=message.sender,
            message_type="response",
            payload={
                "status": traj.final_state.status,
                "final_output": traj.final_state.final_output,
                "steps": len(traj.records),
            },
            parent_id=message.id,
        )
