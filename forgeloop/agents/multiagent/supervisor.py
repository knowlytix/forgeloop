"""Supervisor: routes subtasks to named workers and aggregates responses."""

from __future__ import annotations

from typing import Any

from forgeloop.agents.multiagent.message import AgentMessage, MessageBus
from forgeloop.agents.multiagent.worker import Worker


class Supervisor:
    """Routes delegate messages to named workers and records them on a message bus."""

    def __init__(self, name: str, workers: list[Worker], bus: MessageBus | None = None) -> None:
        self.name = name
        self._workers: dict[str, Worker] = {w.name: w for w in workers}
        self._bus = bus if bus is not None else MessageBus()

    @property
    def bus(self) -> MessageBus:
        return self._bus

    def workers(self) -> list[str]:
        """Return the names of the registered workers."""
        return list(self._workers)

    def delegate(
        self,
        worker_name: str,
        goal: str,
        inputs: dict[str, Any] | None = None,
        constraints: list[str] | None = None,
        max_steps: int = 16,
    ) -> AgentMessage:
        """Send a delegate message to a worker and return its response.

        Both the request and the response are recorded on the bus.

        Args:
            worker_name: Name of the worker to delegate to.
            goal: The goal for the worker's task.
            inputs: Optional task inputs passed in the payload.
            constraints: Optional task constraints passed in the payload.
            max_steps: Maximum steps the worker's harness may run.

        Returns:
            The worker's response message.

        Raises:
            KeyError: If worker_name is not a registered worker.
        """
        if worker_name not in self._workers:
            raise KeyError(f"unknown worker {worker_name!r}")
        request = AgentMessage(
            sender=self.name,
            receiver=worker_name,
            message_type="delegate",
            payload={
                "goal": goal,
                "inputs": inputs or {},
                "constraints": constraints or [],
            },
        )
        self._bus.send(request)
        response = self._workers[worker_name].handle(request, max_steps=max_steps)
        self._bus.send(response)
        return response
