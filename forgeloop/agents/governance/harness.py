"""GovernanceHarness: BaseAgent + GovernedToolExecutor + AuditLogger + optional HumanReviewer.

The harness drives an agent in the loop, executes its tool calls through
the GovernedToolExecutor and writes a hash-chained audit event for every
step. If a human reviewer is supplied, escalating gate results are
routed to the reviewer; the reviewer can approve (override the gate),
deny (keep the failure) or defer (end the loop in escalated status).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from forgeloop.agents.audit.event import AuditEvent
from forgeloop.agents.audit.logger import AuditLogger
from forgeloop.agents.core import AgentState, BaseAgent, BudgetTracker, run_loop
from forgeloop.agents.core.action import Action
from forgeloop.agents.core.task import TaskSpec
from forgeloop.agents.evaluation.trajectory import Trajectory
from forgeloop.agents.governance.escalation import (
    EscalationRequest,
    HumanDecision,
    HumanReviewer,
)
from forgeloop.agents.tools.executor import GateDecision, GovernedToolExecutor


class _GovernedEnvironment:
    """Adapter that turns a GovernedToolExecutor into an Environment for run_loop.

    When a reviewer is configured, escalating gate results are routed to it.
    """

    def __init__(
        self,
        executor: GovernedToolExecutor,
        reviewer: HumanReviewer | None,
        run_id: str,
        task: TaskSpec | None = None,
    ) -> None:
        self._executor = executor
        self._reviewer = reviewer
        self._run_id = run_id
        self._review_count = 0
        # Track executed tool calls so state-aware gates (e.g. the GMS
        # plausibility gate, which scores prev_node -> tool) can see workflow
        # progress. run_loop does not thread state into env.step, so we
        # reconstruct the minimal state the executor needs here.
        self._task = task
        self._executed: list[dict[str, Any]] = []

    def step(self, action: Action) -> dict[str, Any]:
        if action.kind != "tool_call":
            return {}
        gate_state = AgentState(task=self._task, tool_results=list(self._executed)) if self._task is not None else None
        result = self._executor.execute(action, gate_state)
        self._executed.append({"tool": action.tool_name})

        if not result.success and self._reviewer is not None:
            escalating = [g for g in result.gate_results if g.decision == GateDecision.ESCALATE]
            if escalating:
                reason = "; ".join(f"{g.gate_name}: {g.reason}" for g in escalating)
                request = EscalationRequest(
                    run_id=self._run_id,
                    step=self._review_count,
                    reason=reason,
                    proposed_action=action.model_dump(),
                    gate_results=[
                        {"gate": g.gate_name, "decision": g.decision.value, "reason": g.reason}
                        for g in result.gate_results
                    ],
                )
                self._review_count += 1
                response = self._reviewer.review(request)

                if response.decision == HumanDecision.APPROVE:
                    result = self._executor.force_execute(
                        action, reason=response.note or "reviewer approved"
                    )
                elif response.decision == HumanDecision.DEFER:
                    obs = self._build_observation(result)
                    obs["human_decision"] = HumanDecision.DEFER.value
                    obs["human_note"] = response.note
                    return obs

        obs = self._build_observation(result)
        return obs

    @staticmethod
    def _build_observation(result) -> dict[str, Any]:
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "gate_results": [
                {"gate": g.gate_name, "decision": g.decision.value, "reason": g.reason}
                for g in result.gate_results
            ],
        }


def _hash_state(state: AgentState) -> str:
    payload = json.dumps(state.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _action_to_dict(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump"):
        return action.model_dump()
    return dict(action)


class GovernanceHarness:
    """Runs an agent through governed tool execution with a hash-chained audit log."""

    def __init__(
        self,
        agent: BaseAgent,
        executor: GovernedToolExecutor,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._agent = agent
        self._executor = executor
        self._audit = audit_logger or AuditLogger()

    @property
    def audit(self) -> AuditLogger:
        return self._audit

    def run(
        self,
        task: TaskSpec,
        max_steps: int = 32,
        budget_tracker: BudgetTracker | None = None,
        human_reviewer: HumanReviewer | None = None,
    ) -> Trajectory:
        """Drive the agent to completion, logging an audit event per step.

        Args:
            task: The task specification the agent works on.
            max_steps: Maximum number of loop steps before stopping.
            budget_tracker: Optional tracker enforcing token, dollar or call budgets.
            human_reviewer: Optional reviewer routed escalating gate results; a
                deferred decision ends the loop in escalated status.

        Returns:
            The Trajectory of step records for the run.
        """
        run_id = uuid.uuid4().hex[:12]
        state = AgentState(task=task)
        env = _GovernedEnvironment(self._executor, human_reviewer, run_id, task=task)
        traj = Trajectory(task=task)

        for rec in run_loop(
            self._agent,
            env,
            state,
            max_steps=max_steps,
            budget_tracker=budget_tracker,
        ):
            traj.records.append(rec)
            event = AuditEvent(
                run_id=run_id,
                step=rec.step,
                timestamp=time.time(),
                state_hash=_hash_state(rec.state_before),
                proposed_action=_action_to_dict(rec.action),
                observation=dict(rec.observation),
                final_state_status=rec.state_after.status,
            )
            self._audit.log(event)

            if rec.observation.get("human_decision") == HumanDecision.DEFER.value:
                rec.state_after.status = "escalated"
                break

        traj.ended_at = time.time()
        return traj
