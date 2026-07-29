"""Three planner families: workflow, LM and graph search.

WorkflowPlanner returns a fixed plan, reliable but inflexible. LMPlanner
asks an LM for a JSON plan, flexible but parses-or-fails. GraphSearchPlanner
runs BFS over an action graph, rigorous on small domains.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Protocol, runtime_checkable

from agentlab.core.task import TaskSpec
from agentlab.planning.plan import Plan, PlanStep
from agentlab.protocols import BaseLM


@runtime_checkable
class Planner(Protocol):
    def plan(self, task: TaskSpec) -> Plan:
        """Return a plan for the given task."""
        ...


class WorkflowPlanner:
    """Planner that returns a fixed list of steps regardless of the task."""

    def __init__(self, steps: list[PlanStep]) -> None:
        self._steps = steps

    def plan(self, task: TaskSpec) -> Plan:
        """Return a plan containing a copy of the fixed steps."""
        return Plan(steps=list(self._steps))


_LM_PROMPT = """Plan steps for this task. Return a JSON array of objects with fields:
  - id (string)
  - description (string)
  - action_hint (string, optional)

Task goal: {goal}
Constraints: {constraints}

Return only valid JSON, no other text."""


class LMPlanner:
    """Planner that asks a language model for a JSON array of steps."""

    def __init__(self, lm: BaseLM) -> None:
        self._lm = lm

    def plan(self, task: TaskSpec) -> Plan:
        """Prompt the LM with the task goal and constraints and parse its JSON into a plan.

        Args:
            task: The task whose goal and constraints fill the prompt.

        Returns:
            The plan built from the LM's JSON step array.

        Raises:
            ValueError: If the LM response is not valid JSON or is not a JSON array.
        """
        prompt = _LM_PROMPT.format(
            goal=task.goal,
            constraints="; ".join(task.constraints) or "none",
        )
        response = self._lm.complete(prompt)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"LM returned non-JSON plan: {response!r}") from e
        if not isinstance(data, list):
            raise ValueError(f"LM plan must be a JSON array, got {type(data).__name__}")
        steps = [
            PlanStep(
                id=str(item.get("id", f"s{i}")),
                description=str(item["description"]),
                action_hint=str(item.get("action_hint", "")),
            )
            for i, item in enumerate(data)
        ]
        return Plan(steps=steps)


class GraphSearchPlanner:
    """BFS over a state-transition graph.

    `graph` maps state -> {action: next_state}. The planner looks for a path
    from task.inputs["start"] to task.expected_outputs[0].
    """

    def __init__(self, graph: dict[str, dict[str, str]]) -> None:
        self._graph = graph

    def plan(self, task: TaskSpec) -> Plan:
        """Search the graph from task.inputs["start"] to the first expected output.

        Args:
            task: The task supplying the start state and the goal state.

        Returns:
            A plan whose steps are the actions along the found path, empty when
            no path exists.
        """
        start = str(task.inputs.get("start", ""))
        goal = task.expected_outputs[0] if task.expected_outputs else ""
        path = self._bfs(start, goal)
        steps = [
            PlanStep(id=f"s{i}", description=f"take action {a}", action_hint=a)
            for i, a in enumerate(path)
        ]
        return Plan(steps=steps)

    def _bfs(self, start: str, goal: str) -> list[str]:
        if start == goal:
            return []
        queue: deque[tuple[str, list[str]]] = deque([(start, [])])
        seen: set[str] = {start}
        while queue:
            node, path = queue.popleft()
            for action, next_state in self._graph.get(node, {}).items():
                if next_state in seen:
                    continue
                new_path = path + [action]
                if next_state == goal:
                    return new_path
                seen.add(next_state)
                queue.append((next_state, new_path))
        return []
