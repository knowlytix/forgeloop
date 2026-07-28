"""Supervisor-worker multi-agent pattern. See Chapter 14.

When to reach for multi-agent: when you have genuinely distinct
capabilities that benefit from separate prompts, tools or governance
policies. When NOT to: when a well-engineered single agent could do the
job. Multi-agent adds latency, more failure surface and harder audit.
"""

from agentlab.multiagent.message import AgentMessage, MessageBus
from agentlab.multiagent.supervisor import Supervisor
from agentlab.multiagent.worker import Worker

__all__ = ["AgentMessage", "MessageBus", "Supervisor", "Worker"]
