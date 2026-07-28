"""Prompt-and-pray baseline vs. the engineered Chapter 15 capstone agent.

A self-contained benchmark module. It imports the engineered agent and its tool
schemas from ``agentlab`` but does not modify any of them: the prompted baseline,
its model backends, and the comparison runner all live here.

The baseline is the same banking-complaint task built with one instruction-tuned
model and prompts --- no trained classifier head, no GMS guard, no Graph RAG, no
LoRA, no draft verifier, no governance gates. We run it under two model backends,
a local Qwen2.5-3B (the same small model the engineered agent uses internally)
and a larger frontier model (Anthropic Sonnet), so the comparison separates two
questions: how much does the *engineering* buy, and how much does a *bigger
model* buy on its own.
"""
