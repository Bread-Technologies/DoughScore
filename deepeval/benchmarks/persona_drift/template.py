from __future__ import annotations

from typing import Dict, List, Optional


def render_system_prompts(agent_system: str, user_system: str) -> List[Dict[str, str]]:
    """Return two ordered system messages for agent and user personas."""
    return [
        {"role": "system", "content": agent_system},
        {"role": "system", "content": user_system},
    ]


def build_turn_messages(
    *,
    agent_system: str,
    user_system: str,
    prior_messages: List[Dict[str, str]],
    probe_prompt: str,
) -> List[Dict[str, str]]:
    """Compose full messages for a turn using the two system prompts and current probe.

    This is a thin wrapper delegating construction to `build_messages` to keep
    template concerns centralized and consistent with other benchmarks.
    """
    return _build_messages(
        agent_system_prompt=agent_system,
        user_system_prompt=user_system,
        prior_messages=prior_messages,
        probe_prompt=probe_prompt,
    )


def _build_messages(
    *,
    agent_system_prompt: str,
    user_system_prompt: str,
    prior_messages: List[Dict[str, str]],
    probe_prompt: str,
) -> List[Dict[str, str]]:
    """Construct chat messages with two system prompts and a probe for the current turn.

    - Two system messages first (agent, then user).
    - Append prior dialog exactly as provided.
    - Append current probe as a user message.
    """
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": agent_system_prompt})
    messages.append({"role": "system", "content": user_system_prompt})
    for m in prior_messages:
        role = m.get("role")
        content = m.get("content", "")
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": probe_prompt})
    return messages


