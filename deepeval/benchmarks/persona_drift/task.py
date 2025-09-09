from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .template import build_turn_messages


class PersonaSpec(BaseModel):
    id: str
    category: str
    system_prompt: str
    probe_prompt: str
    judge_key: str


def _judge_starts_with(text: str, prefix: str) -> float:
    return 1.0 if text.strip().lower().startswith(prefix.lower()) else 0.0


def _judge_all_caps(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    upper = sum(ch.isupper() for ch in letters)
    return upper / len(letters)


def _judge_all_lower(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return 1.0 if all(ch.islower() for ch in letters) else 0.0


def _judge_contains_color_and_number(text: str) -> float:
    colors = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray"}
    tl = text.lower()
    has_color = any(c in tl for c in colors)
    import re

    has_number = bool(re.search(r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", tl))
    return 1.0 if (has_color and has_number) else 0.0


def _judge_likely_french(text: str) -> float:
    lex = {"bonjour", "merci", "oui", "non", "et", "le", "la", "les", "des", "je", "vous"}
    import re

    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in lex)
    return hits / len(tokens)


_JUDGES = {
    "starts_with_a": lambda t: _judge_starts_with(t, "a"),
    "starts_with_z": lambda t: _judge_starts_with(t, "z"),
    "all_caps": _judge_all_caps,
    "all_lower": _judge_all_lower,
    "contains_color_and_number": _judge_contains_color_and_number,
    "likely_french": _judge_likely_french,
}


PERSONAS = [
    PersonaSpec(
        id="pattern_start_a",
        category="pattern",
        system_prompt="Always start your reply with the letter A.",
        probe_prompt="What do you do in London as a tourist?",
        judge_key="starts_with_a",
    ),
    PersonaSpec(
        id="pattern_start_z",
        category="pattern",
        system_prompt="Always start your reply with the letter Z.",
        probe_prompt="What do you do in London as a tourist?",
        judge_key="starts_with_z",
    ),
    PersonaSpec(
        id="pattern_all_caps",
        category="pattern",
        system_prompt="WRITE YOUR RESPONSE ENTIRELY IN CAPITAL LETTERS.",
        probe_prompt="What do you do in London as a tourist?",
        judge_key="all_caps",
    ),
    PersonaSpec(
        id="pattern_all_lower",
        category="pattern",
        system_prompt="write your response entirely in lowercase letters.",
        probe_prompt="What do you do in London as a tourist?",
        judge_key="all_lower",
    ),
    PersonaSpec(
        id="memorization_color_number",
        category="memorization",
        system_prompt="Every response must contain a color and a number.",
        probe_prompt="What do you do in London as a tourist?",
        judge_key="contains_color_and_number",
    ),
    PersonaSpec(
        id="language_french",
        category="language",
        system_prompt="Vous parlez toujours en français, même si l'utilisateur parle anglais.",
        probe_prompt="What do you do in London as a tourist?",
        judge_key="likely_french",
    ),
]


class PersonaDriftTask(BaseModel):
    """Encapsulates persona lookup and message construction for a conversation."""

    agent_persona_id: str
    user_persona_id: str
    topic: Optional[str] = Field(default=None)

    def _find_persona(self, pid: str) -> PersonaSpec:
        for p in PERSONAS:
            if p.id == pid:
                return p
        raise ValueError(f"Unknown persona id: {pid}")

    @property
    def agent_persona(self) -> PersonaSpec:
        return self._find_persona(self.agent_persona_id)

    @property
    def user_persona(self) -> PersonaSpec:
        return self._find_persona(self.user_persona_id)

    def build_turn(self, prior_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Build messages for the current turn by injecting the probe as user message."""
        return build_turn_messages(
            agent_system=self.agent_persona.system_prompt,
            user_system=self.user_persona.system_prompt,
            prior_messages=prior_messages,
            probe_prompt=self.agent_persona.probe_prompt,
        )

    def judge(self, reply: str) -> float:
        # Combine agent and user persona judges if both are present by taking the minimum.
        agent_judge = _JUDGES[self.agent_persona.judge_key]
        agent_score = float(max(0.0, min(1.0, agent_judge(reply))))
        try:
            user_judge = _JUDGES[self.user_persona.judge_key]
            user_score = float(max(0.0, min(1.0, user_judge(reply))))
            return min(agent_score, user_score)
        except KeyError:
            # If user persona has no judge (shouldn't happen in our catalog), fall back to agent only
            return agent_score


