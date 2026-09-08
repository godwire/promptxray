"""Split a prompt template into numbered blocks that can be ablated.

A "block" is one instruction, one few-shot example, or one formatting rule.
Blocks are separated by a blank line. A block is *pinned* (never removed)
when it contains the {input} placeholder or is marked with [[keep]],
because removing it would break the prompt instead of testing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEEP_MARKER = "[[keep]]"
INPUT_PLACEHOLDER = "{input}"


@dataclass
class Block:
    """One removable piece of the prompt."""

    index: int
    text: str
    pinned: bool
    pin_reason: str = ""

    @property
    def preview(self) -> str:
        """Short single-line version, for tables."""
        flat = " ".join(self.text.split())
        return flat if len(flat) <= 90 else flat[:87] + "..."


def parse_blocks(prompt: str) -> list[Block]:
    """Turn a prompt template into a list of Block objects."""
    chunks = [c.strip() for c in re.split(r"\n\s*\n", prompt.strip())]
    chunks = [c for c in chunks if c]

    blocks: list[Block] = []
    for i, chunk in enumerate(chunks):
        pinned = False
        reason = ""
        if KEEP_MARKER in chunk:
            pinned = True
            reason = "marked [[keep]]"
            chunk = chunk.replace(KEEP_MARKER, "").strip()
        if INPUT_PLACEHOLDER in chunk:
            pinned = True
            reason = "contains {input}"
        blocks.append(Block(index=i, text=chunk, pinned=pinned, pin_reason=reason))
    return blocks


def render(blocks: list[Block], skip: int | None = None) -> str:
    """Rebuild the prompt text, optionally leaving one block out."""
    parts = [b.text for b in blocks if b.index != skip]
    return "\n\n".join(parts)


def validate(blocks: list[Block]) -> None:
    """Fail early on prompts this tool cannot work with."""
    if not blocks:
        raise ValueError("The prompt file is empty.")
    joined = "\n".join(b.text for b in blocks)
    if INPUT_PLACEHOLDER not in joined:
        raise ValueError(
            "The prompt must contain the {input} placeholder - that is where "
            "each row of your dataset gets inserted."
        )
