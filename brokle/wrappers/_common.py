"""
Shared helper functions for LLM SDK wrappers.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ..types import Attrs

if TYPE_CHECKING:
    from ..prompts import Prompt


def extract_brokle_options(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract brokle_options from kwargs and return clean kwargs.

    Args:
        kwargs: Original keyword arguments

    Returns:
        Tuple of (clean_kwargs without brokle_options, brokle_opts dict)
    """
    brokle_options = kwargs.pop("brokle_options", None)
    return kwargs, brokle_options or {}


def add_prompt_attributes(attrs: Dict[str, Any], brokle_opts: Dict[str, Any]) -> None:
    """
    Add prompt attributes to span attributes if prompt is provided and not a fallback.

    Args:
        attrs: Span attributes dict to modify
        brokle_opts: Brokle options containing optional prompt
    """
    prompt: Optional["Prompt"] = brokle_opts.get("prompt")
    if prompt and not prompt.is_fallback:
        attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
        attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
        if prompt.id and prompt.id != "fallback":
            attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id
