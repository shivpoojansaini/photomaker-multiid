# prompt_splitter.py
# Simple prompt parser to split "left person as X, right person as Y" into two prompts

import re
from typing import Tuple, Optional


def split_prompt(prompt: str, trigger_word: str = "img") -> Tuple[str, str]:
    """
    Split a combined prompt into left and right prompts.

    Input: "left person as teacher, right person as doctor"
    Output: ("a man img as teacher", "a man img as doctor")

    Supports patterns:
    - "left person as X, right person as Y"
    - "person on the left as X, person on the right as Y"
    - Comma-separated with position hints

    Args:
        prompt: Combined natural language prompt
        trigger_word: PhotoMaker trigger word (default: "img")

    Returns:
        Tuple of (left_prompt, right_prompt)
    """
    prompt = prompt.strip()

    # Pattern 1: "left person as X, right person as Y"
    pattern1 = re.compile(
        r'left\s+(\w+)\s+(?:as|is|wearing|with|change|in)\s+([^,]+)'
        r'[,;]\s*'
        r'right\s+(\w+)\s+(?:as|is|wearing|with|change|in)\s+(.+)',
        re.IGNORECASE
    )

    match = pattern1.search(prompt)
    if match:
        left_person, left_desc, right_person, right_desc = match.groups()
        left_prompt = f"a {left_person} {trigger_word} {left_desc.strip()}"
        right_prompt = f"a {right_person} {trigger_word} {right_desc.strip()}"
        return left_prompt, right_prompt

    # Pattern 2: "person on the left as X, person on the right as Y"
    pattern2 = re.compile(
        r'(\w+)\s+(?:on\s+the\s+)?left\s+(?:as|is|wearing|with|change|in)\s+([^,]+)'
        r'[,;]\s*'
        r'(\w+)\s+(?:on\s+the\s+)?right\s+(?:as|is|wearing|with|change|in)\s+(.+)',
        re.IGNORECASE
    )

    match = pattern2.search(prompt)
    if match:
        left_person, left_desc, right_person, right_desc = match.groups()
        left_prompt = f"a {left_person} {trigger_word} {left_desc.strip()}"
        right_prompt = f"a {right_person} {trigger_word} {right_desc.strip()}"
        return left_prompt, right_prompt

    # Pattern 3: Simple comma split fallback
    parts = re.split(r'[,;]\s*', prompt, maxsplit=1)
    if len(parts) == 2:
        left_part, right_part = parts

        # Clean and add trigger word if missing
        left_prompt = _ensure_trigger_word(left_part.strip(), trigger_word)
        right_prompt = _ensure_trigger_word(right_part.strip(), trigger_word)

        return left_prompt, right_prompt

    # Single prompt - use for both (fallback)
    single = _ensure_trigger_word(prompt, trigger_word)
    return single, single


def _ensure_trigger_word(text: str, trigger_word: str) -> str:
    """Ensure trigger word is present in the prompt."""
    if trigger_word in text:
        return text

    # Try to insert after person/man/woman keywords
    person_words = ['person', 'man', 'woman', 'guy', 'girl']
    for word in person_words:
        pattern = re.compile(rf'\b({word})\b', re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(rf'\1 {trigger_word}', text, count=1)

    # Prepend "a person img" if no person word found
    return f"a person {trigger_word} {text}"


def extract_positions_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract position hints from prompt for face ordering.

    Returns:
        Tuple of (left_position_hint, right_position_hint)
    """
    # Check for explicit position references
    has_left = bool(re.search(r'\bleft\b', prompt, re.IGNORECASE))
    has_right = bool(re.search(r'\bright\b', prompt, re.IGNORECASE))

    if has_left and has_right:
        return "left", "right"

    # Default: first mentioned is left, second is right
    return "left", "right"
