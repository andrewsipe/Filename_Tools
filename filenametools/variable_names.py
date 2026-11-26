"""Variable-font filenames are set by `fontrenamer` from the font's tables (FontCore DECISIONS.md, Area 5).

The filename commands work on words in a static style (a width moves behind the hyphen, a slope gets reordered), which
would undo the variable form `ProbeDisplayCondensed-Variable3AxisItalic`. So every rewriting command leaves a file
with a variable marker alone, and `check` reports the ones that are not in the house form.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# "Variable" anywhere (glued forms like VariableItalic or Variable3Axis count), or VF / GX / Var as a style token.
_MARKER = re.compile(r"(?i)variable|(?<![a-z])(?:vf|gx)(?![a-z])|(?<=-)var(?![a-z])")
_FONT_SUFFIXES = {".ttf", ".otf", ".woff", ".woff2", ".ttc", ".otc"}


def _stem(name: str) -> str:
    p = Path(name)
    return p.stem if p.suffix.lower() in _FONT_SUFFIXES else p.name


def is_variable_filename(name: str) -> bool:
    """True when the filename carries a variable-font marker."""
    return bool(_MARKER.search(_stem(name)))


def house_form(name: str) -> Optional[str]:
    """The house-form stem for a variable filename, or None when it cannot be worked out from the name alone."""
    from FontCore.core_variable_filename_parser import format_variable_filename, parse_variable_filename

    slots = parse_variable_filename(_stem(name), strip_extension=False)
    if slots is None or not slots.is_valid:
        return None
    return format_variable_filename(slots)
