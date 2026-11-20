"""Single entry point: ``filenametools <target> [args]``.

Each target is one of the original single-purpose scripts; the remaining
arguments go to that script unchanged. Run ``filenametools <target> -h``.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from typing import Dict, List, Optional, Tuple

from filenametools import __version__

# target -> (module, one-line description)
TARGETS: Dict[str, Tuple[str, str]] = {
    "cleaner": ("FontFiles_Cleaner", "unified filename cleaner (light/heavy workflows)"),
    "normalize": ("FNT_Normalizer", "case-insensitive term normalization"),
    "numbers": ("FNT_NumberWordConverter", "number words to digits (One -> 1)"),
    "compound": ("FNT_CompoundWordNormalizer", "compound words (SemiBold -> Semi Bold)"),
    "abbreviations": ("FNT_AbbreviationsExpander", "expand abbreviations (Reg -> Regular)"),
    "styles": ("FNT_StyleNameExpander", "expand style-name abbreviations"),
    "capitalize-after-hyphen": ("FNT_CapitalizeAfterHyphen", "capitalize words after hyphens"),
    "hyphenation": ("FNT_HyphenationConsistency", "consistent hyphenation"),
    "dedupe-words": ("FNT_WordDeduplicator", "remove duplicate words"),
    "regular": ("FNT_RegularInserterEnhanced", "insert a missing Regular"),
    "reorder-widths": ("FNT_ReorderWidths", "reorder width terms"),
    "reorder-slopes": ("FNT_ReorderSlopes", "reorder slope terms"),
    "reorder-optical": ("FNT_ReorderOpticalSizes", "reorder optical-size terms"),
    "reorder-replace": ("FNT_ReorderFind-n-Replace", "reorder via find/replace patterns"),
    "patterns": ("FNT_PatternAnalyzer", "sequence/pattern stats for a filename list"),
    "corpus": ("FNT_CorpusPatternAnalyzer", "n-gram / corpus report"),
}


def _usage() -> str:
    width = max(len(name) for name in TARGETS)
    lines = ["usage: filenametools <target> [options] PATH ...", "", "Targets:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in TARGETS.items()]
    lines += [
        "",
        "Run `filenametools <target> -h` for that target's options.",
        "Example: filenametools reorder-widths fonts/ -r -n",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"filenametools {__version__}")
        return 0

    target, rest = argv[0], argv[1:]
    if target not in TARGETS:
        print(f"filenametools: unknown target '{target}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    module = importlib.import_module(f"filenametools.{TARGETS[target][0]}")
    sys.argv = [f"filenametools {target}", *rest]
    try:
        if inspect.signature(module.main).parameters:
            result = module.main(rest)  # scripts written as main(argv)
        else:
            result = module.main()  # scripts that read sys.argv themselves
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
