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
    "check": ("FNT_Check", "list filenames that need a look (changes nothing)"),
    "cleaner": ("FontFiles_Cleaner", "the usual clean-up in one pass (light to heavy workflows)"),
    "normalize": ("FNT_Normalizer", "make a term's spelling and case consistent"),
    "numbers": ("FNT_NumberWordConverter", "number words to digits (One -> 1)"),
    "compound": ("FNT_CompoundWordNormalizer", "put compound words in the house form (Semi Bold -> Semibold)"),
    "abbreviations": ("FNT_AbbreviationsExpander", "expand abbreviations (Reg -> Regular)"),
    "styles": ("FNT_StyleNameExpander", "expand style-name abbreviations"),
    "capitalize-after-hyphen": ("FNT_CapitalizeAfterHyphen", "capitalize words after hyphens"),
    "hyphenation": ("FNT_HyphenationConsistency", "put hyphens in a consistent place"),
    "dedupe-words": ("FNT_WordDeduplicator", "remove duplicate words"),
    "regular": ("FNT_RegularInserterEnhanced", "insert a missing Regular"),
    "reorder-widths": ("FNT_ReorderWidths", "reorder width terms"),
    "reorder-slopes": ("FNT_ReorderSlopes", "reorder slope terms"),
    "reorder-optical": ("FNT_ReorderOpticalSizes", "reorder optical-size terms"),
    "reorder-replace": ("FNT_ReorderFind-n-Replace", "move words you name to just after the hyphen (-F WORD)"),
    "patterns": ("FNT_PatternAnalyzer", "find the naming patterns in a list of filenames"),
    "corpus": ("FNT_CorpusPatternAnalyzer", "report which words and word pairs the filenames use"),
}


# The help screen groups the targets by job, in the order a run usually goes. Every target is in exactly one group.
GROUPS: List[Tuple[str, List[str]]] = [
    ("Look first (these change nothing)", ["check", "patterns", "corpus"]),
    ("Do the usual clean-up in one go", ["cleaner"]),
    (
        "Fix the words",
        ["normalize", "compound", "abbreviations", "styles", "numbers", "dedupe-words", "capitalize-after-hyphen"],
    ),
    ("Fix hyphens and add Regular", ["hyphenation", "regular"]),
    ("Put the terms in order", ["reorder", "reorder-widths", "reorder-slopes", "reorder-optical", "reorder-replace"]),
]

# `filenametools reorder <family> ...` is the same as `filenametools reorder-<family> ...`; the old names stay.
REORDER_FAMILIES = ("widths", "slopes", "optical", "replace")
REORDER_DESCRIPTION = "move terms into place: reorder widths | slopes | optical | replace"

# Two targets read their arguments by hand and have no -h of their own; the dispatcher answers for them.
OWN_HELP: Dict[str, str] = {
    "patterns": (
        "usage: filenametools patterns [LISTFILE]\n\n"
        "Analyze the naming patterns in a text file that lists one font filename per line. "
        "Asks for the path if you give none. Changes nothing."
    ),
    "corpus": (
        "usage: filenametools corpus PATH [PATH ...] [-r]\n\n"
        "Report the word sequences (n-grams) in the filenames of the fonts found in PATH. "
        "-r includes subfolders. Changes nothing."
    ),
}


def _usage() -> str:
    width = max(len(name) for name in TARGETS)
    lines = ["usage: filenametools <target> [options] PATH ...", ""]
    for title, targets in GROUPS:
        lines.append(f"{title}:")
        lines += [
            f"  {name:<{width}}  {REORDER_DESCRIPTION if name == 'reorder' else TARGETS[name][1]}" for name in targets
        ]
        lines.append("")
    lines += [
        "Run `filenametools <target> -h` for that target's options.",
        "",
        "Suggested order: check, then cleaner (or the single commands you need), then regular,",
        "then the reorder commands. Everything outside the first group renames files on disk:",
        "preview each one with -n first.",
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
    if target == "reorder":
        if not rest or rest[0] not in REORDER_FAMILIES:
            given = f" '{rest[0]}'" if rest else ""
            print(
                f"filenametools reorder: give one of {', '.join(REORDER_FAMILIES)}{given and ' (not' + given + ')'}\n"
                "Example: filenametools reorder widths fonts/ -r -n",
                file=sys.stderr,
            )
            return 2
        target, rest = f"reorder-{rest[0]}", rest[1:]
    if target not in TARGETS:
        print(f"filenametools: unknown target '{target}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    if target in OWN_HELP and any(a in ("-h", "--help") for a in rest):
        print(OWN_HELP[target])
        return 0

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
