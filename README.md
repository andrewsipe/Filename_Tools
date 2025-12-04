# Filename Tools

Font filename normalization and cleaning tools for consistent naming conventions.

## Overview

This directory contains specialized scripts for cleaning, normalizing, and standardizing font filenames. These tools help ensure consistent naming patterns across font collections.

## Scripts

### `FontFiles_Cleaner.py`
**Main unified filename cleaner** - Intelligent pattern matching with auto-adaptive processing.

Cleans font filenames using pattern matching for heavy lifting and minimal dictionaries for edge cases. Auto-detects file quality and adapts processing intensity.

**Usage:**
```bash
# Preview changes
python FontFiles_Cleaner.py /path/to/fonts -R --dry-run

# Light cleaning workflow
python FontFiles_Cleaner.py /path/to/fonts -R --workflow light

# Heavy cleaning workflow
python FontFiles_Cleaner.py /path/to/fonts -R --workflow heavy

# Preview mode (shows what would change)
python FontFiles_Cleaner.py /path/to/fonts -R --preview
```

### `FNT_Normalizer.py`
Normalize specific terms in font filenames using case-insensitive find-and-replace.

**Usage:**
```bash
python FNT_Normalizer.py /path/to/fonts -R
```

### `FNT_NumberWordConverter.py`
Convert spelled-out number words (One, Two, Three) to numeric form (1, 2, 3) in filenames.

**Usage:**
```bash
python FNT_NumberWordConverter.py /path/to/fonts -R
```

### `FNT_CompoundWordNormalizer.py`
Normalize compound words in font filenames (e.g., "SemiBold" → "Semi Bold").

### `FNT_AbbreviationsExpander.py`
Expand abbreviations in filenames (e.g., "Reg" → "Regular").

### `FNT_StyleNameExpander.py`
Expand style name abbreviations and normalize style terms.

### `FNT_CapitalizeAfterHyphen.py`
Capitalize words after hyphens in filenames.

### `FNT_HyphenationConsistency.py`
Ensure consistent hyphenation patterns in filenames.

### `FNT_WordDeduplicator.py`
Remove duplicate words from font filenames.

### `FNT_RegularInserter.py`
Insert "Regular" into filenames that are missing it (for non-bold, non-italic fonts).

### `FNT_ReorderWidths.py`
Reorder width terms in filenames to standard positions.

### `FNT_ReorderOpticalSizes.py`
Reorder optical size terms in filenames.

### `FNT_ReorderFind-n-Replace.py`
Reorder terms in filenames using find-and-replace patterns.

### `FNT_PatternAnalyzer.py`
Analyze filename patterns in a font collection.

### `FNT_CorpusPatternAnalyzer.py`
Analyze patterns across a corpus of font filenames.

## Common Options

Most scripts support:
- `-R, --recursive` - Process directories recursively
- `--dry-run` - Preview changes without modifying files
- `-V, --verbose` - Show detailed processing information

## Dependencies

See `requirements.txt`:
- `word2number` - For number word conversion (FNT_NumberWordConverter.py)
- Core dependencies (fonttools, rich) provided by included `core/` library

## Installation

### Option 1: Install with pipx (Recommended)

pipx installs the tool in an isolated environment and makes it available system-wide:

```bash
# Install directly from GitHub
pipx install git+https://github.com/andrewsipe/Filename_Tools.git

# Or install a specific branch/tag
pipx install git+https://github.com/andrewsipe/Filename_Tools.git@main
```

After installation, run scripts directly:
```bash
python FontFiles_Cleaner.py /path/to/fonts -R --dry-run
```

**Upgrade:**
```bash
pipx upgrade font-filename-tools
```

**Uninstall:**
```bash
pipx uninstall font-filename-tools
```

### Option 2: Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/andrewsipe/Filename_Tools.git
cd Filename_Tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts:
```bash
python FontFiles_Cleaner.py /path/to/fonts -R --dry-run
```

## Workflow Recommendations

1. **Start with analysis**: Use `FNT_PatternAnalyzer.py` to understand your filename patterns
2. **Clean with FontFiles_Cleaner**: Use the unified cleaner for most cases
3. **Apply specific normalizers**: Use individual FNT_ scripts for targeted fixes
4. **Verify with dry-run**: Always use `--dry-run` first to preview changes

## Related Tools

- [FileRenamer](https://github.com/andrewsipe/FileRenamer) - Rename fonts to PostScript names
- [FontNameID](https://github.com/andrewsipe/FontNameID) - Update font metadata (NameID tables)

