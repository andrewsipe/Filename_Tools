# Vendored FontCore subset

filenametools ships a minimal copy of [FontCore](https://github.com/andrewsipe/FontCore)
so it installs without the full library, a submodule, or a symlink.

**Source:** FontCore 0.7.0 (`v0.7.0`)

**Do not edit these files here.** Fix them in FontCore, then re-sync.

## Declared modules

- `core_console_styles.py`
- `core_file_collector.py`
- `core_font_style_dictionaries.py`

## Pulled in as imports of the above

- `core_console_config.py`
- `core_logging_config.py`

## Refresh

From a FontCore checkout:

```bash
python tools/vendor_fontcore.py sync ../filenametools
```

The module list lives in `[tool.fontcore-vendor]` in this project's `pyproject.toml`.
