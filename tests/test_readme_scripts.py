"""Every script in scripts/ must appear in a README.md table row."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {"diag", "__pycache__"}  # scripts/diag is local and not tracked


def table_script_names() -> set[str]:
    """Script file names written in backticks in README.md table rows."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = [line for line in readme.splitlines() if line.startswith("|")]
    names = set()
    for row in rows:
        for ref in re.findall(r"`([^`]+\.py)`", row):
            names.add(Path(ref).name)
    return names


def test_every_script_is_in_a_readme_table():
    scripts = [
        p
        for p in (ROOT / "scripts").rglob("*.py")
        if not EXCLUDED_DIRS & set(p.relative_to(ROOT / "scripts").parts)
    ]
    assert scripts, "no scripts found"
    listed = table_script_names()
    missing = sorted(str(p.relative_to(ROOT)) for p in scripts if p.name not in listed)
    assert not missing, f"scripts missing from the README tables: {missing}"
