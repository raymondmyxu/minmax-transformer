"""Safety checks for the code-only sharing archive."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from package_code_only import ARCHIVE_ROOT, build_code_archive

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_code_archive_contains_sources_and_excludes_generated_files(tmp_path: Path) -> None:
    output = build_code_archive(PROJECT_ROOT, tmp_path / "code-only.zip")

    with ZipFile(output) as archive:
        names = set(archive.namelist())
        readme = archive.read(f"{ARCHIVE_ROOT}/README.md").decode("utf-8")

    assert f"{ARCHIVE_ROOT}/README.md" in names
    assert f"{ARCHIVE_ROOT}/pyproject.toml" in names
    assert f"{ARCHIVE_ROOT}/minmax_transformer/model.py" in names
    assert f"{ARCHIVE_ROOT}/train_single_head_sweep.py" in names
    assert f"{ARCHIVE_ROOT}/package_code_only.py" in names
    assert f"{ARCHIVE_ROOT}/ARCHIVE_MANIFEST.txt" in names
    assert "python3.12 -m venv .venv" in readme

    forbidden_fragments = (
        "/artifacts/",
        "/.venv/",
        "/.idea/",
        "/.git/",
        "/__pycache__/",
        "/.pytest_cache/",
        "/.ruff_cache/",
        ".pt",
        ".pth",
        ".csv",
        ".png",
        ".jpg",
        ".zip",
    )
    assert not any(
        fragment in name.lower() for name in names for fragment in forbidden_fragments
    )
