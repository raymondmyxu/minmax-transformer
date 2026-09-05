"""Create a shareable source archive without generated experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ARCHIVE_ROOT = "minmax-transformer-code-only"
DEFAULT_OUTPUT = Path("dist/minmax-transformer-code-only.zip")
ROOT_DOCUMENTS = (Path(".gitignore"), Path("README.md"), Path("pyproject.toml"))
SOURCE_DIRECTORIES = (Path("minmax_transformer"), Path("tests"))


def collect_shareable_files(project_root: str | Path) -> tuple[Path, ...]:
    """Return an explicit source/documentation allowlist relative to the project root."""

    root = Path(project_root).resolve()
    relative_files = set(ROOT_DOCUMENTS)
    relative_files.update(path.relative_to(root) for path in root.glob("*.py") if path.is_file())
    for directory in SOURCE_DIRECTORIES:
        source_root = root / directory
        relative_files.update(
            path.relative_to(root) for path in source_root.rglob("*.py") if path.is_file()
        )

    missing = [path for path in relative_files if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"shareable source file is missing: {missing[0]}")
    return tuple(sorted(relative_files, key=lambda path: path.as_posix()))


def build_code_archive(
    project_root: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Write a deterministic-layout code-only ZIP and return its absolute path."""

    root = Path(project_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = collect_shareable_files(root)
    manifest_lines = [
        "Fixed-Query Min/Max Transformer — code-only archive",
        "",
        "Excluded by construction:",
        "  artifacts, checkpoints, CSV data, generated figures, virtual environments,",
        "  IDE metadata, caches, coverage output, egg-info, and version-control metadata.",
        "",
        "Included files:",
        *[f"  {path.as_posix()}" for path in files],
        "",
    ]

    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_path in files:
            archive.write(root / relative_path, f"{ARCHIVE_ROOT}/{relative_path.as_posix()}")
        archive.writestr(
            f"{ARCHIVE_ROOT}/ARCHIVE_MANIFEST.txt",
            "\n".join(manifest_lines),
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package source, tests, configuration, and README without experiment outputs."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    archive_path = build_code_archive(project_root, args.output)
    print(f"saved code-only archive: {archive_path}")


if __name__ == "__main__":
    main()
