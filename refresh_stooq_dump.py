from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def resolve_path(p: str) -> Path:
    """
    Supports:
      - ${workspaceFolder} -> current working directory (Cursor/VS Code style)
      - ~ home expansion
      - environment variables like $HOME
      - relative paths resolved from current working directory
    """
    p = (p or "").strip()
    if "${workspaceFolder}" in p:
        p = p.replace("${workspaceFolder}", str(Path.cwd()))
    p = os.path.expandvars(os.path.expanduser(p))
    return Path(p).resolve()


def looks_like_data_dir(p: Path) -> bool:
    """True if p looks like the extracted stooq 'data' directory."""
    return p.is_dir() and (p / "daily").is_dir()


def find_data_dir(root: Path) -> Path:
    """
    Accepts:
      - root is already the data dir (root/daily exists)
      - root contains data/ (root/data/daily exists)
      - root contains nested .../data/daily somewhere
    Returns the actual 'data' directory.
    """
    if looks_like_data_dir(root):
        return root

    candidate = root / "data"
    if looks_like_data_dir(candidate):
        return candidate

    for p in root.rglob("data"):
        if looks_like_data_dir(p):
            return p

    raise FileNotFoundError(f"Could not find a 'data' folder containing 'daily/' under: {root}")


def newest_matching(downloads: Path, pattern: str, min_size_mb: int) -> Path:
    """
    Pick newest item in downloads matching pattern:
      - accept folders (no size check)
      - accept files only if >= min_size_mb
    """
    candidates = sorted(downloads.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.is_dir():
            return p
        if p.is_file() and p.stat().st_size >= min_size_mb * 1024 * 1024:
            return p
    raise FileNotFoundError(
        f"No match for '{pattern}' (folder or file >= {min_size_mb}MB) found in {downloads}"
    )


def unpack_zip_to_temp(zip_path: Path, temp_dir: Path) -> Path:
    """
    Unpack a ZIP into temp_dir. If file has no .zip suffix, still try zip format.
    Returns the directory where contents were extracted (temp_dir).
    """
    try:
        if zip_path.suffix.lower() == ".zip":
            shutil.unpack_archive(str(zip_path), str(temp_dir))
        else:
            shutil.unpack_archive(str(zip_path), str(temp_dir), format="zip")
    except Exception as e:
        raise RuntimeError(f"Failed to unpack {zip_path} as zip: {e}") from e
    return temp_dir


def atomic_replace(dest_dir: Path, staged_dir: Path) -> None:
    """
    Replace dest_dir with staged_dir safely:
      - rename existing dest to backup
      - rename staged into place
      - delete backup
    """
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    backup = dest_dir.with_name(dest_dir.name + "_backup")

    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)

    if dest_dir.exists():
        dest_dir.rename(backup)

    try:
        staged_dir.rename(dest_dir)
    except Exception:
        # rollback
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        if backup.exists():
            backup.rename(dest_dir)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def stage_copy_to_dest_parent(src_data_dir: Path, dest_dir: Path) -> Path:
    """
    Copy src_data_dir into dest_dir's parent as dest_new so we can rename atomically.
    """
    staged = dest_dir.with_name(dest_dir.name + "_new")
    if staged.exists():
        shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(src_data_dir, staged)
    return staged


def main() -> None:
    ap = argparse.ArgumentParser()

    # Backward-compatible: --zip OR --src (both mean "source")
    ap.add_argument(
        "--src", "--zip",
        dest="src",
        default="",
        help='Source Stooq download (FOLDER like "/Users/v/Downloads/data" OR ZIP file)'
    )

    ap.add_argument("--downloads", default="~/Downloads", help="Folder to search if --src/--zip is not provided")
    ap.add_argument("--pattern", default="data*", help="Glob pattern in downloads if --src/--zip not provided")
    ap.add_argument("--min_size_mb", type=int, default=10, help="Minimum size for archive files (default: 10MB)")
    ap.add_argument("--dest", default="${workspaceFolder}/data", help="Destination data folder to replace (default: ./data)")
    ap.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help=(
            "copy = keep the source intact (default). "
            "move = move folder into project (removes it from Downloads; "
            "if source is a zip, it will be deleted after a successful update)."
        ),
    )

    args = ap.parse_args()

    dest_dir = resolve_path(args.dest)

    # Choose source path
    if args.src:
        src_path = resolve_path(args.src)
        if not src_path.exists():
            raise FileNotFoundError(src_path)
    else:
        downloads = resolve_path(args.downloads)
        src_path = newest_matching(downloads, args.pattern, args.min_size_mb)

    print(f"Source selected: {src_path}")

    # Determine the actual 'data' dir
    if src_path.is_dir():
        src_data_dir = find_data_dir(src_path)

        if args.mode == "move":
            # Fast: move folder into place (removes it from Downloads)
            atomic_replace(dest_dir, src_data_dir)

            # If a parent folder was provided, clean it up if now empty.
            if src_data_dir != src_path and src_path.exists():
                try:
                    if not any(src_path.iterdir()):
                        src_path.rmdir()
                except OSError:
                    pass
        else:
            # Safe: copy into project then swap
            staged = stage_copy_to_dest_parent(src_data_dir, dest_dir)
            atomic_replace(dest_dir, staged)

        print(f"✅ Replaced project data folder at: {dest_dir}")
        return

    # Source is a file: treat as zip and extract to temp, then copy into project and swap
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        extracted_root = unpack_zip_to_temp(src_path, td_path)
        src_data_dir = find_data_dir(extracted_root)

        staged = stage_copy_to_dest_parent(src_data_dir, dest_dir)
        atomic_replace(dest_dir, staged)

    if args.mode == "move":
        try:
            src_path.unlink()
            print(f"🧹 Deleted source archive: {src_path}")
        except OSError:
            print(f"⚠️  Could not delete source archive: {src_path}")

    print(f"✅ Replaced project data folder at: {dest_dir}")


if __name__ == "__main__":
    main()
