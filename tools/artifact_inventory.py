"""Create and verify a portable file-size/SHA-256 transfer inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _files(root: Path, excluded: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Links are not accepted in artifact roots: {path}")
        if path.is_file() and path.resolve() != excluded.resolve():
            files.append(path)
    return files


def _identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def create(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Artifact root must be a directory.")
    entries = [{"path": path.relative_to(root).as_posix(), **_identity(path)}
               for path in _files(root, output)]
    inventory = {"schema_version": 1, "files": entries,
                 "total_bytes": sum(entry["bytes"] for entry in entries)}
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def verify(root: Path, inventory_path: Path, *, strict: bool = False) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Artifact root must be a directory.")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (not isinstance(inventory, dict) or inventory.get("schema_version") != 1
            or not isinstance(inventory.get("files"), list)):
        raise ValueError("Unsupported or malformed inventory.")
    entries = inventory["files"]
    seen: set[str] = set()
    errors: list[str] = []
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Inventory entries must be objects.")
        name = entry["path"]
        if not isinstance(name, str):
            raise ValueError("Inventory paths must be strings.")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or "\\" in name or ":" in name or relative.as_posix() != name):
            raise ValueError(f"Unsafe/noncanonical inventory path: {name}")
        if name in seen:
            raise ValueError(f"Duplicate inventory path: {name}")
        seen.add(name)
        size = entry["bytes"]
        digest = entry["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Invalid byte count: {name}")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA-256: {name}")
        total += size
        path = root.joinpath(*relative.parts)
        for part in [path, *path.parents]:
            if part == root:
                break
            if part.is_symlink() or part.is_junction():
                raise ValueError(f"Linked inventory path: {name}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Path escapes artifact root: {name}")
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif _identity(path) != {"bytes": size, "sha256": digest}:
            errors.append(f"size/hash mismatch: {name}")
    if inventory.get("total_bytes") != total:
        raise ValueError("Inventory total_bytes does not match its entries.")
    if strict:
        actual = {path.relative_to(root).as_posix() for path in _files(root, inventory_path)}
        errors.extend(f"unexpected: {name}" for name in sorted(actual - seen))
    return {"verified": not errors, "files": len(seen), "total_bytes": total, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("root", type=Path)
    make.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("root", type=Path)
    check.add_argument("inventory", type=Path)
    check.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create(args.root, args.output)
        else:
            result = verify(args.root, args.inventory, strict=args.strict)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    if args.command == "verify" and not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
