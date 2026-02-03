from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from src.utils.config import load_config
from src.downloader.utils import ensure_dir, extract_archive, stage_download


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _resolve_out_dir(output_cfg: Dict[str, Any], key: str) -> Path:
    root = Path(output_cfg.get("root", "./data")).expanduser().resolve()
    subdir = output_cfg.get(key)
    if subdir:
        return (root / subdir).resolve()
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_cfg = cfg.get("output", {})
    sources = cfg.get("sources", [])
    if not sources:
        raise ValueError("downloader config requires a non-empty sources list")

    root = Path(output_cfg.get("root", "./data")).expanduser().resolve()
    cache_dir = ensure_dir(output_cfg.get("cache_dir", root / "_downloads"))
    manifest: Dict[str, Any] = {
        "root": str(root),
        "outputs": {},
        "sources": [],
    }

    for src_cfg in sources:
        name = src_cfg.get("name")
        if not name:
            raise ValueError("Each source entry must include a name")
        url = src_cfg.get("url")
        if not url:
            raise ValueError(f"Source {name} missing url")
        target_key = src_cfg.get("target_key", name)
        out_dir = _resolve_out_dir(output_cfg, f"{target_key}_root")
        ensure_dir(out_dir)

        archive_path = stage_download(
            url,
            cache_dir=cache_dir,
            filename=src_cfg.get("filename"),
            sha256=src_cfg.get("sha256"),
        )

        if src_cfg.get("extract", True):
            extract_archive(
                archive_path,
                out_dir,
                strip_components=int(src_cfg.get("strip_components", 0)),
                members=src_cfg.get("members"),
            )
        else:
            # If not extracting, just copy to destination
            dest = out_dir / Path(archive_path).name
            if dest.resolve() != archive_path.resolve():
                dest.write_bytes(Path(archive_path).read_bytes())

        manifest["sources"].append({
            "name": name,
            "url": url,
            "archive": str(archive_path),
            "out_dir": str(out_dir),
            "target_key": target_key,
            "extract": bool(src_cfg.get("extract", True)),
            "strip_components": int(src_cfg.get("strip_components", 0)),
        })
        manifest["outputs"][target_key] = str(out_dir)

    manifest_path = Path(output_cfg.get("manifest", root / "manifest.json"))
    _write_manifest(manifest_path, manifest)
    print(f"Downloader finished. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
