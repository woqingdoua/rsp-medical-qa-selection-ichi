"""Read-only public-bundle checks, without loading models or benchmark data."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    for r in manifest:
        path = ROOT / r["release_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["release_sha256"], str(path)
    originals = {r["release_path"]: r["original_sha256"] for r in manifest}
    assert originals["src/train_original_rsp_raw.py"] == "4932e49382d32529fb8a7a4c5e9c2400ca0b8fda4c2ce8cafbb2e03a5f904143"
    assert originals["src/common.py"] == "3f9ff788c001b148d35ad0dde3ec4629a17663c30336b3e91420e70cf5605841"
    assert originals["src/original_rsp_raw_utils.py"] == "0338a955f9277ca6460bed0a63beec85b296d10facfd034650840bb0938be973"
    assert hashlib.sha256((ROOT / "paper/ichi-preprint.pdf").read_bytes()).hexdigest() == "82ec3e135dbb17d7e1644988decd6fc01e370163311f808b725918a596b2185d"
    patterns = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----)")
    python_count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in (".git", "__pycache__") for part in path.relative_to(ROOT).parts):
            continue
        assert path.suffix not in (".npz", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".jsonl"), str(path)
        if path.suffix == ".pdf":
            continue
        text = path.read_text(encoding="utf-8")
        assert not patterns.search(text), f"Potential credential in {path.relative_to(ROOT)}"
        if path.suffix == ".py":
            ast.parse(text, filename=str(path.relative_to(ROOT)))
            python_count += 1
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                assert (path.parent / target.split("#")[0]).exists(), f"Broken link: {target} in {path.name}"
    config = json.loads((ROOT / "configs/ichi.json").read_text())
    assert config["scoring"]["K"] == 8 and config["scoring"]["sigma"] == 0.02
    assert config["training"]["rationale_token_loss"] is False
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "analysis/summarize_results.py"], cwd=ROOT, check=True)
    print(json.dumps({"status": "passed", "retained_files_hash_checked": len(manifest),
                      "python_files_syntax_checked": python_count, "model_calls": 0,
                      "training_runs": 0, "checkpoint_evaluations": 0}, indent=2))


if __name__ == "__main__":
    main()
