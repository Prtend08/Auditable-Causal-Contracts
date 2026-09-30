"""Build the 2026-09-30 code-only release from explicit source allowlists.

Maintainer utility for the full working tree. Never recursively copies the
repository and never deletes a directory. Existing authored release documents
are inputs. Generated code copies, manifests, and the owned ZIP are refreshed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = "Ask-or-Adapt_Code_20260930"
DEST = ROOT / "release" / NAME
ZIP = DEST.with_suffix(".zip")
ALLOWED_EXTENSIONS = {".py", ".sh", ".yaml", ".md", ".txt", ".sha256", ".json"}
FORBIDDEN_PARTS = {"__pycache__", ".git", ".ssh", ".pytest_cache", "datasets", "weights",
                   "results", "remote_artifacts", "analysis", "paper", "synthetic", "node_modules"}
DOCS = {"README.md", "REPRODUCIBILITY.md", "EVENT_SCHEMA.md",
        "environment/requirements-replay.txt", "environment/requirements-analysis.txt",
        "environment/requirements-cuda-reference.txt", "environment/requirements-budget-leakage.txt",
        "config/review_round2.yaml"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_text(path):
    content = path.read_text(encoding="utf-8-sig")
    # Fail closed on credential-like literals. No matched secret values are
    # printed. This is a scoped pattern scan, not a guarantee against every
    # possible secret encoding.
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        "github_token": r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}",
        "huggingface_token": r"\bhf_[A-Za-z0-9]{20,}",
        "openai_style_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}",
        "aws_access_key": r"\bAKIA[A-Z0-9]{16}\b",
        "credential_assignment": r"(?i)\b(?:password|passwd|api_key|access_token|secret_key)\s*[:=]\s*['\"][^'\"\n]{8,}['\"]",
        "credential_url": r"https?://[^\s/:]+:[^\s/@]+@",
        "personal_workspace_path": r"(?i)(?:C:[/\\]Users[/\\]|/ho" + r"me/)[^\s'\"]+",
    }
    issues = [name for name, pattern in patterns.items() if re.search(pattern, content)]
    if issues:
        raise RuntimeError(f"Release scan issue in {path.relative_to(DEST)}: {issues}")
    return content


def main():
    if DEST.resolve().parent != (ROOT / "release").resolve():
        raise RuntimeError("Invalid release destination")
    missing = [name for name in DOCS if not (DEST / name).is_file()]
    if missing:
        raise RuntimeError(f"Required authored release documents missing: {missing}")
    sources = []
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("scripts", "*.sh"), ("protocol", "*.yaml")):
        for path in sorted((ROOT / folder).rglob(pattern)):
            if not any(part in FORBIDDEN_PARTS for part in path.relative_to(ROOT).parts):
                sources.append((path, path.relative_to(ROOT)))
    # One explicitly reviewed analysis utility lives beside the manuscript in
    # the working tree. Copy only this Python source into src; paper artifacts,
    # generated analyses, and result files remain excluded.
    sources.append((ROOT / "paper/analyze_paired_anchor_ci.py", Path("src/analyze_paired_anchor_ci.py")))
    sources.append((ROOT / "paper/audit_budget_leakage_replay.py", Path("src/audit_budget_leakage_replay.py")))
    sources.append((ROOT / "release/Ask-or-Adapt_Code_20260814/environment/requirements-autodl-lock.txt",
                    Path("environment/requirements-historical-autodl-lock.txt")))
    sources.append((ROOT / "release/Ask-or-Adapt_Code_20260814/environment/requirements-core.txt",
                    Path("environment/requirements-historical-core.txt")))
    for source, relative in sources:
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Copy hash mismatch: {relative}")
    expected = DOCS | {str(relative).replace("\\", "/") for _, relative in sources}
    expected |= {"CODE_MANIFEST.sha256", "RELEASE_VALIDATION.json"}
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    unexpected = [str(p.relative_to(DEST)) for p in files if p.relative_to(DEST).as_posix() not in expected]
    if unexpected:
        raise RuntimeError(f"Unexpected files in release (not removed automatically): {unexpected}")
    py_count = 0
    for path in files:
        rel = path.relative_to(DEST)
        if path.suffix not in ALLOWED_EXTENSIONS or any(part in FORBIDDEN_PARTS for part in rel.parts):
            raise RuntimeError(f"Forbidden release file: {rel}")
        content = scan_text(path)
        if path.suffix == ".py":
            ast.parse(content, filename=str(rel))
            py_count += 1
    test_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    tests = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           cwd=DEST, env=test_environment, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(f"Packaged unit tests failed: {tests.stdout}\n{tests.stderr}")
    test_count = re.search(r"(\d+) passed", tests.stdout)
    if not test_count:
        raise RuntimeError("Packaged test success count missing")
    validation = {
        "release_name": NAME,
        "source_copy_files": len(sources),
        "python_files_syntax_checked": py_count,
        "source_copies_sha256_match": True,
        "allowlisted_text_files_only": True,
        "excluded": ["datasets", "weights", "NPZ", "CSV", "event logs", "results", "images", "PDF", "DOCX", "nested archives", "bytecode", "credentials"],
        "credential_scan": "No matches for private-key, common access-token, credential-literal/URL, or personal-workspace-path patterns. Pattern scan is not a universal secret-detection guarantee.",
        "public_upload_status": "Prepared locally; no public URL assigned by packaging.",
        "unit_tests": {"passed": int(test_count.group(1)), "exit_code": tests.returncode,
                       "python": sys.version.split()[0], "scope": "Packaged synthetic core tests; no benchmark data."},
        "test_scope": "This artifact does not claim a clean-install or end-to-end dataset reproduction.",
    }
    (DEST / "RELEASE_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    hashed_files = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "CODE_MANIFEST.sha256")
    manifest = "".join(f"{digest(path)}  {path.relative_to(DEST).as_posix()}\n" for path in hashed_files)
    (DEST / "CODE_MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    temporary_zip = ZIP.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            rel = path.relative_to(DEST)
            info = zipfile.ZipInfo(f"{NAME}/{rel.as_posix()}", date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix == ".sh" else 0o644) << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(temporary_zip) as archive:
        if archive.testzip() is not None or len(archive.infolist()) != len(files):
            raise RuntimeError("ZIP integrity or count failure")
        for path in files:
            member = f"{NAME}/{path.relative_to(DEST).as_posix()}"
            if hashlib.sha256(archive.read(member)).hexdigest() != digest(path):
                raise RuntimeError(f"ZIP content mismatch: {member}")
    temporary_zip.replace(ZIP)
    print(json.dumps({"directory": str(DEST), "zip": str(ZIP), "files": len(files),
                      "python_files": py_count, "bytes_uncompressed": sum(p.stat().st_size for p in files),
                      "zip_bytes": ZIP.stat().st_size, "zip_sha256": digest(ZIP),
                      "credential_pattern_matches": 0, "forbidden_files": 0,
                      "zip_integrity": "PASS", "all_source_hashes_match": True}, indent=2))


if __name__ == "__main__":
    main()
