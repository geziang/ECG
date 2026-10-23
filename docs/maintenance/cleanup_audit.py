"""Read-only audit for the 2026-10-23 workspace cleanup (stdlib only).

plan records exact file candidates; verify checks preservation. Deletion is a
separate, manifest-driven PowerShell step. Never traverses Git objects or links.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "ECG"
OUT = Path(__file__).resolve().parent / "cleanup_2026-10-23"
MANIFEST = OUT / "manifest.json.gz"
EDITABLE = {
    "WORKSPACE_INDEX.md", "ECG/PROJECT_MAP.md", "ECG/.gitignore",
    "ECG/docs/maintenance/cleanup_audit.py",
}
RENDER_DIRS = {
    "qa_source", "qa_sci_v1", "qa_sci_v2",
    ".docx_fix/render1", ".docx_fix/render2", ".docx_fix/render3",
    "outputs/paper_revision_20261023/render",
    "outputs/paper_revision_20261023/render_checked",
    "outputs/paper_revision_20261023/render_final",
}
# Only snapshots predating the cleanup day; reports and repair scripts stay.
SNAPSHOT_CUTOFF = 1792684800  # 2026-10-23 00:00:00 +08:00


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-c", "core.quotePath=false", *args], cwd=REPO)


def git_state() -> dict:
    objects = sorted(git("cat-file", "--batch-all-objects", "--batch-check=%(objectname)").splitlines())
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "index_sha256": sha(REPO / ".git/index"),
        "refs_sha256": hashlib.sha256(git("show-ref")).hexdigest(),
        "reflog_sha256": hashlib.sha256(git("reflog", "show", "--all", "--format=%H %gs")).hexdigest(),
        "object_count": len(objects),
        "object_ids_sha256": hashlib.sha256(b"\n".join(objects)).hexdigest(),
        "status": git("status", "--porcelain=v1", "-uall").decode("utf-8").splitlines(),
    }


def inventory() -> dict:
    files = {}
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d != ".git" and not (Path(base) / d).is_junction() and not (Path(base) / d).is_symlink()]
        for name in names:
            path = Path(base) / name
            if path.is_symlink() or path.is_junction():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if path.is_relative_to(OUT):
                continue
            stat = path.stat()
            files[rel] = {"bytes": stat.st_size, "sha256": sha(path), "mtime": stat.st_mtime}
    return files


def plan() -> None:
    if MANIFEST.exists():
        raise RuntimeError("Preserve the existing baseline; use verify for subsequent checks.")
    before = inventory()
    tracked = {"ECG/" + p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p}
    candidates = {}
    for rel, record in before.items():
        path = Path(rel)
        parts = path.parts
        category = None
        if "__pycache__" in parts:
            category = "python_bytecode"
        elif any(parts[i:i+2] == (".mimosa", "hook-state") for i in range(len(parts)-1)) and record["mtime"] < SNAPSHOT_CUTOFF:
            category = "old_tool_snapshots"
        elif path.parent.as_posix() in RENDER_DIRS and re.fullmatch(r"(?:page-\d+\.png|contact-[\d-]+\.jpg)", path.name) and list((ROOT / path.parent).glob("*.pdf")):
            category = "pdf_page_renders"
        elif path.parent.as_posix() == "tmp/pdfs" and re.fullmatch(r"lfbt-page-\d+\.png", path.name) and list(ROOT.glob("*Barlow*.pdf")):
            category = "source_pdf_page_renders"
        if category:
            if rel in tracked:
                raise RuntimeError(f"Refusing tracked cleanup candidate: {rel}")
            candidates[rel] = {**record, "category": category}
    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    for record in candidates.values():
        groups[record["category"]]["files"] += 1
        groups[record["category"]]["bytes"] += record["bytes"]
    git_files = [p for p in (REPO / ".git").rglob("*") if p.is_file()]
    data = {
        "workspace": str(ROOT), "date": "2026-10-23", "policy": "regenerable files only; no tracked deletions",
        "before_working_files": before, "candidates": candidates,
        "summary": dict(groups), "git_before": git_state(),
        "before_git_bytes": sum(p.stat().st_size for p in git_files),
        "before_git_files": len(git_files),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(MANIFEST, "wt", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False)
    print(json.dumps({"working_files": len(before), "working_bytes": sum(r["bytes"] for r in before.values()), "git_bytes": data["before_git_bytes"], "candidates": dict(groups), "manifest": str(MANIFEST)}, ensure_ascii=False))


def verify() -> None:
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    before = data["before_working_files"]
    candidates = data["candidates"]
    missing, changed, remaining = [], [], []
    preserved = 0
    for rel, record in before.items():
        path = ROOT / rel
        if rel in candidates:
            if path.exists():
                remaining.append(rel)
        elif rel not in EDITABLE:
            if not path.is_file():
                missing.append(rel)
            elif sha(path) != record["sha256"]:
                changed.append(rel)
            else:
                preserved += 1
    archive = REPO / "ECG_SSL_LFBT-main/runlog/preds_archive"
    archive_data = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))
    archives = {}
    for batch, record in archive_data["batches"].items():
        path = archive / record["zip"]
        with zipfile.ZipFile(path) as zipped:
            corrupt = zipped.testzip()
        archives[batch] = {"sha256_ok": sha(path) == record["sha256"], "zip_crc_ok": corrupt is None}
    state = git_state()
    git_equal = {k: state[k] == data["git_before"][k] for k in data["git_before"] if k != "status"}
    v6 = next(ROOT.glob("主论文中文成稿版V6-*.md"))
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", v6.read_text(encoding="utf-8"))
    image_ok = all((ROOT / name).is_file() for name in images)
    after = inventory()
    git_files = [p for p in (REPO / ".git").rglob("*") if p.is_file()]
    result = {
        "missing_protected": missing, "changed_protected": changed, "remaining_candidate_count": len(remaining),
        "preserved_files_checked": preserved, "git_equal": git_equal, "archives": archives,
        "v6_images": len(images), "v6_image_paths_ok": image_ok,
        "deleted_files": len(candidates) - len(remaining),
        "deleted_bytes": sum(r["bytes"] for rel, r in candidates.items() if rel not in remaining),
        "after_working_files": len(after), "after_working_bytes": sum(r["bytes"] for r in after.values()),
        "after_git_files": len(git_files), "after_git_bytes": sum(p.stat().st_size for p in git_files),
        "git_status_after": state["status"],
        "expected_cleanup_edits": sorted(EDITABLE),
    }
    result["cleanup_complete"] = not remaining
    result["preservation_pass"] = not (missing or changed) and all(git_equal.values()) and image_ok and all(all(r.values()) for r in archives.values())
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["preservation_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "verify"])
    args = parser.parse_args()
    {"plan": plan, "verify": verify}[args.mode]()
