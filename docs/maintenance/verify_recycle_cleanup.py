"""Verify the recoverable cleanup against the fresh V7-aware baseline."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "cleanup_recycle_2026-10-23"
EXPECTED_EDITS = {
    "WORKSPACE_INDEX.md", "ECG/PROJECT_MAP.md",
    "ECG/docs/maintenance/CLEANUP_AUDIT_2026-10-23.md",
    "ECG/docs/maintenance/recycle_cleanup.ps1",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-c", "core.quotePath=false", *args], cwd=ROOT / "ECG")


with gzip.open(OUT / "plan.json.gz", "rt", encoding="utf-8") as f:
    plan = json.load(f)
missing, changed = [], []
checked = 0
for rel, record in plan["protected"].items():
    if rel in EXPECTED_EDITS:
        continue
    path = ROOT / rel
    if not path.is_file():
        missing.append(rel)
    elif sha(path) != record["sha256"]:
        changed.append(rel)
    else:
        checked += 1
remaining = [rel for rel in plan["candidates"] if (ROOT / rel).exists()]
journal = [json.loads(line) for line in (OUT / "recycled.jsonl").read_text(encoding="utf-8").splitlines() if line]
journal_ok = len(journal) == len(plan["candidates"]) and len({r["path"] for r in journal}) == len(journal)
for row in journal:
    expected = plan["candidates"].get(row["path"])
    journal_ok = journal_ok and expected is not None and row["operation"] == "SendToRecycleBin" and row["bytes"] == expected["bytes"] and row["sha256"] == expected["sha256"]
state = {name: hashlib.sha256(git(*args)).hexdigest() for name, args in {
    "refs": ["show-ref"], "reflog": ["reflog", "show", "--all", "--format=%H %gs"],
    "objects": ["cat-file", "--batch-all-objects", "--batch-check=%(objectname)"],
}.items()}
state["index"] = sha(ROOT / "ECG/.git/index")
state["head"] = git("rev-parse", "HEAD").decode().strip()
git_equal = {k: value == plan["git_before"][k] for k, value in state.items()}
archives = {}
archive_dir = ROOT / "ECG/ECG_SSL_LFBT-main/runlog/preds_archive"
archive_manifest = json.loads((archive_dir / "archive_manifest.json").read_text(encoding="utf-8"))
for batch, row in archive_manifest["batches"].items():
    path = archive_dir / row["zip"]
    with zipfile.ZipFile(path) as z:
        crc_ok = z.testzip() is None
    archives[batch] = {"sha256_ok": sha(path) == row["sha256"], "zip_crc_ok": crc_ok}
papers = {}
for path in sorted(ROOT.glob("主论文中文成稿版V[67]-*.md")):
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
    papers[path.name] = {"images": len(links), "paths_ok": all((ROOT / link).is_file() for link in links)}
result = {
    "operation": "SendToRecycleBin", "recycled_files": len(journal),
    "recycled_bytes": sum(row["bytes"] for row in journal), "recycle_bin_emptied": False,
    "journal_matches_plan": journal_ok, "remaining_candidate_count": len(remaining),
    "missing_protected": missing, "changed_protected": changed,
    "preserved_files_checked": checked, "expected_cleanup_edits": sorted(EXPECTED_EDITS),
    "git_equal": git_equal, "archives": archives, "papers": papers,
    "git_status_after": git("status", "--porcelain=v1", "-uall").decode("utf-8").splitlines(),
}
result["pass"] = not (missing or changed or remaining) and journal_ok and all(git_equal.values()) and all(all(row.values()) for row in archives.values()) and all(row["paths_ok"] for row in papers.values())
(OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "git_status_after"}, ensure_ascii=True))
if not result["pass"]:
    raise SystemExit(1)
