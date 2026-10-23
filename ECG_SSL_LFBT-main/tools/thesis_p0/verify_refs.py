"""Verify reference source files at pinned commits against source_lock.json."""
import hashlib
import json
import sys
from pathlib import Path

GIT_ROOT = Path(r"F:\新实验")
LOCK = json.loads((GIT_ROOT / "docs/experiments/THESIS_P0_20261023/source_lock.json").read_text(encoding="utf-8"))
REFS = Path(r"D:\thesis_p0_20261023_backup\refs")
REPO_DIR = {"heartlang": "HeartLang", "ecg_tokenizer": "ecg_tokenizer", "shapeefm": "shapeefm"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    results, issues = [], []
    for ref in LOCK["references"]:
        repo_dir = REPO_DIR[ref["id"]]
        for req in ref["required_source_files"]:
            p = REFS / repo_dir / req["path"]
            entry = {"repo": ref["id"], "commit": ref["commit"], "path": req["path"], "expected_sha256": req["sha256"]}
            if not p.exists():
                entry["status"] = "missing"
                issues.append(f"{ref['id']}:{req['path']} missing")
            else:
                actual = sha256_of(p)
                entry["actual_sha256"] = actual
                entry["status"] = "ok" if actual == req["sha256"] else "sha_mismatch"
                if entry["status"] != "ok":
                    issues.append(f"{ref['id']}:{req['path']} sha mismatch")
            results.append(entry)
    out = {
        "refs_root": str(REFS),
        "files": results,
        "passed": not issues,
        "issues": issues,
    }
    dest = GIT_ROOT / "ECG_SSL_LFBT-main/runlog/THESIS_P0_20261023/refs_verification.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"refs: {n_ok}/{len(results)} verified")
    for i in issues:
        print("ISSUE:", i)
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
