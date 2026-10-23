"""Combine preflight + refs verification + HeartLang weight status into
source_and_checkpoint_manifest.json. Re-runnable: fills weight actual hash
once the download finishes."""
import hashlib
import json
import sys
import time
from pathlib import Path

RESULTS = Path(r"F:\新实验\ECG_SSL_LFBT-main\runlog\THESIS_P0_20261023")
WEIGHT = Path(r"D:\thesis_p0_20261023_backup\heartlang_weights\checkpoint-200.pth")
EXPECTED_WEIGHT_SHA = "9dc89d72a3f1cacd941892103eaa523d44e373164cd4fadfb975907cc65864f6"
EXPECTED_WEIGHT_BYTES = 534928678


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pre = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    refs = json.loads((RESULTS / "refs_verification.json").read_text(encoding="utf-8"))

    weight = {
        "repo": "https://huggingface.co/PKUDigitalHealth/HeartLang",
        "revision": "b8118c1bc34a43cc2202b6d30267244ac7fc52c1",
        "path_in_repo": "checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth",
        "expected_sha256_lfs": EXPECTED_WEIGHT_SHA,
        "expected_bytes": EXPECTED_WEIGHT_BYTES,
        "local_path": str(WEIGHT),
    }
    if WEIGHT.exists():
        weight["actual_bytes"] = WEIGHT.stat().st_size
        if WEIGHT.stat().st_size == EXPECTED_WEIGHT_BYTES:
            weight["actual_sha256"] = sha256_of(WEIGHT)
            weight["verified"] = weight["actual_sha256"] == EXPECTED_WEIGHT_SHA
        else:
            weight["verified"] = False
            weight["note"] = "size mismatch - download incomplete?"
    else:
        weight["verified"] = False
        weight["note"] = "not yet downloaded"

    manifest = {
        "protocol_id": "thesis-p0-20261023-v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_snapshot": {
            "git_commit": pre["git"]["head"],
            "archive": "D:\\thesis_p0_20261023_backup\\code_snapshot_3c3c35c.zip",
            "archive_sha256": "4dbc46f67156d8ac73207b86bde4a83e97cf4420f1a550d096fe6b041f22a6f8",
            "dirty_files_at_snapshot": pre["git"]["dirty_files"],
        },
        "checkpoints": pre["checkpoints"],
        "local_assets": pre["local_assets"],
        "reference_sources": refs,
        "heartlang_pretrained_checkpoint": weight,
    }
    dest = RESULTS / "source_and_checkpoint_manifest.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("heartlang weight verified:", weight.get("verified"))
    print("->", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
