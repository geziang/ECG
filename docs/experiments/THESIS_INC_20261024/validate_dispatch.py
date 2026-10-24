"""Validate the dispatched experiment package without loading data or training."""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--execution-time-check", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    root = package.parents[2]
    protocol = json.loads((package / "protocol.json").read_text(encoding="utf-8-sig"))
    lock = json.loads((package / "source_lock.json").read_text(encoding="utf-8-sig"))
    with (package / "task_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert protocol["protocol_id"] == lock["protocol_id"]
    assert protocol["training_authorized_by_this_document"] is True
    assert not protocol["test_policy"]["test_signals_or_label_access_allowed"]
    assert not protocol["test_policy"]["test_metrics_allowed"]
    assert len(rows) == protocol["budgets"]["max_primary_training_jobs"] == 50
    ids = [row["job_id"] for row in rows]
    assert len(set(ids)) == len(ids), "Duplicate job IDs"
    stages = dict(Counter(row["stage"] for row in rows))
    assert stages == protocol["budgets"]["stage_logical_job_caps"], stages
    for row in rows:
        assert row["protocol_id"] == protocol["protocol_id"]
        assert not row["depends_on"] or row["depends_on"] in ids
    assert sum(row["kind"] == "pretrain" for row in rows) == 23
    assert sum(row["kind"] == "linear_probe" for row in rows) == 23
    for row in rows:
        if row["kind"] == "linear_probe":
            parent = next(r for r in rows if r["job_id"] == row["depends_on"])
            for field in ("seed", "pt_epochs", "recipe_or_slot"):
                assert row[field] == parent[field]
    window = protocol["window"]
    times = [datetime.fromisoformat(window[key]) for key in (
        "latest_start_pretraining", "latest_start_gpu_job",
        "gpu_stop_and_save_by", "final_report_by")]
    assert times == sorted(times) and len(set(times)) == 4
    assert times[-1].isoformat() == "2026-10-26T12:00:00+08:00"
    if args.execution_time_check:
        assert datetime.now(timezone.utc) < times[2], "GPU execution window ended"
    for item in lock["new_task_files"]:
        data = (package / item["path"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == item["sha256_lf"], item["path"]
    checked_sources = 0
    if args.verify_sources:
        for item in lock["local_assets"]:
            data = subprocess.check_output(
                ["git", "show", lock["observed_git_base"] + ":" +
                 item["path_relative_to_git_root"]], cwd=root)
            assert hashlib.sha256(data).hexdigest() == item["canonical_git_blob_sha256"]
            checked_sources += 1
    print(json.dumps({"status": "PASS", "protocol_id": protocol["protocol_id"],
                      "matrix_slots": len(rows), "stage_slots": stages,
                      "canonical_source_hashes_checked": checked_sources,
                      "deadline": window["final_report_by"],
                      "gpu_training_started_by_this_script": False},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
