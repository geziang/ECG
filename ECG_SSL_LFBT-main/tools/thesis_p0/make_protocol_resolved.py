"""Freeze protocol_resolved.json before the first training job of the batch."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.thesis_p0 import common as C

RESOLVED_PATH = C.RESULTS / "protocol_resolved.json"


def main():
    import torch, sklearn, wfdb, numpy
    pre = json.loads((C.RESULTS / "preflight.json").read_text(encoding="utf-8"))
    dm = json.loads((C.RESULTS / "data_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((C.RESULTS / "source_and_checkpoint_manifest.json").read_text(encoding="utf-8"))

    resolved = {
        "protocol_id": C.PROTOCOL["protocol_id"],
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "冻结于本批首个训练作业前; 训练开始后不再修改",
        "execution_code_commit": C.git_head(),
        "code_dirty_at_freeze": C.code_diff_dirty(),
        "environment": {
            "hostname": pre["machine"]["hostname"],
            "python": pre["environment"]["python_version"].split(" ")[0],
            "torch": torch.__version__, "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__, "wfdb": wfdb.__version__,
            "gpu": pre["environment"].get("gpu_name"),
        },
        "resolved_paths": {
            "git_root": str(C.GIT_ROOT),
            "adaptation_data": str(C.ADAPT),
            "feature_cache": str(C.FEAT_CACHE),
            "results_root": str(C.RESULTS),
            "checkpoints": {f"{m}_s{s}": str(p) for (m, s), p in C.CKPT_PATHS.items()},
            "heartlang_repo": str(Path(r"D:\thesis_p0_20261023_backup\refs\HeartLang")),
            "heartlang_weight": manifest["heartlang_pretrained_checkpoint"]["local_path"],
        },
        "data": {
            "class_order": C.CLASS_ORDER,
            "counts": dm["counts"],
            "npz_sha256": {sp: dm["export"][sp]["npz_sha256"] for sp in ("train", "val")},
            "preprocessing_provenance": dm["preprocessing_provenance"],
        },
        "training_params_local_and_diagnostic": C.PROTOCOL["local_and_diagnostic_training"],
        "representation_diagnostic": C.PROTOCOL["representation_diagnostic"],
        "heartlang": C.PROTOCOL["heartlang"],
        "budgets": C.PROTOCOL["budgets"],
    }
    if RESOLVED_PATH.exists():
        old = json.loads(RESOLVED_PATH.read_text(encoding="utf-8"))
        assert old["execution_code_commit"] == resolved["execution_code_commit"] or old.get("amended"), \
            "protocol_resolved.json already frozen at a different code commit"
    RESOLVED_PATH.write_text(json.dumps(resolved, ensure_ascii=False, indent=1), encoding="utf-8")
    print("frozen ->", RESOLVED_PATH, "code:", resolved["execution_code_commit"][:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
