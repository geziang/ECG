import subprocess, sys

cmd = [
    r'D:\software\envs\yolov8\python.exe',
    r'D:\LBTF\ECG_SSL_LFBT-main\run_pt.py',
    '--data-dir', r'D:\LBTF\ECG_SSL_LFBT-main\data\pretrain',
    '--epochs', '2',
    '--batch-size', '16',
    '--workers', '0',
    '--print-freq', '1',
    '--checkpoint-dir', r'D:\LBTF\ECG_SSL_LFBT-main\checkpoint'
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd=r'D:\LBTF\ECG_SSL_LFBT-main')
print("=== STDOUT ===")
print(result.stdout)
print("=== STDERR ===")
print(result.stderr)
print(f"=== Exit code: {result.returncode} ===")
