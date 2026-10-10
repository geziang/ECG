import time, glob, random
import numpy as np
files = glob.glob("data/pt_pretrain_nfh/**/*.npy", recursive=True)
print(f"total npy: {len(files)}")
random.seed(0); sample = random.sample(files, 400)
t0 = time.time(); nbytes = 0
for f in sample:
    a = np.load(f); nbytes += a.nbytes
dt = time.time() - t0
print(f"400 files: {dt:.2f}s, {nbytes/1e6:.1f}MB -> {nbytes/1e6/dt:.1f}MB/s, {400/dt:.1f} files/s")
