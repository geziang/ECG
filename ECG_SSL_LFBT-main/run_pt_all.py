"""Quick full-pipeline verification with reduced settings"""
import sys, time, torch, gc
import numpy as np
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')

from pathlib import Path
from torch import nn, optim
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from models.vgg_1d import VGG16

log_path = r'D:\LBTF\ECG_SSL_LFBT-main\pt_full_log.txt'
log_file = open(log_path, 'w', buffering=1)

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

# Memory-constrained settings
device = 'cuda'
batch_size = 4
epochs = 10  # Quick verification
workers = 0
learning_rate = 0.001
gamma = 0.8
lambd = 0.0051
projector = '2048-2048-2048'
print_freq = 200

data_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\data\pretrain')
checkpoint_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\checkpoint')

log(f'=== Pretraining (CUDA) ===')
log(f'device={device} epochs={epochs} batch={batch_size}')

torch.cuda.empty_cache()
gc.collect()

# Build model on CPU first, then move to CUDA one at a time
log('Building model on CPU...')
backbone_group = []
for i in range(8):
    backbone = VGG16(ch_in=1, alpha=0.125)
    backbone.fc = nn.Identity()
    backbone_group.append(backbone)
log(f'  {len(backbone_group)} backbones built')

sizes = [64] + list(map(int, projector.split('-')))
projector_group = []
for i in range(8):
    layers = []
    for j in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[j + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    projector_group.append(nn.Sequential(*layers))
log(f'  {len(projector_group)} projectors built')

bn_group = [nn.BatchNorm1d(sizes[-1], affine=False) for _ in range(8)]
log('Moving models to CUDA...')
gc.collect()
torch.cuda.empty_cache()
for i in range(8):
    backbone_group[i] = backbone_group[i].to(device)
    projector_group[i] = projector_group[i].to(device)
    bn_group[i] = bn_group[i].to(device)
    if i % 2 == 0:
        torch.cuda.empty_cache()
log('All models on CUDA')

# Optimizer
param_weights, param_biases = [], []
for md in backbone_group + projector_group + bn_group:
    for param in md.parameters():
        (param_biases if param.ndim == 1 else param_weights).append(param)
total_params = sum(p.numel() for p in param_weights + param_biases)
log(f'Params: {total_params:,}')
optimizer = optim.Adam(param_weights + param_biases, lr=learning_rate)

# Data
t = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
dataset = ECGDatasetFolder(data_dir, transform=MultiViewDataInjector([t, t]))
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
log(f'Dataset: {len(dataset)} samples, {len(loader)} batches/epoch')

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

best_loss = float('inf')
for epoch in range(epochs):
    total_loss, total_loss_r, total_loss_t = 0, 0, 0
    ep_start = time.time()
    
    for step, ((y1, y2), _) in enumerate(loader):
        optimizer.zero_grad()
        y1, y2 = y1.to(device), y2.to(device)
        
        z1_list, z2_list = [], []
        for i in range(8):
            z1_list.append(projector_group[i](backbone_group[i](y1[:, [i], :])))
            z2_list.append(projector_group[i](backbone_group[i](y2[:, [i], :])))
        
        loss_r, loss_t = 0, 0
        for i in range(8):
            for j in range(8):
                c = bn_group[i](z1_list[i]).T @ bn_group[j](z2_list[j])
                c.div_(batch_size)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                ls = on_diag + lambd * off_diag
                if i == j: loss_r += ls
                else: loss_t += ls
        
        loss_r = loss_r / 8
        loss_t = loss_t / (8 * 7)
        loss = gamma * loss_r + (1 - gamma) * loss_t
        loss.backward()
        optimizer.step()
        
        if step % print_freq == 0:
            log(f'E[{epoch:3d}] S[{step:3d}/{len(loader)}] loss={loss.item():.1f} | R={loss_r.item():.1f} T={loss_t.item():.1f}')
        total_loss += loss.item()
        total_loss_r += loss_r.item()
        total_loss_t += loss_t.item()
    
    avg_loss = total_loss / len(loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({'epoch': epoch, 'backbone_state_dict': backbone_group, 'loss': avg_loss},
                   checkpoint_dir / 'encoder_group_best.pth')
    
    torch.save({'epoch': epoch, 'backbone_state_dict': backbone_group, 'loss': avg_loss},
               checkpoint_dir / 'encoder_group.pth')
    
    elapsed = time.time() - ep_start
    log(f'E[{epoch:3d}] avg_loss={avg_loss:.1f} R={total_loss_r/len(loader):.1f} T={total_loss_t/len(loader):.1f} time={elapsed:.0f}s')
    gc.collect()

log(f'\nBest loss: {best_loss:.1f}')
log_file.close()
