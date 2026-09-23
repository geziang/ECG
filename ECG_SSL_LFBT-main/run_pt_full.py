"""Full Pretraining with 200 epochs on full PTB-XL dataset"""
import sys, time, torch
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')

from pathlib import Path

from utils.pathguard import open_out
from torch import nn, optim
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from models.vgg_1d import VGG16

log_file = open_out(Path(__file__).resolve().parent, 'pt_full_log.txt', buffering=1)

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

# Config
data_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\data\pretrain')
checkpoint_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\checkpoint')
num_leads = 8
workers = 0
epochs = 200
batch_size = 8
learning_rate = 0.001
gamma = 0.8
lambd = 0.0051
projector = '2048-2048-2048'
print_freq = 50
device = 'cuda'

log(f'=== Full Pretraining ===')
log(f'device={device} epochs={epochs} batch_size={batch_size} lr={learning_rate} workers={workers}')
log(f'gamma={gamma} lambd={lambd}')
log(f'data: {data_dir}')

checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Build model
log('Building model...')
backbone_group = []
for i in range(num_leads):
    backbone = VGG16(ch_in=1, alpha=0.125)
    backbone.fc = nn.Identity()
    backbone_group.append(backbone.to(device))

sizes = [64] + list(map(int, projector.split('-')))
projector_group = []
for i in range(num_leads):
    layers = []
    for j in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[j + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    projector_group.append(nn.Sequential(*layers).to(device))

bn_group = [nn.BatchNorm1d(sizes[-1], affine=False).to(device) for _ in range(num_leads)]

# Optimizer
param_weights, param_biases = [], []
for md in backbone_group + projector_group + bn_group:
    for param in md.parameters():
        (param_biases if param.ndim == 1 else param_weights).append(param)
optimizer = optim.Adam(param_weights + param_biases, lr=learning_rate)
log(f'Params: {sum(p.numel() for p in param_weights+param_biases):,}')

# Data loader
t = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
dataset = ECGDatasetFolder(data_dir, transform=MultiViewDataInjector([t, t]))
loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True)
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
        y1, y2 = y1.to(device), y2.to(device)
        optimizer.zero_grad()
        
        z1_list, z2_list = [], []
        for i in range(num_leads):
            z1_list.append(projector_group[i](backbone_group[i](y1[:, [i], :])))
            z2_list.append(projector_group[i](backbone_group[i](y2[:, [i], :])))
        
        loss_r, loss_t = 0, 0
        for i in range(num_leads):
            for j in range(num_leads):
                c = bn_group[i](z1_list[i]).T @ bn_group[j](z2_list[j])
                c.div_(batch_size)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                ls = on_diag + lambd * off_diag
                if i == j: loss_r += ls
                else: loss_t += ls
        
        loss_r = loss_r / num_leads
        loss_t = loss_t / (num_leads * (num_leads - 1))
        loss = gamma * loss_r + (1 - gamma) * loss_t
        loss.backward()
        optimizer.step()
        
        if step % print_freq == 0:
            log(f'E[{epoch:3d}] S[{step:3d}/{len(loader)}] '
                f'loss={loss.item():.1f} | R={loss_r.item():.1f} T={loss_t.item():.1f}')
        total_loss += loss.item()
        total_loss_r += loss_r.item()
        total_loss_t += loss_t.item()
    
    avg_loss = total_loss / len(loader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({'epoch': epoch, 'backbone_state_dict': backbone_group,
                     'optimizer': optimizer.state_dict(), 'loss': avg_loss},
                   checkpoint_dir / 'encoder_group_best.pth')
        log(f'  -> Best checkpoint saved (loss={avg_loss:.1f})')
    
    torch.save({'epoch': epoch, 'backbone_state_dict': backbone_group,
                 'optimizer': optimizer.state_dict(), 'loss': avg_loss},
               checkpoint_dir / 'encoder_group.pth')
    
    elapsed = time.time() - ep_start
    log(f'E[{epoch:3d}] DONE | avg_loss={avg_loss:.1f} R={total_loss_r/len(loader):.1f} '
        f'T={total_loss_t/len(loader):.1f} time={elapsed:.0f}s')

log(f'\n=== Pretraining Complete ===')
log(f'Best loss: {best_loss:.1f}')
log_file.close()
