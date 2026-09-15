import sys, time, torch, os
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')

from pathlib import Path
from torch import nn, optim
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from models.vgg_1d import VGG16

# Write log to file
log_file = open(r'D:\LBTF\ECG_SSL_LFBT-main\pt_run_log.txt', 'w', buffering=1)
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

data_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\data\pretrain')
checkpoint_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\checkpoint')
num_leads = 8
workers = 0
epochs = 2
batch_size = 16
learning_rate = 0.001
gamma = 0.8
lambd = 0.0051
projector = '2048-2048-2048'
print_freq = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

log(f'Device: {device}')
log(f'epochs={epochs} batch={batch_size} lr={learning_rate}')

checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

bn_group = []
for i in range(num_leads):
    bn_group.append(nn.BatchNorm1d(sizes[-1], affine=False).to(device))

param_weights, param_biases = [], []
for md in backbone_group + projector_group + bn_group:
    for param in md.parameters():
        (param_biases if param.ndim == 1 else param_weights).append(param)

optimizer = optim.Adam(param_weights + param_biases, lr=learning_rate)

t = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
dataset = ECGDatasetFolder(data_dir, transform=MultiViewDataInjector([t, t]))
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True)
log(f'Dataset: {len(dataset)} samples, {len(loader)} batches')

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

for epoch in range(epochs):
    total_loss, total_loss_r, total_loss_t = 0, 0, 0
    ep_start_time = time.time()
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
            log(f'E{epoch} S{step}: loss={loss.item():.4f} R={loss_r.item():.4f} T={loss_t.item():.4f}')
        total_loss += loss.item()
        total_loss_r += loss_r.item()
        total_loss_t += loss_t.item()
    
    total_loss /= len(loader)
    log(f'Epoch {epoch} done: avg_loss={total_loss:.4f} R_avg={total_loss_r/len(loader):.4f} T_avg={total_loss_t/len(loader):.4f} time={time.time()-ep_start_time:.1f}s')

torch.save({'backbone_state_dict': backbone_group}, checkpoint_dir / 'encoder_group.pth')
log('Checkpoint saved to encoder_group.pth!')
log_file.close()
