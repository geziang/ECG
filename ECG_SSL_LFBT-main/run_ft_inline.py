import sys, torch, numpy as np, os, time
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')

from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from data_utils.cls_datasets import get_data_loaders
from models.mbn import MultiBranchNet

# Logging
log_file = open(r'D:\LBTF\ECG_SSL_LFBT-main\ft_run_log.txt', 'w', buffering=1)
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

data_dir = r'D:\LBTF\ECG_SSL_LFBT-main\data\downstream/'
checkpoint = Path(r'D:\LBTF\ECG_SSL_LFBT-main\checkpoint\encoder_group.pth')
num_classes = 22
workers = 0
epochs = 20
batch_size = 16
learning_rate = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

log(f'Device: {device}')
log(f'epochs={epochs} batch={batch_size} lr={learning_rate}')

# Load MultiBranchNet
model = MultiBranchNet(num_classes, checkpoint=checkpoint).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, workers, train_ratio=1.0)
log(f'Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}')

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
min_loss = 999999999

for ep in range(1, epochs + 1):
    # Train
    model.train()
    for data, target in tqdm(train_loader, desc=f'Epoch {ep}'):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
    val_loss /= len(val_loader)
    
    if val_loss < min_loss:
        min_loss = val_loss
        log(f'Epoch {ep}: val_loss={val_loss:.4f} (best)')
    else:
        log(f'Epoch {ep}: val_loss={val_loss:.4f}')

# Test
model.eval()
y_pred_list, y_true_list, y_pred_s_list = [], [], []
with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Test'):
        data, target = data.to(device), target.to(device)
        output = model(data)
        output_softmax = nn.functional.softmax(output, dim=1)
        pred = output.argmax(dim=1)
        y_true_list.append(target.cpu().numpy())
        y_pred_list.append(pred.cpu().numpy())
        y_pred_s_list.append(output_softmax.cpu().numpy())

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)
y_pred_s = np.concatenate(y_pred_s_list)

y_true_onehot = np.eye(len(np.unique(y_true)))[y_true]
auroc = roc_auc_score(y_true_onehot, y_pred_s, average="macro", multi_class='ovr')
auprc = average_precision_score(y_true_onehot, y_pred_s, average="macro")
acc = (y_pred == y_true).mean()
conf_mat = confusion_matrix(y_true, y_pred)

log(f'\nFine-Tuning Test Results:')
log(f'ACC={acc:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f}')
log(f'Confusion Matrix:\n{conf_mat}')
log_file.close()
log('Done!')
