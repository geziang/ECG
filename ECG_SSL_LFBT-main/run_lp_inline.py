import sys, torch, numpy as np, os
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')

from pathlib import Path
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from models.linear import LinearClassifier
from data_utils.cls_datasets import get_data_loaders
from data_utils.cls_datasets import LinearClsDataset
from models.vgg_1d import VGG16

# Logging
log_file = open(r'D:\LBTF\ECG_SSL_LFBT-main\lp_run_log.txt', 'w', buffering=1)
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

data_dir = r'D:\LBTF\ECG_SSL_LFBT-main\data\downstream/'
checkpoint = Path(r'D:\LBTF\ECG_SSL_LFBT-main\checkpoint\encoder_group.pth')
num_classes = 22
feat_dir = Path(r'D:\LBTF\ECG_SSL_LFBT-main\feat')
num_leads = 8
batch_size = 64
workers = 0
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

log(f'Device: {device}')
log(f'data_dir: {data_dir}')
log(f'checkpoint: {checkpoint}')
log(f'num_classes: {num_classes}')

# Load encoder
load_params = torch.load(checkpoint, map_location=device)
encoder_list = []
lead_names = ["ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"]
for i in range(num_leads):
    encoder = VGG16(ch_in=1, alpha=0.125)
    if 'backbone_state_dict' in load_params:
        missing_keys, unexpected_keys = encoder.load_state_dict(
            load_params['backbone_state_dict'][i].state_dict(), strict=False)
    encoder_list.append(torch.nn.Sequential(*list(encoder.children())[:-1]).to(device))

# Feature inference
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, workers, False)
feat_dir.mkdir(parents=True, exist_ok=True)

def infer_features(loader, name):
    feat_list, label_list = [], []
    log(f"Infer {name} features...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            feat_ld_list = []
            for j in range(num_leads):
                enc = encoder_list[j]
                enc.eval()
                feat_ld = enc(x[:, [j], :]).cpu().numpy()
                feat_ld = np.squeeze(feat_ld)
                feat_ld_list.append(feat_ld)
            feat_ld_cat = np.concatenate(feat_ld_list, axis=-1)
            feat_list.append(feat_ld_cat)
            label_list.append(y.cpu().numpy())
    
    feats = np.concatenate(feat_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    np.save(feat_dir / f"X_{name}.npy", feats)
    np.save(feat_dir / f"y_{name}.npy", labels)
    log(f"{name} feat shape: {feats.shape}, label shape: {labels.shape}")
    return feats, labels

train_feat, train_label = infer_features(train_loader, "train")
val_feat, val_label = infer_features(val_loader, "val")
test_feat, test_label = infer_features(test_loader, "test")
log("Feature inference done.")

# Train linear classifier
train_loader2 = data.DataLoader(LinearClsDataset(feat_dir / "X_train.npy", feat_dir / "y_train.npy"), batch_size=128, shuffle=True)
val_loader2 = data.DataLoader(LinearClsDataset(feat_dir / "X_val.npy", feat_dir / "y_val.npy"), batch_size=128, shuffle=False)

classifier = LinearClassifier(feat_dim=512, num_classes=num_classes).to(device)
epoch = 50
opt = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
min_val_loss = 1e9

log("Start linear classifier training...")
for ep in range(epoch):
    classifier.train()
    for x, y in train_loader2:
        x, y = x.to(device), y.to(device)
        out = classifier(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    classifier.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xv, yv in val_loader2:
            xv, yv = xv.to(device), yv.to(device)
            yv_ = classifier(xv)
            loss = loss_fn(yv_, yv)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader2)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(classifier.state_dict(), feat_dir / "classifier_best_ckpt.pth")
    
    if ep % 10 == 0:
        log(f"Epoch {ep}: val_loss={val_loss:.4f}")

classifier.load_state_dict(torch.load(feat_dir / "classifier_best_ckpt.pth"))
classifier.eval()

# Test
test_loader2 = data.DataLoader(LinearClsDataset(feat_dir / "X_test.npy", feat_dir / "y_test.npy"), batch_size=128, shuffle=False)

y_pred_list, y_true_list, y_pred_s_list = [], [], []
with torch.no_grad():
    for x, y in tqdm(test_loader2):
        x, y = x.to(device), y.to(device)
        output = classifier(x)
        output_softmax = nn.functional.softmax(output, dim=1)
        pred = output.argmax(dim=1)
        y_true_list.append(y.cpu().numpy())
        y_pred_list.append(pred.cpu().numpy())
        y_pred_s_list.append(output_softmax.cpu().numpy())

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)
y_pred_s = np.concatenate(y_pred_s_list)

num_classes_test = len(np.unique(y_true))
y_true_onehot = np.eye(num_classes_test)[y_true]
if y_pred_s.shape[1] < num_classes_test:
    pad_width = num_classes_test - y_pred_s.shape[1]
    y_pred_s = np.pad(y_pred_s, ((0,0),(0,pad_width)), mode='constant')

auroc = roc_auc_score(y_true_onehot, y_pred_s, average="macro", multi_class='ovr')
auprc = average_precision_score(y_true_onehot, y_pred_s, average="macro")
conf_mat = confusion_matrix(y_true, y_pred)
acc = (y_pred == y_true).mean()

log(f"Test Results: ACC={acc:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f}")
log(f"Confusion Matrix:\n{conf_mat}")

log_file.close()
log("Done!")
