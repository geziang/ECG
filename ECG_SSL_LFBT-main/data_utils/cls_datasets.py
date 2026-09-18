import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
from data_utils.augmentations import ToTensor
from data_utils.data_folder import ECGDatasetFolder


ts = transforms.Compose(
    [
        ToTensor(),
    ]
)


def get_data_loaders(data_path, batch_size, num_workers, train_shuffle=True, train_ratio=1.0, seed=0,
                     return_metadata=False):
    # 用 pathlib 拼接, 不再依赖传入路径是否带末尾分隔符 (指南 §5 P0)
    # return_metadata=True: 额外返回 class_names(run_e006_downstream 需要; 2026-09-18 兼容补)
    root = Path(data_path)
    train_dataset = ECGDatasetFolder(root / "train", transform=ts)
    val_dataset = ECGDatasetFolder(root / "val", transform=ts)
    test_dataset = ECGDatasetFolder(root / "test", transform=ts)
    class_names = list(getattr(train_dataset, "classes", []))  # random_split 前先取

    if train_ratio < 1.0:
        train_num = len(train_dataset)
        select_train_num = int(train_num * train_ratio)
        generator = torch.Generator().manual_seed(seed)
        train_dataset, _ = data.random_split(
            train_dataset, [select_train_num, train_num - select_train_num], generator=generator)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_metadata:
        return train_loader, val_loader, test_loader, class_names
    return train_loader, val_loader, test_loader


class LinearClsDataset(data.Dataset):
    def __init__(self, data_path, label_path):
        self.Data = np.load(data_path)
        self.Label = np.load(label_path)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.Data[idx])
        lb = torch.tensor(self.Label[idx])
        return feat.float(), lb.long()
