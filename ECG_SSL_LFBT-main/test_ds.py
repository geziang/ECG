import sys
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor

print("Building dataset...")
t = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
ds = ECGDatasetFolder(r'D:\LBTF\ECG_SSL_LFBT-main\data\pretrain', transform=MultiViewDataInjector([t, t]))
print('Dataset size:', len(ds))
(y1, y2), _ = ds[0]
print('Sample shapes:', y1.shape, y2.shape)
print('OK')
