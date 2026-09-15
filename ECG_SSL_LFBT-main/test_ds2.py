import sys, os
sys.path.insert(0, r'D:\LBTF\ECG_SSL_LFBT-main')

log_path = r'D:\LBTF\ECG_SSL_LFBT-main\test_result.txt'
with open(log_path, 'w') as f:
    f.write("Starting...\n")
    try:
        f.write("Importing torchvision...\n")
        import torchvision.transforms as transforms
        f.write("Importing data_utils...\n")
        from data_utils.data_folder import ECGDatasetFolder
        from data_utils.multi_view_data_injector import MultiViewDataInjector
        from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
        
        f.write("Building dataset...\n")
        t = transforms.Compose([RandomResizeCropTimeOut(), ToTensor()])
        ds = ECGDatasetFolder(r'D:\LBTF\ECG_SSL_LFBT-main\data\pretrain', transform=MultiViewDataInjector([t, t]))
        f.write(f'Dataset size: {len(ds)}\n')
        (y1, y2), _ = ds[0]
        f.write(f'Sample shapes: {y1.shape}, {y2.shape}\n')
        f.write('OK\n')
    except Exception as e:
        f.write(f'ERROR: {e}\n')
        import traceback
        f.write(traceback.format_exc())

print('Wrote to', log_path)
