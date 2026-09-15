import zipfile
import os
import shutil
import sys

wheel = os.path.join(os.path.dirname(__file__), 'torch.whl')
dest = os.path.join(os.path.dirname(__file__), 'torch_extract')

print(f'Extracting {wheel} to {dest}')
os.makedirs(dest, exist_ok=True)

with zipfile.ZipFile(wheel, 'r') as z:
    z.extractall(dest)

print('Extraction done!')

# Now copy torch to the Python site-packages
site_packages = sys.prefix
# Find torch directory in the extracted content
torch_dir = os.path.join(dest, 'torch')
torch_info = os.path.join(dest, 'torch-2.5.1+cu121.dist-info')

if os.path.exists(torch_dir):
    target_torch = os.path.join(site_packages, 'Lib', 'site-packages', 'torch')
    target_info = os.path.join(site_packages, 'Lib', 'site-packages', 'torch-2.5.1+cu121.dist-info')
    
    if os.path.exists(target_torch):
        shutil.rmtree(target_torch)
    if os.path.exists(target_info):
        shutil.rmtree(target_info)
    
    shutil.copytree(torch_dir, target_torch)
    shutil.copytree(torch_info, target_info)
    print('torch installed to site-packages!')
else:
    print('torch directory not found in extracted content')
    for item in os.listdir(dest):
        print(f'  {item}')
