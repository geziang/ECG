"""
Preprocess PTB-XL data from .mat/.hea to .npy format for LFBT model.
- Selects 8 leads: II, III, V1, V2, V3, V4, V5, V6
- Z-score normalizes each lead
- Resamples to 1000Hz (from 500Hz)  
- Outputs pretrain data (flat folder) and downstream data (train/val/test by class)
"""
import os
import sys
import numpy as np
import scipy.io
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
from scipy.signal import resample

# PTB-XL has 12 leads: I(0), II(1), III(2), aVR(3), aVL(4), aVF(5), V1(6), V2(7), V3(8), V4(9), V5(10), V6(11)
# LFBT model uses 8 leads
LEAD_INDICES = [1, 2, 6, 7, 8, 9, 10, 11]  # II, III, V1, V2, V3, V4, V5, V6
TARGET_SAMPLING_RATE = 1000  # Hz (resample from 500Hz)
ORIG_SAMPLING_RATE = 500
TARGET_LENGTH = 5000  # samples (5 seconds at 1000Hz, or keep original 5000 from 500Hz=10s)

def ptbxl_class_map():
    """PTB-XL diagnostic superclass mapping based on SCP codes"""
    return {
        'g1': 'NORM',    # Normal ECG
        'g2': 'MI',      # Myocardial Infarction
        'g3': 'STTC',    # ST/T Change
        'g4': 'CD',      # Conduction Disturbance
        'g5': 'HYP',     # Hypertrophy
    }

def z_score_normalize(sig):
    """Z-score normalize each lead"""
    mean = np.mean(sig, axis=1, keepdims=True)
    std = np.std(sig, axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (sig - mean) / std

def process_mat_file(mat_path, target_len=5000):
    """Load .mat file, select 8 leads, normalize, optionally resample"""
    try:
        data = scipy.io.loadmat(str(mat_path))
        ecg = data['val']  # shape: (12, 5000)
        
        # Select 8 leads
        ecg = ecg[LEAD_INDICES, :].astype(np.float64)
        
        # Resample to target length
        current_len = ecg.shape[1]
        if current_len != target_len:
            ecg = resample(ecg, target_len, axis=1)
        
        # Z-score normalize
        ecg = z_score_normalize(ecg)
        
        return ecg
    except Exception as e:
        print(f"Error processing {mat_path}: {e}")
        return None

def build_pretrain_data(ptbxl_dir, output_dir, target_len=5000, max_samples=None):
    """Build pretraining dataset (all PTB-XL data, no labels needed)"""
    output_dir = Path(output_dir)
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all mat files
    mat_files = list(Path(ptbxl_dir).rglob('*.mat'))
    if max_samples:
        mat_files = mat_files[:max_samples]
    
    print(f"Processing {len(mat_files)} files for pretraining...")
    count = 0
    for mat_path in tqdm(mat_files, desc="Pretrain"):
        ecg = process_mat_file(mat_path, target_len)
        if ecg is not None:
            out_name = mat_path.stem + '.npy'
            np.save(samples_dir / out_name, ecg)
            count += 1
    
    print(f"\nPretrain: {count} samples saved to {samples_dir}")
    return count

def build_downstream_data(ptbxl_dir, output_dir, target_len=5000, 
                          test_size=0.15, val_size=0.15, max_per_class=None):
    """Build downstream dataset with train/val/test split by class"""
    output_dir = Path(output_dir)
    
    # Collect files by class group
    class_files = {}
    for group_dir in sorted(Path(ptbxl_dir).iterdir()):
        if group_dir.is_dir() and group_dir.name.startswith('g'):
            class_name = group_dir.name
            mat_files = list(group_dir.glob('*.mat'))
            if max_per_class:
                mat_files = mat_files[:max_per_class]
            if mat_files:
                class_files[class_name] = mat_files
    
    print(f"Found {len(class_files)} classes: {list(class_files.keys())}")
    for cls, files in class_files.items():
        print(f"  {cls}: {len(files)} files")
    
    # For each class, split and save
    for class_name, mat_files in class_files.items():
        # Split
        train_files, test_files = train_test_split(mat_files, test_size=test_size, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=val_size/(1-test_size), random_state=42)
        
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for mat_path in tqdm(files, desc=f"{class_name}/{split_name}", leave=False):
                ecg = process_mat_file(mat_path, target_len)
                if ecg is not None:
                    out_name = mat_path.stem + '.npy'
                    np.save(split_dir / out_name, ecg)
        
        print(f"  {class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    total = sum(len(v) for v in class_files.values())
    print(f"\nTotal downstream samples: {total}")
    print(f"Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess PTB-XL for LFBT')
    parser.add_argument('--ptbxl-dir', type=str, required=True, help='Path to PTB-XL raw data directory')
    parser.add_argument('--pt-output', type=str, default='./data/pretrain', help='Pretrain output directory')
    parser.add_argument('--down-output', type=str, default='./data/downstream', help='Downstream output directory')
    parser.add_argument('--target-len', type=int, default=5000, help='Target signal length')
    parser.add_argument('--max-pretrain', type=int, default=None, help='Max pretrain samples')
    parser.add_argument('--max-per-class', type=int, default=None, help='Max samples per class for downstream')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Building pretraining dataset...")
    print("="*60)
    build_pretrain_data(args.ptbxl_dir, args.pt_output, args.target_len, args.max_pretrain)
    
    print("\n" + "="*60)
    print("Building downstream dataset...")
    print("="*60)
    build_downstream_data(args.ptbxl_dir, args.down_output, args.target_len, max_per_class=args.max_per_class)

if __name__ == '__main__':
    main()
