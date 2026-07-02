
"""
Numerical conditional 3D rock dataset processing script
Supports three numerical conditions: porosity, pore size mean, pore size std
Porosity is computed in real-time, pore size parameters are parsed from filenames

{pore_size_mean}_{pore_size_std}.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import re
import json
from pathlib import Path


def compute_rock_physics_parameters(volume: np.ndarray) -> Dict[str, float]:
    
    from scipy import ndimage
    
    # Label connected components
    labeled_volume, num_features = ndimage.label(volume == 0)  # Pore connected components
    
    if num_features > 0:
        # Compute the size of each connected component
        feature_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled_volume == i)
            if size > 0:
                feature_sizes.append(size)
        
        if feature_sizes:
            # Pore size mean (based on connected component size)
            pore_size_mean = np.mean(feature_sizes) ** (1/3)  # Cube root as linear dimension
            # Pore size std
            pore_size_std = np.std(feature_sizes) ** (1/3)
        else:
            pore_size_mean = 1.0
            pore_size_std = 0.1
    else:
        pore_size_mean = 1.0
        pore_size_std = 0.1
    
    return {
        'pore_size_mean': float(pore_size_mean),
        'pore_size_std': float(pore_size_std)
    }


def validate_parameters(params: Dict[str, float]) -> bool:
   
    pore_size_mean = params.get('pore_size_mean', 0)
    pore_size_std = params.get('pore_size_std', 0)
    
    # Basic sanity check
    if pore_size_mean <= 0 or pore_size_std < 0:
        return False
    
    return True

class NumericalRocks3DDataset(Dataset):
    
    def __init__(self, 
                 data_root: str,
                 target_shape: Tuple[int, int, int] = (64, 64, 64),
                 condition_ranges: Dict = None,
                 transform=None,
                 auto_generate_labels: bool = True):
        """
        Initialize dataset
        
        Args:
            data_root: Data root directory
            target_shape: Target shape (D, H, W)
            condition_ranges: Condition ranges
            transform: Data transform
            auto_generate_labels: Whether to auto-generate labels (from filename or random)
        """
        self.data_root = data_root
        self.target_shape = target_shape
        self.transform = transform
        self.auto_generate_labels = auto_generate_labels
        
        self.condition_ranges = None  
        
        # Load data files
        self.data_files = self._load_data_files()
        print(f"Loaded {len(self.data_files)} numerical condition data files")
        
        # Analyze condition distribution
        self._analyze_conditions()
    
    def _load_data_files(self) -> List[Dict]:
        """Load data files and parse conditions"""
        data_files = []
        
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data directory does not exist: {self.data_root}")
        
        for filename in os.listdir(self.data_root):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.data_root, filename)
                
                # Try to parse conditions from filename
                conditions = self._parse_filename_conditions(filename)
                
                # If filename parsing fails, compute real conditions
                if conditions is None:
                   
                    if filename.startswith('limestone_'):
                        try:
                            # Load data and compute real conditions
                            volume = np.load(filepath)
                            conditions = self._compute_real_conditions(volume)
                            print(f"Computed real conditions for {filename}: {conditions}")
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
                            if self.auto_generate_labels:
                                conditions = self._generate_random_conditions()
                                print(f"Generated random conditions for {filename}: {conditions}")
                            else:
                                print(f"Skipping {filename} (cannot parse conditions and auto-generation disabled)")
                                continue
                    else:
                       
                        print(f"Skipping {filename} (filename format mismatch: {self._get_expected_format()})")
                        continue
                
                if conditions is not None:
                    # Determine condition source
                    if filename.startswith('limestone_'):
                        source = "computation"
                    else:
                        source = "filename"
                    
                    print(f"Read from {source} {filename}: pore_size_mean={conditions['pore_size_mean']:.3f}, pore_size_std={conditions['pore_size_std']:.3f}")
                    
                    data_files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'pore_size_mean': conditions['pore_size_mean'],
                        'pore_size_std': conditions['pore_size_std']
                    })
        
        return sorted(data_files, key=lambda x: x['filename'])
    
    def _parse_filename_conditions(self, filename: str) -> Optional[Dict]:
        """Parse conditions from filename"""
        
        if filename.endswith('.npy'):
            name_without_ext = filename[:-4]  # Remove .npy extension
            parts = name_without_ext.split('_')
            
            if len(parts) == 2:
                try:
                    pore_size_mean = float(parts[0])
                    pore_size_std = float(parts[1])
                    return {
                        'pore_size_mean': pore_size_mean,
                        'pore_size_std': pore_size_std
                    }
                except ValueError:
                    print(f"Warning: {filename} cannot parse values: {parts[0]}, {parts[1]}")
                    return None
            else:
                print(f"Warning: {filename} format incorrect, expected: value1_value2.npy, got: {name_without_ext}")
                return None
        
        return None
    
    def _get_expected_format(self) -> str:
        """Return expected filename format"""
        return "Format: value1_value2.npy (e.g. 1.234_0.567.npy or 10.019_13.516.npy)"
    
    def _compute_real_conditions(self, volume: np.ndarray) -> Dict[str, float]:
        """Compute real rock physics parameters"""
        try:
            # Use rock physics calculator to compute real parameters
            params = compute_rock_physics_parameters(volume)
            
            # Validate parameter reasonableness
            if validate_parameters(params):
                return params
            else:
                print(f"Warning: computed parameters are unreasonable: {params}")
                return self._generate_random_conditions()
        except Exception as e:
            print(f"Error computing rock physics parameters: {e}")
            return self._generate_random_conditions()
    
    def _generate_random_conditions(self) -> Dict:
        """Generate random conditions"""
        pore_size_mean = np.random.uniform(
            self.condition_ranges['pore_size_mean'][0],
            self.condition_ranges['pore_size_mean'][1]
        )
        pore_size_std = np.random.uniform(
            self.condition_ranges['pore_size_std'][0],
            self.condition_ranges['pore_size_std'][1]
        )
        
        return {
            'pore_size_mean': pore_size_mean,
            'pore_size_std': pore_size_std
        }
    
    def _analyze_conditions(self):
        """Analyze condition distribution"""
        if not self.data_files:
            return
        
        pore_size_means = [f['pore_size_mean'] for f in self.data_files]
        pore_size_stds = [f['pore_size_std'] for f in self.data_files]
        
        # Statistics on data source
        from_filename = sum(1 for f in self.data_files if not f['filename'].startswith('limestone_'))
        from_computation = sum(1 for f in self.data_files if f['filename'].startswith('limestone_'))
        
        print(f"Condition distribution statistics:")
        print(f"   Pore size mean: {min(pore_size_means):.3f} - {max(pore_size_means):.3f} (mean: {np.mean(pore_size_means):.3f})")
        print(f"   Pore size std: {min(pore_size_stds):.3f} - {max(pore_size_stds):.3f} (mean: {np.mean(pore_size_stds):.3f})")
        print(f"   Porosity: computed in real-time (MCDDPM style)")
        print(f"   Data source: {from_filename} files from filename, {from_computation} files from computation")
        print(f"   Parsing method: split filename by underscore (MCDDPM style)")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """Get data item - based on MCDDPM logic"""
        data_info = self.data_files[idx]
        
        # Load 3D data
        volume = np.load(data_info['filepath'])
        
        # # Ensure data is binarized
        # if volume.dtype != np.uint8:
        #     volume = (volume > 0.5).astype(np.uint8)
        
        # Resize shape
        if volume.shape != self.target_shape:
            volume = self._resize_volume(volume, self.target_shape)
        
        # Compute porosity in real-time
        porosity = np.sum(volume == 0) / volume.size
        
        # Convert to tensor
        volume = torch.from_numpy(volume).float()  # [D, H, W]
        
        # Apply transform
        if self.transform:
            volume = self.transform(volume)
        
        # Normalize conditions to [0,1] range (only normalize pore size parameters)
        normalized_conditions = self._normalize_conditions({
            'pore_size_mean': data_info['pore_size_mean'],
            'pore_size_std': data_info['pore_size_std']
        })
        
        return {
            'image': volume,
            'porosity': torch.tensor([porosity], dtype=torch.float32),  # Real-time computed porosity
            'pore_size_mean': torch.tensor([normalized_conditions['pore_size_mean']], dtype=torch.float32),
            'pore_size_std': torch.tensor([normalized_conditions['pore_size_std']], dtype=torch.float32),
            'filename': data_info['filename']
        }
    
    def _resize_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D voxel, preserving binary characteristics"""
        from scipy.ndimage import zoom
        
        zoom_factors = [
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1],
            target_shape[2] / volume.shape[2]
        ]
        
        resized = zoom(volume, zoom_factors, order=0)  # Nearest neighbor interpolation
        return (resized > 0.5).astype(np.uint8)  
    
    def _normalize_conditions(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Normalize conditions to [0,1] range - MCDDPM style"""
        normalized = {}
        for key, value in conditions.items():
            if self.condition_ranges and key in self.condition_ranges:
                min_val, max_val = self.condition_ranges[key]
                normalized[key] = (value - min_val) / (max_val - min_val)
                normalized[key] = np.clip(normalized[key], 0.0, 1.0)
            else:
               
                normalized[key] = value
        return normalized

class NumericalRocks3DDataModule:
    
    def __init__(self,
                 train_data_root: str,
                 val_data_root: str = None,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 target_shape: Tuple[int, int, int] = (64, 64, 64),
                 condition_ranges: Dict = None,
                 auto_generate_labels: bool = True,
                 **kwargs):
        """
        Initialize data module
        
        Args:
            train_data_root: Training data directory
            val_data_root: Validation data directory
            batch_size: Batch size
            num_workers: Number of worker processes
            target_shape: Target shape
            condition_ranges: Condition ranges
            auto_generate_labels: Whether to auto-generate labels
        """
        self.train_data_root = train_data_root
        self.val_data_root = val_data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_shape = target_shape
        self.condition_ranges = condition_ranges
        self.auto_generate_labels = auto_generate_labels
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        
        if self.train_data_root:
            self.train_dataset = NumericalRocks3DDataset(
                data_root=self.train_data_root,
                target_shape=self.target_shape,
                condition_ranges=self.condition_ranges,
                auto_generate_labels=self.auto_generate_labels
            )
        
        if self.val_data_root:
            self.val_dataset = NumericalRocks3DDataset(
                data_root=self.val_data_root,
                target_shape=self.target_shape,
                condition_ranges=self.condition_ranges,
                auto_generate_labels=self.auto_generate_labels
            )
    
    def train_dataloader(self):
        """Training data loader"""
        if self.train_dataset is None:
            return None
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Validation data loader"""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def create_labeled_dataset(source_dir: str, 
                         output_dir: str, 
                         condition_ranges: Dict = None,
                         num_samples: int = None):
    """
    Create a labeled version for an existing dataset
    
    Args:
        source_dir: Source data directory
        output_dir: Output directory
        condition_ranges: Condition ranges
        num_samples: Number of samples to generate (if None, process all files)
    """
    if condition_ranges is None:
        condition_ranges = {
            'porosity': [0.05, 0.35],
            'pore_size_mean': [0.1, 2.0],
            'pore_size_std': [0.05, 0.5]
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all npy files
    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    if num_samples is not None:
        npy_files = npy_files[:num_samples]
    
    print(f"Processing {len(npy_files)} files...")
    
    # Process each file
    for i, filename in enumerate(npy_files):
        # Generate random conditions
        porosity = np.random.uniform(condition_ranges['porosity'][0], condition_ranges['porosity'][1])
        pore_size_mean = np.random.uniform(condition_ranges['pore_size_mean'][0], condition_ranges['pore_size_mean'][1])
        pore_size_std = np.random.uniform(condition_ranges['pore_size_std'][0], condition_ranges['pore_size_std'][1])
        
        # Create new filename
        new_filename = f"{porosity:.3f}_{pore_size_mean:.3f}_{pore_size_std:.3f}.npy"
        
        # Copy file
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, new_filename)
        
        # Copy file
        import shutil
        shutil.copy2(source_path, output_path)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(npy_files)} files")
    
    print(f"Done! Created {len(npy_files)} labeled files in {output_dir}")
    
    # Save condition range info
    ranges_info = {
        'condition_ranges': condition_ranges,
        'num_samples': len(npy_files),
        'source_dir': source_dir
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(ranges_info, f, indent=2)
    
    print(f"Dataset info saved to {output_dir}/dataset_info.json")


def test_dataset():
    """Test dataset"""
    # Create test data
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate some test data
    for i in range(5):
        # Generate random 3D data
        volume = np.random.randint(0, 2, size=(64, 64, 64), dtype=np.uint8)
        
        # Generate random conditions
        porosity = np.random.uniform(0.1, 0.3)
        pore_size_mean = np.random.uniform(0.5, 1.5)
        pore_size_std = np.random.uniform(0.1, 0.4)
        
        # Save file
        filename = f"{porosity:.3f}_{pore_size_mean:.3f}_{pore_size_std:.3f}.npy"
        np.save(os.path.join(test_dir, filename), volume)
    
    # Test dataset
    dataset = NumericalRocks3DDataset(
        data_root=test_dir,
        target_shape=(64, 64, 64),
        auto_generate_labels=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    sample = dataset[0]
    print(f"Sample info:")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Porosity: {sample['porosity'].item():.3f}")
    print(f"   Pore size mean: {sample['pore_size_mean'].item():.3f}")
    print(f"   Pore size std: {sample['pore_size_std'].item():.3f}")
    print(f"   Filename: {sample['filename']}")
    
    # Clean up test data
    import shutil
    shutil.rmtree(test_dir)
    
    print("Dataset test complete!")


if __name__ == '__main__':
    # Test dataset
    test_dataset()

