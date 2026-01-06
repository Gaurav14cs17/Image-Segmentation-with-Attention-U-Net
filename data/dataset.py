"""
Dataset classes for image segmentation with U²-Net

Supports various segmentation dataset formats:
- Paired image/mask directories
- DUTS dataset format
- Custom dataset formats
"""

import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset for paired image and mask files.
    
    Supports multiple directory structures:
        Structure 1 (default):
            data_dir/
                images/
                    image1.jpg
                    ...
                masks/
                    image1.png
                    ...
        
        Structure 2 (IN/GT):
            data_dir/
                IN/
                    image1.png
                    ...
                GT/
                    image1.png
                    ...
    
    Args:
        data_dir: Root directory containing image and mask folders
        image_size: Target size for images (default: 320)
        augment: Whether to apply data augmentation (default: False)
        image_ext: Image file extensions to look for
        mask_ext: Mask file extensions to look for
        image_folder: Name of folder containing images (default: auto-detect)
        mask_folder: Name of folder containing masks (default: auto-detect)
    """
    
    def __init__(self, data_dir, image_size=320, augment=False,
                 image_ext=('.jpg', '.jpeg', '.png', '.bmp'),
                 mask_ext=('.png', '.jpg', '.bmp'),
                 image_folder=None, mask_folder=None):
        super().__init__()
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        
        # Auto-detect folder structure
        if image_folder is None or mask_folder is None:
            # Check for IN/GT structure first
            if os.path.exists(os.path.join(data_dir, 'IN')) and os.path.exists(os.path.join(data_dir, 'GT')):
                image_folder = 'IN'
                mask_folder = 'GT'
            # Then check for images/masks structure
            elif os.path.exists(os.path.join(data_dir, 'images')) and os.path.exists(os.path.join(data_dir, 'masks')):
                image_folder = 'images'
                mask_folder = 'masks'
            else:
                # Default fallback
                image_folder = 'images'
                mask_folder = 'masks'
        
        self.image_dir = os.path.join(data_dir, image_folder)
        self.mask_dir = os.path.join(data_dir, mask_folder)
        
        print(f"Using image folder: {self.image_dir}")
        print(f"Using mask folder: {self.mask_dir}")
        
        # Get list of images
        self.images = []
        self.masks = []
        
        if os.path.exists(self.image_dir):
            for filename in sorted(os.listdir(self.image_dir)):
                if filename.lower().endswith(image_ext):
                    image_path = os.path.join(self.image_dir, filename)
                    
                    # Find corresponding mask
                    base_name = os.path.splitext(filename)[0]
                    mask_path = None
                    
                    for ext in mask_ext:
                        potential_mask = os.path.join(self.mask_dir, base_name + ext)
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                            break
                    
                    if mask_path is not None:
                        self.images.append(image_path)
                        self.masks.append(mask_path)
        
        print(f"Found {len(self.images)} image-mask pairs in {data_dir}")
        
        # Transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def _augment(self, image, mask):
        """Apply strong augmentations to image and mask for better generalization"""
        
        # Random horizontal flip (50%)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip (50%)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (70% chance, wider range)
        if random.random() > 0.3:
            angle = random.uniform(-45, 45)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Random affine transform (30% chance) - scale and translate
        if random.random() > 0.7:
            scale = random.uniform(0.8, 1.2)
            translate_x = random.uniform(-0.1, 0.1)
            translate_y = random.uniform(-0.1, 0.1)
            image = TF.affine(image, angle=0, translate=(int(translate_x * self.image_size), 
                              int(translate_y * self.image_size)), scale=scale, shear=0)
            mask = TF.affine(mask, angle=0, translate=(int(translate_x * self.image_size), 
                             int(translate_y * self.image_size)), scale=scale, shear=0)
        
        # Random color jitter (only for image, 70% chance)
        if random.random() > 0.3:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))
        
        # Random Gaussian blur (20% chance)
        if random.random() > 0.8:
            image = TF.gaussian_blur(image, kernel_size=3)
        
        return image, mask
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Apply augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': self.images[idx],
            'mask_path': self.masks[idx]
        }


class DUTSDataset(Dataset):
    """
    DUTS Dataset for salient object detection
    
    Directory structure:
        data_dir/
            DUTS-TR-Image/
                ILSVRC2012_test_00000003.jpg
                ...
            DUTS-TR-Mask/
                ILSVRC2012_test_00000003.png
                ...
    
    Args:
        data_dir: Root directory of DUTS dataset
        split: 'train' or 'test'
        image_size: Target size for images
        augment: Whether to apply data augmentation
    """
    
    def __init__(self, data_dir, split='train', image_size=320, augment=False):
        super().__init__()
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        
        if split == 'train':
            self.image_dir = os.path.join(data_dir, 'DUTS-TR-Image')
            self.mask_dir = os.path.join(data_dir, 'DUTS-TR-Mask')
        else:
            self.image_dir = os.path.join(data_dir, 'DUTS-TE-Image')
            self.mask_dir = os.path.join(data_dir, 'DUTS-TE-Mask')
        
        # Get list of images
        self.images = []
        self.masks = []
        
        if os.path.exists(self.image_dir):
            for filename in sorted(os.listdir(self.image_dir)):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(self.image_dir, filename)
                    mask_path = os.path.join(self.mask_dir, filename.replace('.jpg', '.png'))
                    
                    if os.path.exists(mask_path):
                        self.images.append(image_path)
                        self.masks.append(mask_path)
        
        print(f"Found {len(self.images)} images in DUTS {split} set")
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def _augment(self, image, mask):
        """Apply random augmentations"""
        
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
        
        return image, mask
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        if self.augment:
            image, mask = self._augment(image, mask)
        
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        image = self.normalize(image)
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': self.images[idx]
        }


class InferenceDataset(Dataset):
    """
    Dataset for inference (no masks required)
    
    Args:
        image_dir: Directory containing images
        image_size: Target size for images
    """
    
    def __init__(self, image_dir, image_size=320):
        super().__init__()
        
        self.image_dir = image_dir
        self.image_size = image_size
        
        self.images = []
        self.original_sizes = []
        
        if os.path.exists(image_dir):
            for filename in sorted(os.listdir(image_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.images.append(os.path.join(image_dir, filename))
        
        print(f"Found {len(self.images)} images for inference")
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Resize
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_tensor = TF.to_tensor(image_resized)
        image_tensor = self.normalize(image_tensor)
        
        return {
            'image': image_tensor,
            'image_path': self.images[idx],
            'original_size': original_size
        }


def get_dataloader(data_dir, batch_size=8, image_size=320, 
                   split='train', num_workers=4, dataset_type='generic'):
    """
    Create a DataLoader for segmentation training/validation
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        image_size: Target image size
        split: 'train' or 'val'/'test'
        num_workers: Number of data loading workers
        dataset_type: 'generic' or 'duts'
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    augment = (split == 'train')
    
    if dataset_type == 'duts':
        dataset = DUTSDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            augment=augment
        )
    else:
        dataset = SegmentationDataset(
            data_dir=data_dir,
            image_size=image_size,
            augment=augment
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    print("Testing SegmentationDataset...")
    
    # Create dummy data for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'images'))
        os.makedirs(os.path.join(tmpdir, 'masks'))
        
        # Create dummy images
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(os.path.join(tmpdir, 'images', f'test_{i}.jpg'))
            
            mask = Image.fromarray(np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255)
            mask.save(os.path.join(tmpdir, 'masks', f'test_{i}.png'))
        
        # Test dataset
        dataset = SegmentationDataset(tmpdir, image_size=320, augment=True)
        print(f"Dataset length: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Image path: {sample['image_path']}")

