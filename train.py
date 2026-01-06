"""
Training script for U²-Net with periodic result visualization

Features:
- Saves sample predictions every N epochs to train_results/
- Keeps only the last M result folders (removes oldest)
- Uses u2net_small for faster training
"""

import os
import shutil
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from model import get_model
from data import SegmentationDataset


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - focuses on hard examples"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        
        # Focal loss formula
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Dice Loss for better overlap"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss - better for imbalanced data, controls FP/FN trade-off"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives (higher = penalize missing foreground more)
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """
    Combined Loss: Focal + Dice + Tversky for best results
    - Focal: Handles class imbalance, focuses on hard examples
    - Dice: Optimizes overlap directly
    - Tversky: Controls precision/recall trade-off
    """
    def __init__(self, focal_weight=0.3, dice_weight=0.4, tversky_weight=0.3,
                 focal_alpha=0.75, focal_gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    
    def forward(self, pred, target):
        return (self.focal_weight * self.focal(pred, target) +
                self.dice_weight * self.dice(pred, target) +
                self.tversky_weight * self.tversky(pred, target))


class MultiOutputLoss(nn.Module):
    """Loss for U²-Net's multiple outputs with deep supervision"""
    def __init__(self):
        super().__init__()
        self.loss_fn = CombinedLoss()
        # Higher weight for fused output and early decoder stages
        self.weights = [2.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2]
    
    def forward(self, outputs, target):
        total_loss = 0
        for output, weight in zip(outputs, self.weights):
            total_loss += weight * self.loss_fn(output, target)
        return total_loss


def compute_metrics(pred, target, threshold=0.5):
    """Compute segmentation metrics"""
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    
    return {'dice': dice.item(), 'iou': iou.item()}


def add_border(img, color, thickness=5):
    """Add colored border to image"""
    h, w = img.shape[:2]
    bordered = img.copy()
    # Top and bottom borders
    bordered[:thickness, :] = color
    bordered[-thickness:, :] = color
    # Left and right borders
    bordered[:, :thickness] = color
    bordered[:, -thickness:] = color
    return bordered


def add_label(img, text, color=(255, 255, 255)):
    """Add text label to top of image"""
    img_with_label = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img_with_label, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(img_with_label, text, (10, text_h + 10), font, font_scale, color, thickness)
    
    return img_with_label


def save_sample_results(model, dataset, device, output_dir, num_samples=5):
    """
    Save sample prediction results as [IN with Prediction Overlay | Prediction | GT]
    - IN: Blue border - Input image with green prediction overlay
    - Prediction: Green border - Model output
    - GT: Red border - Ground truth
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Colors for borders (BGR format for cv2)
    BLUE = (255, 0, 0)    # Input
    GREEN = (0, 255, 0)   # Prediction
    RED = (0, 0, 255)     # Ground Truth
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            mask_gt = sample['mask'].numpy().squeeze()
            
            # Predict
            outputs = model(image)
            pred = outputs[0].squeeze().cpu().numpy()
            
            # Denormalize image (IN)
            img_tensor = sample['image'] * std + mean
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Convert masks to binary
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            gt_binary = (mask_gt > 0.5).astype(np.uint8) * 255
            
            # 1. IN with Prediction Overlay (green overlay on input)
            in_with_overlay = img_np.copy()
            # Add green overlay where prediction is positive
            overlay_mask = pred_binary > 127
            in_with_overlay[overlay_mask] = (
                in_with_overlay[overlay_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            ).astype(np.uint8)
            # Draw prediction contours
            contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(in_with_overlay, contours, -1, GREEN, 2)
            # Add blue border and label
            in_with_overlay = add_border(in_with_overlay, BLUE, thickness=5)
            in_with_overlay = add_label(in_with_overlay, "IN + Prediction", BLUE)
            
            # 2. Prediction output (green border)
            pred_rgb = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2RGB)
            pred_rgb = add_border(pred_rgb, GREEN, thickness=5)
            pred_rgb = add_label(pred_rgb, "Prediction", GREEN)
            
            # 3. GT (red border)
            gt_rgb = cv2.cvtColor(gt_binary, cv2.COLOR_GRAY2RGB)
            gt_rgb = add_border(gt_rgb, RED, thickness=5)
            gt_rgb = add_label(gt_rgb, "Ground Truth", RED)
            
            # Create comparison: [IN+Overlay | Prediction | GT]
            comparison = np.hstack([in_with_overlay, pred_rgb, gt_rgb])
            
            # Save
            Image.fromarray(comparison).save(os.path.join(output_dir, f'sample_{i+1}.png'))
    
    model.train()


def manage_result_folders(results_base_dir, max_folders=5):
    """Keep only the last N result folders, remove oldest"""
    if not os.path.exists(results_base_dir):
        return
    
    # Get all epoch folders
    folders = []
    for f in os.listdir(results_base_dir):
        folder_path = os.path.join(results_base_dir, f)
        if os.path.isdir(folder_path) and f.startswith('epoch_'):
            try:
                epoch_num = int(f.split('_')[1])
                folders.append((epoch_num, folder_path))
            except:
                pass
    
    # Sort by epoch number
    folders.sort(key=lambda x: x[0])
    
    # Remove oldest folders if more than max_folders
    while len(folders) > max_folders:
        _, oldest_folder = folders.pop(0)
        print(f"Removing old results folder: {oldest_folder}")
        shutil.rmtree(oldest_folder)


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        with torch.no_grad():
            metrics = compute_metrics(outputs[0], masks)
        
        total_loss += loss.item()
        total_dice += metrics['dice']
        num_batches += 1
        
        # Show learning rate in progress bar
        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{metrics["dice"]:.4f}', 'lr': f'{lr:.2e}'})
    
    return total_loss / num_batches, total_dice / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            metrics = compute_metrics(outputs[0], masks)
            
            total_loss += loss.item()
            total_dice += metrics['dice']
            num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train U²-Net with result visualization')
    
    parser.add_argument('--data_dir', type=str, default='WebknossImages')
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--val_split', type=float, default=0.15)
    
    parser.add_argument('--model', type=str, default='attention_unet', 
                        choices=['u2net', 'u2net_small', 'attention_unet'])
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Result saving options
    parser.add_argument('--results_dir', type=str, default='train_results',
                        help='Directory to save training results')
    parser.add_argument('--save_every', type=int, default=2,
                        help='Save results every N epochs')
    parser.add_argument('--max_result_folders', type=int, default=5,
                        help='Maximum number of result folders to keep')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample images to save')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_small')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Dataset
    print(f"\nLoading dataset from {args.data_dir}")
    full_dataset = SegmentationDataset(
        args.data_dir,
        image_size=args.image_size,
        augment=True
    )
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    
    # Model
    print(f"\nCreating {args.model} model")
    model = get_model(args.model, in_channels=3, out_channels=1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = MultiOutputLoss()
    
    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    best_dice = 0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nLoading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'dice' in checkpoint:
                best_dice = checkpoint['dice']
            print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs (from epoch {start_epoch + 1})")
    print(f"Results saved every {args.save_every} epochs to {args.results_dir}/")
    print(f"Keeping last {args.max_result_folders} result folders")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save results every N epochs
        if (epoch + 1) % args.save_every == 0:
            result_folder = os.path.join(args.results_dir, f'epoch_{epoch + 1:03d}')
            print(f"\nSaving sample results to {result_folder}")
            
            # Use the underlying dataset (not the Subset)
            save_sample_results(model, full_dataset, device, result_folder, args.num_samples)
            
            # Manage result folders (keep only last N)
            manage_result_folders(args.results_dir, args.max_result_folders)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_dice
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"✓ Saved best model with dice: {best_dice:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': val_dice
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'dice': val_dice
    }, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()

