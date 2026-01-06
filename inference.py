"""
Inference script for segmentation model

Supports:
- Single image or directory processing
- Patch-based inference for large images
- Overlay visualization
- Object counting

Usage:
    # Normal inference
    python inference.py --image image.jpg --checkpoint model.pth

    # Patch-based for large images
    python inference.py --image large.jpg --checkpoint model.pth --patch_size 512

    # With counting and overlay
    python inference.py --image_dir ./images --checkpoint model.pth --overlay --count
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm
import cv2

from model import get_model


class Predictor:
    """Segmentation predictor with patch-based inference and counting"""

    def __init__(self, checkpoint_path, model_name='attention_unet',
                 device='cuda', image_size=320):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # Load model
        self.model = get_model(model_name, in_channels=3, out_channels=1)
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Model: {model_name} | Device: {self.device}")
        print(f"Loaded: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """Standard prediction - resize to model size"""
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        original_size = image.size  # (W, H)
        image_resized = image.resize((self.image_size, self.image_size),
                                     Image.BILINEAR)
        image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)

        outputs = self.model(image_tensor)
        mask = outputs[0].squeeze().cpu().numpy()

        # Resize back to original
        mask_resized = cv2.resize(mask, original_size,
                                  interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > threshold).astype(np.uint8) * 255

        return mask_binary

    @torch.no_grad()
    def predict_patch(self, image, patch_size=512, overlap=64, threshold=0.5):
        """Patch-based prediction for large images."""
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_np = np.array(image)
        H, W = img_np.shape[:2]

        # Output arrays
        output_sum = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)
        weight = self._create_weight_matrix(patch_size, overlap)

        step = patch_size - overlap
        pad_h = (step - (H - patch_size) % step) % step
        pad_w = (step - (W - patch_size) % step) % step

        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)),
                           mode='reflect')

        H_pad, W_pad = img_np.shape[:2]
        positions = []

        for y in range(0, H_pad - patch_size + 1, step):
            for x in range(0, W_pad - patch_size + 1, step):
                positions.append((y, x))

        print(f"Image: {W}x{H} | Patches: {len(positions)}")

        for y, x in tqdm(positions, desc="Patches", leave=False):
            patch = img_np[y:y+patch_size, x:x+patch_size]
            patch_pil = Image.fromarray(patch)
            patch_resized = patch_pil.resize((self.image_size, self.image_size),
                                             Image.BILINEAR)
            patch_tensor = self.transform(patch_resized).unsqueeze(0).to(self.device)

            outputs = self.model(patch_tensor)
            pred = outputs[0].squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred, (patch_size, patch_size),
                                      interpolation=cv2.INTER_LINEAR)

            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            h_valid = y_end - y
            w_valid = x_end - x

            if y < H and x < W:
                output_sum[y:y_end, x:x_end] += (pred_resized[:h_valid, :w_valid]
                                                  * weight[:h_valid, :w_valid])
                weight_sum[y:y_end, x:x_end] += weight[:h_valid, :w_valid]

        mask = np.divide(output_sum, weight_sum, where=weight_sum > 0)
        mask_binary = (mask > threshold).astype(np.uint8) * 255

        return mask_binary

    def _create_weight_matrix(self, size, overlap):
        """Create weight matrix with smooth falloff at edges"""
        weight = np.ones((size, size), dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0, 1, overlap)
            for i in range(overlap):
                weight[i, :] *= ramp[i]
                weight[-(i+1), :] *= ramp[i]
                weight[:, i] *= ramp[i]
                weight[:, -(i+1)] *= ramp[i]
        return weight

    def count_objects(self, mask, min_area=100, max_area=None, morphology=True):
        """
        Count segmented objects in mask.

        Args:
            mask: Binary mask (numpy array, 0 or 255)
            min_area: Minimum area to count as object (filters noise)
            max_area: Maximum area (None = no limit)
            morphology: Apply morphological cleanup (close gaps, remove noise)

        Returns:
            dict with count, areas, centroids, bboxes
        """
        # Apply morphological operations to clean up mask
        if morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Close small gaps
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        objects = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter by area
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            objects.append({
                'id': len(objects) + 1,
                'area': int(area),
                'centroid': (float(centroids[i][0]), float(centroids[i][1])),
                'bbox': (int(x), int(y), int(w), int(h))
            })

        # Sort by area (largest first)
        objects.sort(key=lambda x: x['area'], reverse=True)

        # Update IDs after sorting
        for i, obj in enumerate(objects):
            obj['id'] = i + 1

        total_area = sum(obj['area'] for obj in objects)
        mask_area = mask.shape[0] * mask.shape[1]

        return {
            'count': len(objects),
            'total_area': total_area,
            'coverage': round(total_area / mask_area * 100, 2),
            'objects': objects
        }

    def create_overlay(self, image, mask, color=(0, 255, 0), alpha=0.5):
        """Create colored overlay visualization"""
        if isinstance(image, str):
            image = Image.open(image)

        image_np = np.array(image.convert('RGB'))
        overlay = image_np.copy()
        overlay[mask > 127] = (overlay[mask > 127] * (1 - alpha) +
                               np.array(color) * alpha).astype(np.uint8)

        return Image.fromarray(overlay)

    def create_count_overlay(self, image, mask, count_info,
                             show_bbox=True, show_id=True, show_count=True):
        """
        Create overlay with object annotations.

        Args:
            image: Input image
            mask: Binary mask
            count_info: Dict from count_objects()
            show_bbox: Draw bounding boxes
            show_id: Show object IDs
            show_count: Show total count on image

        Returns:
            Annotated PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image)

        # Create base overlay
        overlay = self.create_overlay(image, mask, color=(0, 255, 0), alpha=0.4)
        overlay_np = np.array(overlay)

        # Draw annotations
        for obj in count_info['objects']:
            x, y, w, h = obj['bbox']
            cx, cy = obj['centroid']

            if show_bbox:
                # Draw bounding box
                cv2.rectangle(overlay_np, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if show_id:
                # Draw ID at centroid
                text = str(obj['id'])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = int(cx - tw/2)
                ty = int(cy + th/2)

                # Background rectangle
                cv2.rectangle(overlay_np, (tx-2, ty-th-2), (tx+tw+2, ty+2),
                              (0, 0, 0), -1)
                cv2.putText(overlay_np, text, (tx, ty), font, font_scale,
                            (255, 255, 255), thickness)

        # Draw total count
        if show_count:
            count_text = f"Count: {count_info['count']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3

            (tw, th), _ = cv2.getTextSize(count_text, font, font_scale, thickness)

            # Background
            cv2.rectangle(overlay_np, (5, 5), (tw+15, th+15), (0, 0, 0), -1)
            cv2.putText(overlay_np, count_text, (10, th+10), font, font_scale,
                        (0, 255, 0), thickness)

        return Image.fromarray(overlay_np)


def main():
    parser = argparse.ArgumentParser(description='Segmentation Inference')

    # Input/Output
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--image_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--checkpoint', type=str, required=True)

    # Model
    parser.add_argument('--model', type=str, default='attention_unet',
                        choices=['u2net', 'u2net_small', 'attention_unet'])
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--device', type=str, default='cuda')

    # Inference
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--overlay', action='store_true', help='Save overlay')

    # Patch-based
    parser.add_argument('--patch_size', type=int, default=None)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--auto_patch', action='store_true',
                        help='Auto patch for images > 2000px')

    # Counting
    parser.add_argument('--count', action='store_true',
                        help='Count objects and save results')
    parser.add_argument('--min_area', type=int, default=100,
                        help='Min object area (pixels)')
    parser.add_argument('--max_area', type=int, default=None,
                        help='Max object area (pixels)')
    parser.add_argument('--no_morphology', action='store_true',
                        help='Disable morphological cleanup before counting')
    parser.add_argument('--show_bbox', action='store_true',
                        help='Show bounding boxes on overlay')
    parser.add_argument('--show_id', action='store_true',
                        help='Show object IDs on overlay')

    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image_dir required")

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize
    predictor = Predictor(args.checkpoint, args.model, args.device,
                          args.image_size)

    # Collect images
    if args.image:
        images = [args.image]
    else:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        images = [os.path.join(args.image_dir, f)
                  for f in sorted(os.listdir(args.image_dir))
                  if f.lower().endswith(extensions)]

    print(f"\nProcessing {len(images)} image(s)")
    if args.patch_size:
        print(f"Patch mode: {args.patch_size}x{args.patch_size}")
    if args.count:
        print(f"Counting enabled (min_area={args.min_area})")

    # Results for counting
    all_counts = {}

    for img_path in tqdm(images, desc='Processing'):
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)

        img = Image.open(img_path)
        W, H = img.size

        # Patch mode decision
        use_patch = args.patch_size is not None
        if args.auto_patch and max(W, H) > 2000:
            use_patch = True
        patch_size = args.patch_size or 512

        # Predict
        if use_patch:
            mask = predictor.predict_patch(img, patch_size, args.overlap,
                                           args.threshold)
        else:
            mask = predictor.predict(img, args.threshold)

        # Save mask
        Image.fromarray(mask).save(
            os.path.join(args.output_dir, f"{name}_mask.png"))

        # Count objects
        if args.count:
            count_info = predictor.count_objects(
                mask, args.min_area, args.max_area,
                morphology=not args.no_morphology
            )
            all_counts[filename] = {
                'count': count_info['count'],
                'total_area': count_info['total_area'],
                'coverage': count_info['coverage'],
                'objects': count_info['objects']
            }

            # Save count overlay
            if args.overlay:
                overlay = predictor.create_count_overlay(
                    img, mask, count_info,
                    show_bbox=args.show_bbox,
                    show_id=args.show_id,
                    show_count=True
                )
                overlay.save(
                    os.path.join(args.output_dir, f"{name}_overlay.png"))
        elif args.overlay:
            # Simple overlay without counting
            overlay = predictor.create_overlay(img, mask)
            overlay.save(os.path.join(args.output_dir, f"{name}_overlay.png"))

    # Save count results
    if args.count:
        # Summary
        total_objects = sum(c['count'] for c in all_counts.values())
        print(f"\n{'='*50}")
        print(f"COUNTING RESULTS")
        print(f"{'='*50}")
        print(f"{'Image':<40} {'Count':>8}")
        print(f"{'-'*50}")
        for fname, info in all_counts.items():
            print(f"{fname:<40} {info['count']:>8}")
        print(f"{'-'*50}")
        print(f"{'TOTAL':<40} {total_objects:>8}")
        print(f"{'='*50}")

        # Save JSON
        json_path = os.path.join(args.output_dir, 'counts.json')
        with open(json_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_images': len(all_counts),
                    'total_objects': total_objects
                },
                'images': all_counts
            }, f, indent=2)
        print(f"\nCount details saved to {json_path}")

    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
