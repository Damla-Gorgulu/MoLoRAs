"""
Similarity calculation for LoRA style matching.

This module calculates similarities between a new style image and
anchor style images using either CLIP or VGG-19, then normalizes them
to create weights for LoRA combination.
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms
from typing import List, Dict, Tuple, Any
import os


def load_image(image_path: Path) -> Image.Image:
    """Load and convert image to RGB format."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_image_from_folder(folder_path: Path) -> Image.Image:
    """Get the first image from a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = [f for f in folder_path.iterdir() 
              if f.suffix in image_extensions and f.is_file()]
    
    if not images:
        raise ValueError(f"No images found in {folder_path}")
    
    # Sort to ensure consistent ordering, take first one
    images.sort()
    return load_image(images[0])


# ImageNet normalization for VGG
VGG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def calculate_clip_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    image1: Image.Image,
    image2: Image.Image,
    device: str = "cuda"
) -> float:
    """Calculate CLIP cosine similarity between two images."""
    # Process images
    inputs = processor(images=[image1, image2], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Normalize embeddings
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            image_features[0:1], 
            image_features[1:2], 
            dim=1
        ).item()
    
    return similarity


def calculate_vgg_similarity(
    model: torch.nn.Module,
    image1: Image.Image,
    image2: Image.Image,
    device: str = "cuda"
) -> float:
    """Calculate VGG-19 cosine similarity between two images using conv features."""
    img1_tensor = VGG_TRANSFORM(image1).unsqueeze(0).to(device)
    img2_tensor = VGG_TRANSFORM(image2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat1 = model(img1_tensor)
        feat2 = model(img2_tensor)
        feat1 = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
        feat2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        similarity = F.cosine_similarity(feat1, feat2, dim=1).item()
    
    return similarity


def calculate_similarities_to_anchors(
    anchor_folders: List[Path],
    new_image_path: Path,
    model: Any,
    processor: Any,
    device: str,
    model_type: str,
) -> Dict[str, float]:
    """Calculate similarities between new image and all anchor images."""
    new_image = load_image(new_image_path)
    similarities = {}
    
    for folder_path in anchor_folders:
        folder_name = folder_path.name
        try:
            anchor_image = get_image_from_folder(folder_path)
            if model_type == "clip":
                similarity = calculate_clip_similarity(
                    model, processor, new_image, anchor_image, device
                )
            else:  # vgg
                similarity = calculate_vgg_similarity(
                    model, new_image, anchor_image, device
                )
            similarities[folder_name] = similarity
            print(f"✓ Calculated similarity for {folder_name}: {similarity:.4f}")
        except Exception as e:
            print(f"✗ Error processing {folder_name}: {e}")
            similarities[folder_name] = 0.0
    
    return similarities


def normalize_similarities(similarities: Dict[str, float]) -> Dict[str, float]:
    """Normalize similarities to sum to 1."""
    total = sum(similarities.values())
    if total == 0:
        # If all similarities are 0, return equal weights
        n = len(similarities)
        return {k: 1.0 / n for k in similarities.keys()}
    
    return {k: v / total for k, v in similarities.items()}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate style similarities using CLIP or VGG-19."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "vgg"],
        default="clip",
        help="Model to use for similarity: 'clip' or 'vgg' (default: clip)",
    )
    return parser.parse_args()


def main():
    """Main function to calculate and print similarities."""
    args = parse_args()
    model_type = args.model
    
    # Configuration
    base_dir = Path("/home/ohitit20/ip-lora/ref_images/style")
    
    # Anchor style folders (using available folders)
    # Note: watercolor and painting folders don't exist, using available ones
    anchor_folders = [
        base_dir / "cartoon",
        base_dir / "crayon",
        base_dir / "drawing1",
        base_dir / "house_3d",
        base_dir / "line",
        base_dir / "rose",
        base_dir / "deer",
        base_dir / "kiss",
        base_dir / "microscope",
        base_dir / "watercolor",
        # base_dir / "sticker",
        # base_dir / "village_oil",
    ]
    
    # New image to test (drawing2)
    new_image_path = Path('/home/ohitit20/ip-lora/ref_images/style/village_oil/1.jpg')
    
    # Check if paths exist
    if not new_image_path.exists():
        print(f"Error: New image not found at {new_image_path}")
        return
    
    missing_folders = [f for f in anchor_folders if not f.exists()]
    if missing_folders:
        print(f"Warning: Some anchor folders don't exist: {missing_folders}")
        anchor_folders = [f for f in anchor_folders if f.exists()]
    
    if not anchor_folders:
        print("Error: No anchor folders found!")
        return
    
    print(f"Anchor folders: {[f.name for f in anchor_folders]}")
    print(f"New image: {new_image_path}")
    print("-" * 60)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "CLIP" if model_type == "clip" else "VGG-19"
    print(f"Loading {model_name} model on {device}...")
    
    if model_type == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()
    else:  # vgg
        vgg_full = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        model = vgg_full.features.to(device)
        model.eval()
        processor = None
    
    print("Model loaded successfully!")
    print("-" * 60)
    
    # Calculate similarities
    print(f"Calculating {model_name} similarities...")
    similarities = calculate_similarities_to_anchors(
        anchor_folders, new_image_path, model, processor, device, model_type
    )
    
    print("-" * 60)
    print(f"\nRaw {model_name} Similarities:")
    print("-" * 60)
    for folder_name, similarity in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"{folder_name:20s}: {similarity:7.4f}")
    
    # Normalize similarities
    normalized = normalize_similarities(similarities)
    
    print("-" * 60)
    print("\nNormalized Weights (sum = 1.0):")
    print("-" * 60)
    total = 0.0
    for folder_name, weight in sorted(normalized.items(), key=lambda x: x[1], reverse=True):
        print(f"{folder_name:20s}: {weight:7.4f}")
        total += weight
    
    print("-" * 60)
    print(f"Total (should be 1.0): {total:.6f}")
    print("-" * 60)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    similarities_list = list(similarities.values())
    print(f"Max similarity: {max(similarities_list):.4f}")
    print(f"Min similarity: {min(similarities_list):.4f}")
    print(f"Mean similarity: {np.mean(similarities_list):.4f}")
    print(f"Std similarity: {np.std(similarities_list):.4f}")
    
    # Count similarities above different thresholds
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
    print("\nNumber of similarities above thresholds:")
    for threshold in thresholds:
        count = sum(1 for s in similarities_list if s >= threshold)
        print(f"  ≥ {threshold:.2f}: {count}/{len(similarities_list)}")


if __name__ == "__main__":
    main()
