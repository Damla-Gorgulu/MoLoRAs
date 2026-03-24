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
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from torchvision import models, transforms
from typing import List, Dict, Tuple, Any
import os
import lpips


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

# LPIPS expects images in [-1, 1] range
LPIPS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0, 1] -> [-1, 1]
])


def patch_and_shuffle_image(
    image: Image.Image,
    patch_size: int = 32,
) -> Image.Image:
    """
    Split the image into non-overlapping patches and shuffle them.

    This operation keeps local color/texture statistics (style)
    while destroying global spatial structure (content).
    """
    width, height = image.size

    # Ensure we have at least one full patch in each dimension
    patched_width = (width // patch_size) * patch_size
    patched_height = (height // patch_size) * patch_size
    if patched_width == 0 or patched_height == 0:
        return image

    # Work on a cropped region that is divisible by patch_size
    image_cropped = image.crop((0, 0, patched_width, patched_height))
    arr = np.array(image_cropped)

    patches = []
    for y in range(0, patched_height, patch_size):
        for x in range(0, patched_width, patch_size):
            patches.append(arr[y : y + patch_size, x : x + patch_size, :])

    patches = np.array(patches)
    # Shuffle patches in-place
    np.random.shuffle(patches)

    # Reconstruct image from shuffled patches
    shuffled = np.zeros_like(arr)
    idx = 0
    for y in range(0, patched_height, patch_size):
        for x in range(0, patched_width, patch_size):
            shuffled[y : y + patch_size, x : x + patch_size, :] = patches[idx]
            idx += 1

    return Image.fromarray(shuffled)


def apply_style_transformations(
    image: Image.Image,
    patch_and_shuffle: bool = False,
    grayscale: bool = False,
) -> Image.Image:
    """
    Apply optional style-preserving, content-destroying transformations.

    - patch_and_shuffle: break image into patches and shuffle them
    - grayscale: convert to grayscale and back to RGB (for 3-channel models)
    """
    if patch_and_shuffle:
        image = patch_and_shuffle_image(image)

    if grayscale:
        # Convert to single channel then back to RGB so downstream
        # models that expect 3 channels still work.
        image = image.convert("L").convert("RGB")

    return image


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


def calculate_lpips_similarity(
    model: torch.nn.Module,
    image1: Image.Image,
    image2: Image.Image,
    device: str = "cuda"
) -> float:
    """
    Calculate LPIPS perceptual similarity between two images.
    Note: LPIPS returns distance (lower is more similar), so we return 1 - distance.
    """
    img1_tensor = LPIPS_TRANSFORM(image1).unsqueeze(0).to(device)
    img2_tensor = LPIPS_TRANSFORM(image2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # LPIPS returns distance (0 = identical, higher = more different)
        distance = model(img1_tensor, img2_tensor).item()
        # Convert to similarity (1 = identical, 0 = very different)
        similarity = 1.0 / (1.0 + distance)
    
    return similarity


def calculate_dinov2_similarity(
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    image1: Image.Image,
    image2: Image.Image,
    device: str = "cuda"
) -> float:
    """Calculate DINOv2 cosine similarity between two images using [CLS] token embeddings."""
    # Process images
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}
    
    with torch.no_grad():
        # Get [CLS] token embeddings (first token)
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        
        # Use pooler_output or last_hidden_state[:, 0]
        if hasattr(outputs1, 'pooler_output') and outputs1.pooler_output is not None:
            feat1 = outputs1.pooler_output
            feat2 = outputs2.pooler_output
        else:
            feat1 = outputs1.last_hidden_state[:, 0]
            feat2 = outputs2.last_hidden_state[:, 0]
        
        # Normalize embeddings
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(feat1, feat2, dim=1).item()
    
    return similarity


def calculate_similarities_to_anchors(
    anchor_folders: List[Path],
    new_image_path: Path,
    model: Any,
    processor: Any,
    device: str,
    model_type: str,
    patch_and_shuffle: bool = False,
    grayscale: bool = False,
) -> Dict[str, float]:
    """Calculate similarities between new image and all anchor images."""
    new_image = load_image(new_image_path)
    if patch_and_shuffle or grayscale:
        new_image = apply_style_transformations(
            new_image,
            patch_and_shuffle=patch_and_shuffle,
            grayscale=grayscale,
        )
    similarities = {}
    
    for folder_path in anchor_folders:
        folder_name = folder_path.name
        try:
            anchor_image = get_image_from_folder(folder_path)
            if patch_and_shuffle or grayscale:
                anchor_image = apply_style_transformations(
                    anchor_image,
                    patch_and_shuffle=patch_and_shuffle,
                    grayscale=grayscale,
                )
            if model_type == "clip":
                similarity = calculate_clip_similarity(
                    model, processor, new_image, anchor_image, device
                )
            elif model_type == "vgg":
                similarity = calculate_vgg_similarity(
                    model, new_image, anchor_image, device
                )
            elif model_type == "lpips":
                similarity = calculate_lpips_similarity(
                    model, new_image, anchor_image, device
                )
            elif model_type == "dinov2" or model_type == "dinov3":
                similarity = calculate_dinov2_similarity(
                    model, processor, new_image, anchor_image, device
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
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
        description="Calculate style similarities using CLIP, VGG-19, LPIPS, DINOv2, or DINOv3."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "vgg", "lpips", "dinov2", "dinov3"],
        default="clip",
        help="Model to use for similarity: 'clip', 'vgg', 'lpips', 'dinov2', or 'dinov3' (default: clip)",
    )
    parser.add_argument(
        "--patch-and-shuffle",
        action="store_true",
        help=(
            "If set, split images into small patches and shuffle them, "
            "destroying content while keeping local style statistics."
        ),
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="If set, convert all images to grayscale before computing similarities.",
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
    
    # Model name mapping
    model_names = {
        "clip": "CLIP",
        "vgg": "VGG-19",
        "lpips": "LPIPS",
        "dinov2": "DINOv2",
        "dinov3": "DINOv3"
    }
    model_name = model_names.get(model_type, model_type.upper())
    print(f"Loading {model_name} model on {device}...")
    
    if model_type == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()
    elif model_type == "vgg":
        vgg_full = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        model = vgg_full.features.to(device)
        model.eval()
        processor = None
    elif model_type == "lpips":
        model = lpips.LPIPS(net='alex').to(device)
        model.eval()
        processor = None
    elif model_type == "dinov2":
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model.eval()
    elif model_type == "dinov3":
        # DINOv3 uses DINOv2-large for better performance
        model = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m').to(device)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("Model loaded successfully!")
    print("-" * 60)
    
    # Calculate similarities
    print(f"Calculating {model_name} similarities...")
    similarities = calculate_similarities_to_anchors(
        anchor_folders,
        new_image_path,
        model,
        processor,
        device,
        model_type,
        patch_and_shuffle=args.patch_and_shuffle,
        grayscale=args.grayscale,
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
