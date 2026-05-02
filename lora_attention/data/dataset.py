"""
Datasets for Stage 1 and Stage 2 training of MoELoRA.

v1.0 Datasets (kept for backward compatibility):
  - Stage1Dataset:  GT LoRA in pool, one-hot target, MSE loss.
  - Stage2Dataset:  GT excluded, LDM loss.

v2.0 Datasets:
  - WikiArtStage1Dataset:  WikiArt images, soft CLIP-similarity targets, KL loss.
  - WikiArtStage2Dataset:  WikiArt images, GT excluded, LDM + entropy reg.

Variable-N batching:
  Since N varies per sample, use the provided collate_fn which returns
  lists (not padded tensors) for pool-level data.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from ..models.lora_pool import LoRAPool


def _readable_style_name(raw: str) -> str:
    """Convert pool key like 'style_0022_Post_Impressionism' → 'Post Impressionism'."""
    # Strip leading 'style_NNNN_' prefix if present
    cleaned = re.sub(r'^style_\d+_', '', raw)
    return cleaned.replace('_', ' ')


def _sample_pool_indices(
    gt_idx: int,
    n: int,
    num_experts: int,
    exclude_gt: bool,
    rng: random.Random,
) -> List[int]:
    """
    Lightweight pool sampler — uses only num_experts, not the full LoRAPool.
    Avoids pickling the pool (which would fork 3 GB of raw tensors into workers).
    """
    all_ids = list(range(num_experts))
    if exclude_gt:
        candidates = [i for i in all_ids if i != gt_idx]
        n = min(n, len(candidates))
        return rng.sample(candidates, n)
    else:
        candidates = [i for i in all_ids if i != gt_idx]
        n = min(n, num_experts)
        others = rng.sample(candidates, n - 1)
        pool = others + [gt_idx]
        rng.shuffle(pool)
        return pool


def _exact_style_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Style-preserving exact-exemplar augmentation."""
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)

    angle = rng.uniform(-5.0, 5.0)
    if abs(angle) > 0.25:
        fill = img.getpixel((0, 0))
        img = img.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=fill,
        )

    img = ImageEnhance.Brightness(img).enhance(1.0 + rng.uniform(-0.05, 0.05))
    img = ImageEnhance.Contrast(img).enhance(1.0 + rng.uniform(-0.05, 0.05))
    img = ImageEnhance.Color(img).enhance(1.0 + rng.uniform(-0.03, 0.03))
    img = ImageEnhance.Sharpness(img).enhance(1.0 + rng.uniform(-0.03, 0.03))
    return img


# ──────────────────────────────────────────────────────────────
# Image discovery helpers
# ──────────────────────────────────────────────────────────────
_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG"}


def _find_images_for_style(style_name: str, image_dirs: List[Path]) -> List[Path]:
    """
    Return all image paths matching style_name across given image_dirs.
    Searches for files whose stem starts with or equals style_name.
    Falls back to all images in a sub-directory named style_name.
    """
    found: List[Path] = []
    for d in image_dirs:
        if not d.exists():
            continue
        # Sub-directory named after style
        sub = d / style_name
        if sub.is_dir():
            found += [p for p in sub.iterdir() if p.suffix in _IMG_EXTENSIONS]
        # Files in directory with matching stem prefix
        for p in d.iterdir():
            if p.suffix in _IMG_EXTENSIONS and (
                p.stem == style_name or p.stem.startswith(style_name)
            ):
                found.append(p)
    return list(set(found))


# ──────────────────────────────────────────────────────────────
# Base dataset
# ──────────────────────────────────────────────────────────────
class _LoRAAttentionDatasetBase(Dataset):
    """
    Shared base for Stage1Dataset and Stage2Dataset.

    Args:
        pool:           LoRAPool — provides features and tensor data.
        image_dirs:     Directories to search for style images.
                        Defaults to blora_zoo/style_images relative to pool.
        min_pool_size:  Minimum number of experts per sample (≥ 1 for Stage 2).
        max_pool_size:  Maximum number of experts per sample.
        rank:           LoRA rank (used to build target tensors). Default 64.
        seed:           Optional random seed.
        image_transform: Optional callable applied to PIL images before return.
    """

    def __init__(
        self,
        pool: LoRAPool,
        image_dirs: Optional[List[str]] = None,
        min_pool_size: int = 3,
        max_pool_size: int = 20,
        rank: int = 64,
        seed: Optional[int] = None,
        image_transform=None,
    ):
        self.pool = pool
        self.rank = rank
        self.min_pool_size = min_pool_size
        self.max_pool_size = min(max_pool_size, pool.num_experts)
        self.image_transform = image_transform

        # Resolve image directories
        if image_dirs is None:
            zoo_root = Path(pool.zoo_dir).parent
            image_dirs = [str(zoo_root / "style_images")]
        self.image_dirs = [Path(d) for d in image_dirs]

        # Build index: list of (style_idx, image_path)
        self.samples: List[Tuple[int, Path]] = []
        for idx, name in enumerate(pool.style_names):
            imgs = _find_images_for_style(name, self.image_dirs)
            if not imgs:
                # Fall back: try bloras/<name>/ for any image
                bloras_dir = Path(pool.zoo_dir) / name
                if bloras_dir.is_dir():
                    imgs = [
                        p for p in bloras_dir.iterdir()
                        if p.suffix in _IMG_EXTENSIONS
                    ]
            for img_path in imgs:
                self.samples.append((idx, img_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No style images found. Check image_dirs and zoo_dir paths.\n"
                f"  image_dirs: {self.image_dirs}\n"
                f"  zoo_dir:    {pool.zoo_dir}"
            )

        self.rng = random.Random(seed)
        print(
            f"[Dataset] {self.__class__.__name__}: "
            f"{len(self.samples)} samples, "
            f"pool_size ∈ [{self.min_pool_size}, {self.max_pool_size}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img

    def _sample_n(self) -> int:
        """Sample pool size N ∈ [min, max] uniformly."""
        return self.rng.randint(self.min_pool_size, self.max_pool_size)

    def _gt_target(self, gt_pos: int, n: int) -> torch.Tensor:
        """
        Build Stage-1 target attention matrix.
        Shape: (N, rank) — row gt_pos is all 1s, others all 0s.
        """
        target = torch.zeros(n, self.rank)
        target[gt_pos, :] = 1.0
        return target

    # ── To be implemented by subclasses ──────────────────────
    def __getitem__(self, idx: int):
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# Stage 1 Dataset
# ──────────────────────────────────────────────────────────────
class Stage1Dataset(_LoRAAttentionDatasetBase):
    """
    Stage 1: GT LoRA is included in the pool.

    Returns per sample:
        image        : PIL.Image  (RGB)
        gt_idx       : int        index into LoRAPool
        pool_indices : List[int]  N expert indices (GT is somewhere in here)
        gt_pos       : int        position of GT within pool_indices
        target       : Tensor(N, rank)  one-hot target for the attention matrix
    """

    def __getitem__(self, idx: int) -> Dict:
        gt_idx, img_path = self.samples[idx]
        image = self._load_image(img_path)

        n = self._sample_n()
        # Sample pool with GT included
        pool_indices = self.pool.sample_pool(
            gt_idx=gt_idx,
            n=n,
            exclude_gt=False,
        )
        gt_pos = pool_indices.index(gt_idx)
        target = self._gt_target(gt_pos=gt_pos, n=len(pool_indices))

        return {
            "image": image,
            "gt_idx": gt_idx,
            "pool_indices": pool_indices,
            "gt_pos": gt_pos,
            "target": target,           # (N, rank)
        }


# ──────────────────────────────────────────────────────────────
# Stage 2 Dataset
# ──────────────────────────────────────────────────────────────
class Stage2Dataset(_LoRAAttentionDatasetBase):
    """
    Stage 2: GT LoRA is EXCLUDED from the pool.

    The model must reconstruct the query style from combinations of
    other experts. Loss is the LDM noise prediction loss (not MSE on A).

    Returns per sample:
        image        : PIL.Image  (RGB)
        prompt       : str        text prompt (loaded from prompt.txt if available,
                                  else default "A {style_name} artwork")
        gt_idx       : int
        pool_indices : List[int]  N indices, none equal to gt_idx
    """

    def __init__(self, *args, prompts_file: Optional[str] = None, **kwargs):
        # Stage 2 minimum pool size must be ≥ 2 (need at least 2 non-GT experts)
        if "min_pool_size" not in kwargs:
            kwargs["min_pool_size"] = 3
        super().__init__(*args, **kwargs)

        # Load prompts if provided
        self._prompts: Optional[List[str]] = None
        if prompts_file is not None:
            import json
            with open(prompts_file) as f:
                self._prompts = json.load(f)

    def __getitem__(self, idx: int) -> Dict:
        gt_idx, img_path = self.samples[idx]
        image = self._load_image(img_path)
        style_name = self.pool.style_names[gt_idx]

        n = self._sample_n()
        # Sample pool WITHOUT GT
        pool_indices = self.pool.sample_pool(
            gt_idx=gt_idx,
            n=n,
            exclude_gt=True,
        )

        # Build prompt — use human-readable style name, no activation tokens
        readable = _readable_style_name(style_name)
        if self._prompts is not None:
            prompt = self.rng.choice(self._prompts)
            prompt = f"{prompt} in {readable} style"
        else:
            prompt = f"A {readable} artwork"

        return {
            "image": image,
            "prompt": prompt,
            "gt_idx": gt_idx,
            "pool_indices": pool_indices,
        }


# ──────────────────────────────────────────────────────────────
# Collate functions for DataLoader
# ──────────────────────────────────────────────────────────────
def stage1_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate Stage1 samples. Returns lists where per-sample N varies.

    Returned dict keys:
        images        : List[PIL.Image]
        gt_indices    : List[int]
        pool_indices  : List[List[int]]
        gt_positions  : List[int]
        targets       : List[Tensor(N_i, rank)]
    """
    return {
        "images": [s["image"] for s in batch],
        "gt_indices": [s["gt_idx"] for s in batch],
        "pool_indices": [s["pool_indices"] for s in batch],
        "gt_positions": [s["gt_pos"] for s in batch],
        "targets": [s["target"] for s in batch],
    }


def stage2_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate Stage2 samples.

    Returned dict keys:
        images        : List[PIL.Image]
        prompts       : List[str]
        gt_indices    : List[int]
        pool_indices  : List[List[int]]
    """
    return {
        "images": [s["image"] for s in batch],
        "prompts": [s["prompt"] for s in batch],
        "gt_indices": [s["gt_idx"] for s in batch],
        "pool_indices": [s["pool_indices"] for s in batch],
    }


# ══════════════════════════════════════════════════════════════
# v2.0 Datasets: WikiArt + Soft Targets
# ══════════════════════════════════════════════════════════════

class WikiArtStage1Dataset(Dataset):
    """
    v2.0 Stage 1: WikiArt images with soft CLIP-similarity targets.

    Each sample:
        image        : PIL.Image from WikiArt
        gt_idx       : int — pool expert index for this style category
        pool_indices : List[int] — N experts (GT included at random position)
        gt_pos       : int — position of GT within pool_indices
        soft_target  : Tensor(N,) — soft distribution from CLIP similarity
        style_name   : str — WikiArt category name

    The soft target is broadcast to (N, r) during training by repeating
    across rank positions — each rank gets the same expert-level target.

    Args:
        pool:                LoRAPool instance.
        wikiart_dir:         Path to WikiArt dataset root.
        label_map_path:      Path to wikiart_label_map.json.
        similarity_path:     Path to clip_similarity.pt (109×109 matrix).
        tau_label:           Soft target temperature. Default 0.3.
        min_pool_size:       Minimum pool size N. Default 5.
        max_pool_size:       Maximum pool size N. Default 20.
        max_images_per_style: Cap images per style to balance dataset. Default 500.
        rank:                LoRA rank. Default 64.
        seed:                Random seed.
        image_transform:     Optional transform applied to PIL images.
    """

    def __init__(
        self,
        pool: LoRAPool,
        wikiart_dir: str,
        label_map_path: str,
        similarity_path: Optional[str] = None,
        tau_label: float = 0.3,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
        max_images_per_style: int = 500,
        rank: int = 64,
        seed: Optional[int] = None,
        image_transform=None,
    ):
        # Store only num_experts from pool — avoids forking 3 GB of raw tensors
        # into DataLoader workers when pool._style_tensors is large.
        self._num_experts = pool.num_experts
        self.rank = rank
        self.tau_label = tau_label
        self.min_pool_size = min_pool_size
        self.max_pool_size = min(max_pool_size, pool.num_experts)
        self.image_transform = image_transform

        # ── Load label map ────────────────────────────────────
        with open(label_map_path) as f:
            self.label_map: Dict[str, List[int]] = json.load(f)

        # ── Load similarity matrix (optional — only needed for KL mode) ──
        if similarity_path is not None:
            sim_data = torch.load(similarity_path, map_location="cpu", weights_only=False)
            self.similarity_matrix: Optional[torch.Tensor] = sim_data["similarity_matrix"]
        else:
            self.similarity_matrix = None

        # ── Build samples: (expert_idx, image_path) ──────────
        wikiart_path = Path(wikiart_dir)
        self.samples: List[Tuple[int, Path, str]] = []  # (expert_idx, img_path, category)

        rng = random.Random(seed)
        for category, expert_indices in self.label_map.items():
            cat_dir = wikiart_path / category
            if not cat_dir.is_dir():
                continue

            # Find all images in this category
            imgs = sorted([
                p for p in cat_dir.iterdir()
                if p.suffix in _IMG_EXTENSIONS
            ])
            if not imgs:
                continue

            # Cap images per expert to balance dataset
            if len(imgs) > max_images_per_style * len(expert_indices):
                imgs = rng.sample(imgs, max_images_per_style * len(expert_indices))

            # Assign images round-robin to experts in this category
            for i, img_path in enumerate(imgs):
                expert_idx = expert_indices[i % len(expert_indices)]
                self.samples.append((expert_idx, img_path, category))

        if not self.samples:
            raise RuntimeError(
                "No WikiArt samples found. Check wikiart_dir and label_map.\n"
                f"  wikiart_dir: {wikiart_dir}\n"
                f"  label_map:   {label_map_path}\n"
                f"  categories:  {list(self.label_map.keys())[:5]}..."
            )

        self.rng = random.Random(seed)
        target_mode = "KL (CLIP-similarity)" if similarity_path is not None else "one-hot CE"
        print(
            f"[WikiArtStage1Dataset] {len(self.samples)} samples, "
            f"{len(self.label_map)} categories, "
            f"target={target_mode}, τ_label={tau_label}, pool ∈ [{min_pool_size}, {self.max_pool_size}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        gt_idx, img_path, category = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        # Sample pool (GT included)
        n = self.rng.randint(self.min_pool_size, self.max_pool_size)
        pool_indices = _sample_pool_indices(
            gt_idx, n, self._num_experts, exclude_gt=False, rng=self.rng
        )
        gt_pos = pool_indices.index(gt_idx)

        # Soft or one-hot target
        if self.similarity_matrix is not None:
            # KL mode: CLIP-similarity soft distribution
            sims = self.similarity_matrix[gt_idx, pool_indices]  # (N,)
            soft_target = torch.softmax(sims / self.tau_label, dim=0)  # (N,)
        else:
            # CE mode: one-hot at gt_pos (used as reference; training reads gt_pos directly)
            soft_target = torch.zeros(len(pool_indices))
            soft_target[gt_pos] = 1.0

        return {
            "image": img,
            "gt_idx": gt_idx,
            "pool_indices": pool_indices,
            "gt_pos": gt_pos,
            "soft_target": soft_target,  # (N,)
            "style_name": category,
        }


class WikiArtStage2Dataset(Dataset):
    """
    v2.0 Stage 2: WikiArt images, GT excluded from pool, LDM loss.

    Same as WikiArtStage1Dataset but:
      - GT expert is EXCLUDED from pool (model must compose from others).
      - Returns a text prompt instead of target.
      - Used with LDM noise prediction + entropy regularisation.
    """

    def __init__(
        self,
        pool: LoRAPool,
        wikiart_dir: str,
        label_map_path: str,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
        max_images_per_style: int = 500,
        rank: int = 64,
        seed: Optional[int] = None,
        image_transform=None,
        prompts_file: Optional[str] = None,
    ):
        # Store only num_experts — avoids forking 3 GB of raw tensors into DataLoader workers
        self._num_experts = pool.num_experts
        self.rank = rank
        self.min_pool_size = min_pool_size
        self.max_pool_size = min(max_pool_size, pool.num_experts)
        self.image_transform = image_transform

        # ── Load label map ────────────────────────────────────
        with open(label_map_path) as f:
            self.label_map: Dict[str, List[int]] = json.load(f)

        # ── Load prompts ──────────────────────────────────────
        self._prompts: Optional[List[str]] = None
        if prompts_file is not None:
            with open(prompts_file) as f:
                self._prompts = json.load(f)

        # ── Build samples ─────────────────────────────────────
        wikiart_path = Path(wikiart_dir)
        self.samples: List[Tuple[int, Path, str]] = []

        rng = random.Random(seed)
        for category, expert_indices in self.label_map.items():
            cat_dir = wikiart_path / category
            if not cat_dir.is_dir():
                continue

            imgs = sorted([
                p for p in cat_dir.iterdir()
                if p.suffix in _IMG_EXTENSIONS
            ])
            if not imgs:
                continue

            if len(imgs) > max_images_per_style * len(expert_indices):
                imgs = rng.sample(imgs, max_images_per_style * len(expert_indices))

            for i, img_path in enumerate(imgs):
                expert_idx = expert_indices[i % len(expert_indices)]
                self.samples.append((expert_idx, img_path, category))

        if not self.samples:
            raise RuntimeError(
                "No WikiArt samples found for Stage 2."
            )

        self.rng = random.Random(seed)
        print(
            f"[WikiArtStage2Dataset] {len(self.samples)} samples, "
            f"{len(self.label_map)} categories, "
            f"pool ∈ [{min_pool_size}, {self.max_pool_size}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        gt_idx, img_path, category = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        # Sample pool WITHOUT GT
        n = self.rng.randint(self.min_pool_size, self.max_pool_size)
        pool_indices = _sample_pool_indices(
            gt_idx, n, self._num_experts, exclude_gt=True, rng=self.rng
        )

        # Build prompt — use human-readable category name, no activation tokens
        style_name = category.replace("_", " ")
        if self._prompts is not None:
            prompt = self.rng.choice(self._prompts)
            prompt = f"{prompt} in {style_name} style"
        else:
            prompt = f"A painting in {style_name} style"

        return {
            "image": img,
            "prompt": prompt,
            "gt_idx": gt_idx,
            "pool_indices": pool_indices,
            "style_name": category,
        }


class ExactExemplarStage1Dataset(Dataset):
    """
    Stage 1 exact-instance retrieval dataset.

    Each sample uses the exact B-LoRA exemplar image as the query source, with
    style-preserving augmentations applied on the fly.
    """

    def __init__(
        self,
        pool: LoRAPool,
        manifest_path: str,
        min_pool_size: int = 4,
        max_pool_size: int = 4,
        views_per_style: int = 32,
        seed: Optional[int] = None,
        deterministic_augment: bool = False,
        image_transform=None,
    ):
        self.pool = pool
        self.rank = 64
        self.min_pool_size = min_pool_size
        self.max_pool_size = min(max_pool_size, pool.num_experts)
        self.image_transform = image_transform
        self.deterministic_augment = deterministic_augment
        self.seed = seed or 0

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.entries: List[Dict] = manifest["styles"]
        self.samples: List[Tuple[int, Path, str]] = []

        for entry in self.entries:
            expert_name = entry["expert_name"]
            source_image = Path(entry["source_image"])
            gt_idx = pool.index_of(expert_name)
            for _ in range(views_per_style):
                self.samples.append((gt_idx, source_image, expert_name))

        if not self.samples:
            raise RuntimeError(f"No exact exemplar samples found in {manifest_path}")

        self.rng = random.Random(seed)
        print(
            f"[ExactExemplarStage1Dataset] {len(self.samples)} samples, "
            f"{len(self.entries)} exemplars, pool ∈ [{self.min_pool_size}, {self.max_pool_size}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image, idx: int) -> Image.Image:
        if self.deterministic_augment:
            rng = random.Random(self.seed + idx * 101)
        else:
            # Keep the augmentation stochastic across epochs while still being
            # reproducible enough to debug the data path.
            rng = self.rng
        return _exact_style_augment(img, rng)

    def __getitem__(self, idx: int) -> Dict:
        gt_idx, img_path, style_name = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self._augment(img, idx)
        if self.image_transform is not None:
            img = self.image_transform(img)

        n = self.rng.randint(self.min_pool_size, self.max_pool_size)
        pool_indices = _sample_pool_indices(
            gt_idx, n, self.pool.num_experts, exclude_gt=False, rng=self.rng
        )
        gt_pos = pool_indices.index(gt_idx)
        soft_target = torch.zeros(len(pool_indices))
        soft_target[gt_pos] = 1.0

        return {
            "image": img,
            "gt_idx": gt_idx,
            "pool_indices": pool_indices,
            "gt_pos": gt_pos,
            "soft_target": soft_target,
            "style_name": style_name,
        }


class ExactExemplarStage2Dataset(Dataset):
    """
    Stage 2 exact-instance dataset.

    Uses the same exact exemplar source images as Stage 1, but excludes the GT
    expert from the pool so Stage 2 can test whether composition over the other
    experts can still reconstruct a useful style signal.
    """

    def __init__(
        self,
        pool: LoRAPool,
        manifest_path: str,
        min_pool_size: int = 3,
        max_pool_size: int = 4,
        views_per_style: int = 32,
        seed: Optional[int] = None,
        deterministic_augment: bool = False,
        image_transform=None,
        prompt_mode: str = "neutral",
    ):
        self.pool = pool
        self.min_pool_size = min_pool_size
        self.max_pool_size = min(max_pool_size, max(pool.num_experts - 1, 1))
        self.image_transform = image_transform
        self.deterministic_augment = deterministic_augment
        self.seed = seed or 0
        self.prompt_mode = prompt_mode

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.entries: List[Dict] = manifest["styles"]
        self.samples: List[Tuple[int, Path, str, str]] = []

        for entry in self.entries:
            expert_name = entry["expert_name"]
            source_image = Path(entry["source_image"])
            gt_idx = pool.index_of(expert_name)
            category = entry.get("category", expert_name)
            for _ in range(views_per_style):
                self.samples.append((gt_idx, source_image, expert_name, category))

        if not self.samples:
            raise RuntimeError(f"No exact exemplar Stage 2 samples found in {manifest_path}")

        self.rng = random.Random(seed)
        print(
            f"[ExactExemplarStage2Dataset] {len(self.samples)} samples, "
            f"{len(self.entries)} exemplars, pool ∈ [{self.min_pool_size}, {self.max_pool_size}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image, idx: int) -> Image.Image:
        if self.deterministic_augment:
            rng = random.Random(self.seed + idx * 101)
        else:
            rng = self.rng
        return _exact_style_augment(img, rng)

    def _build_prompt(self, category: str) -> str:
        readable = category.replace("_", " ")
        if self.prompt_mode == "style":
            return f"A painting in {readable} style"
        if self.prompt_mode == "minimal":
            return "A painting"
        return "A detailed painting"

    def __getitem__(self, idx: int) -> Dict:
        gt_idx, img_path, style_name, category = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self._augment(img, idx)
        if self.image_transform is not None:
            img = self.image_transform(img)

        n = self.rng.randint(self.min_pool_size, self.max_pool_size)
        pool_indices = _sample_pool_indices(
            gt_idx, n, self.pool.num_experts, exclude_gt=True, rng=self.rng
        )
        prompt = self._build_prompt(category)

        return {
            "image": img,
            "prompt": prompt,
            "gt_idx": gt_idx,
            "pool_indices": pool_indices,
            "style_name": style_name,
        }


# ──────────────────────────────────────────────────────────────
# v2.0 collate functions
# ──────────────────────────────────────────────────────────────
def wikiart_stage1_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate WikiArtStage1 samples. Variable N per sample.

    Returned dict keys:
        images        : List[PIL.Image]
        gt_indices    : List[int]
        pool_indices  : List[List[int]]
        gt_positions  : List[int]
        soft_targets  : List[Tensor(N_i,)]
        style_names   : List[str]
    """
    return {
        "images": [s["image"] for s in batch],
        "gt_indices": [s["gt_idx"] for s in batch],
        "pool_indices": [s["pool_indices"] for s in batch],
        "gt_positions": [s["gt_pos"] for s in batch],
        "soft_targets": [s["soft_target"] for s in batch],
        "style_names": [s["style_name"] for s in batch],
    }


def exact_stage1_collate_fn(batch: List[Dict]) -> Dict:
    """Collate exact-exemplar Stage 1 samples."""
    return {
        "images": [s["image"] for s in batch],
        "gt_indices": [s["gt_idx"] for s in batch],
        "pool_indices": [s["pool_indices"] for s in batch],
        "gt_positions": [s["gt_pos"] for s in batch],
        "soft_targets": [s["soft_target"] for s in batch],
        "style_names": [s["style_name"] for s in batch],
    }


def exact_stage2_collate_fn(batch: List[Dict]) -> Dict:
    """Collate exact-exemplar Stage 2 samples."""
    return {
        "images": [s["image"] for s in batch],
        "prompts": [s["prompt"] for s in batch],
        "gt_indices": [s["gt_idx"] for s in batch],
        "pool_indices": [s["pool_indices"] for s in batch],
        "style_names": [s["style_name"] for s in batch],
    }


def wikiart_stage2_collate_fn(batch: List[Dict]) -> Dict:
    """Collate WikiArtStage2 samples."""
    return {
        "images": [s["image"] for s in batch],
        "prompts": [s["prompt"] for s in batch],
        "gt_indices": [s["gt_idx"] for s in batch],
        "pool_indices": [s["pool_indices"] for s in batch],
        "style_names": [s["style_name"] for s in batch],
    }
