"""
CelebA Dataset Loading and Preprocessing.

Includes conditional edge/sketch generation for Canny/XDoG/HED/PiDiNet and
mixed conditioning for robust freehand-style control.
"""

import os
from typing import Optional, Tuple, Callable, Union, List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid as torch_make_grid
from torchvision.utils import save_image as torch_save_image
from PIL import Image
import numpy as np


_HED_DETECTOR = None
_PIDI_DETECTOR = None
_DETECTOR_WARNED = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _DETECTOR_WARNED:
        return
    _DETECTOR_WARNED.add(key)
    print(msg)


def _ensure_white_edges_on_black(gray_uint8: np.ndarray) -> np.ndarray:
    # Heuristic polarity fix: if bright background dominates, invert to ensure
    # white edges on black background.
    if gray_uint8.mean() > 127:
        gray_uint8 = 255 - gray_uint8
    return gray_uint8


def _gray_to_rgb_pil(gray_uint8: np.ndarray) -> Image.Image:
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _get_hed_detector():
    global _HED_DETECTOR
    if _HED_DETECTOR is not None:
        return _HED_DETECTOR
    try:
        from controlnet_aux import HEDdetector
    except ImportError as exc:
        raise ImportError(
            "HED edge extraction requires controlnet-aux. Install with: pip install controlnet-aux"
        ) from exc
    _HED_DETECTOR = HEDdetector.from_pretrained("lllyasviel/Annotators")
    _warn_once("hed_loaded", "[edge] Loaded HED detector (lllyasviel/Annotators).")
    return _HED_DETECTOR


def _get_pidinet_detector():
    global _PIDI_DETECTOR
    if _PIDI_DETECTOR is not None:
        return _PIDI_DETECTOR
    try:
        from controlnet_aux import PidiNetDetector
    except ImportError as exc:
        raise ImportError(
            "PiDiNet edge extraction requires controlnet-aux. Install with: pip install controlnet-aux"
        ) from exc
    _PIDI_DETECTOR = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    _warn_once("pidi_loaded", "[edge] Loaded PiDiNet detector (lllyasviel/Annotators).")
    return _PIDI_DETECTOR


def canny_edges(
    pil_image: Image.Image,
    sigma: float = 1.2,
    low: int = 50,
    high: int = 150,
) -> Image.Image:
    """Extract Canny edges as 3-channel white edges on black."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for edge extraction. Install with: pip install opencv-python-headless"
        ) from exc

    arr = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(blurred, low, high)
    edges = _ensure_white_edges_on_black(edges)
    return _gray_to_rgb_pil(edges)


def xdog_edges(
    pil_image: Image.Image,
    sigma: float = 0.29,
    k: float = 1.6,
    gamma: float = 0.98,
    epsilon: float = 0.01,
    phi: float = 10.0,
) -> Image.Image:
    """Extract XDoG edges as 3-channel white edges on black."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for edge extraction. Install with: pip install opencv-python-headless"
        ) from exc

    arr = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k, sigmaY=sigma * k)
    dog = g1 - gamma * g2
    dog = dog / (np.max(np.abs(dog)) + 1e-8)
    result = np.where(dog >= epsilon, np.ones_like(dog), 1.0 + np.tanh(phi * (dog - epsilon)))
    result = np.clip(result, 0.0, 1.0)
    edges = ((1.0 - result) * 255.0).astype(np.uint8)
    edges = _ensure_white_edges_on_black(edges)
    return _gray_to_rgb_pil(edges)


def hed_edges(
    pil_image: Image.Image,
    detect_resolution: Optional[int] = None,
    image_resolution: Optional[int] = None,
) -> Image.Image:
    """Extract HED edges as 3-channel white edges on black."""
    detector = _get_hed_detector()
    kwargs = {}
    if detect_resolution is not None:
        kwargs["detect_resolution"] = detect_resolution
    if image_resolution is not None:
        kwargs["image_resolution"] = image_resolution

    # controlnet_aux signatures vary by version, so we fallback gracefully.
    try:
        out = detector(pil_image.convert("RGB"), **kwargs)
    except TypeError:
        out = detector(pil_image.convert("RGB"))

    gray = np.array(out.convert("L"), dtype=np.uint8)
    gray = _ensure_white_edges_on_black(gray)
    return _gray_to_rgb_pil(gray)


def pidinet_edges(
    pil_image: Image.Image,
    threshold: float = 0.5,
    use_binary: bool = False,
) -> Image.Image:
    """Extract PiDiNet edges as 3-channel white edges on black."""
    detector = _get_pidinet_detector()
    threshold = float(np.clip(threshold, 0.0, 1.0))

    try:
        out = detector(pil_image.convert("RGB"), safe=True, apply_filter=False)
    except TypeError:
        out = detector(pil_image.convert("RGB"), safe=True)

    gray = np.array(out.convert("L"), dtype=np.uint8)
    if use_binary:
        gray = (gray >= int(255 * threshold)).astype(np.uint8) * 255
    gray = _ensure_white_edges_on_black(gray)
    return _gray_to_rgb_pil(gray)


class CelebADataset(Dataset):
    """
    CelebA dataset wrapper with preprocessing for diffusion models.
    """

    def __init__(
        self,
        root: str = "./data/celeba-subset",
        split: str = "train",
        image_size: int = 64,
        augment: bool = True,
        conditional: bool = False,
        edge_method: str = "xdog",
        from_hub: bool = False,
        repo_name: str = "electronickale/cmu-10799-celeba64-subset",
        edge_mix_methods: Optional[List[str]] = None,
        edge_mix_alpha: float = 1.0,
        edge_mix_binary_prob: float = 0.35,
        edge_mix_dropout_prob: float = 0.05,
        edge_mix_blur_prob: float = 0.20,
        edge_mix_morph_prob: float = 0.20,
        edge_mix_hed_detect_resolution: Optional[int] = None,
        edge_mix_hed_image_resolution: Optional[int] = None,
        edge_mix_pidinet_threshold: float = 0.5,
        edge_mix_pidinet_binary: bool = False,
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.conditional = conditional
        self.edge_method = edge_method.lower()
        self.from_hub = from_hub
        self.repo_name = repo_name

        # Mixed edge conditioning options.
        self.edge_mix_methods = (
            [m.lower() for m in edge_mix_methods]
            if edge_mix_methods is not None
            else ["canny", "xdog", "hed", "pidinet"]
        )
        if len(self.edge_mix_methods) == 0:
            raise ValueError("edge_mix_methods cannot be empty when edge_method='mixed'.")
        self.edge_mix_alpha = float(edge_mix_alpha)
        self.edge_mix_binary_prob = float(np.clip(edge_mix_binary_prob, 0.0, 1.0))
        self.edge_mix_dropout_prob = float(np.clip(edge_mix_dropout_prob, 0.0, 1.0))
        self.edge_mix_blur_prob = float(np.clip(edge_mix_blur_prob, 0.0, 1.0))
        self.edge_mix_morph_prob = float(np.clip(edge_mix_morph_prob, 0.0, 1.0))
        self.edge_mix_hed_detect_resolution = edge_mix_hed_detect_resolution
        self.edge_mix_hed_image_resolution = edge_mix_hed_image_resolution
        self.edge_mix_pidinet_threshold = float(np.clip(edge_mix_pidinet_threshold, 0.0, 1.0))
        self.edge_mix_pidinet_binary = bool(edge_mix_pidinet_binary)

        if self.edge_mix_alpha <= 0:
            raise ValueError("edge_mix_alpha must be > 0.")

        self.transform = self._build_transforms()
        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if from_hub:
            self._load_from_hub()
        else:
            self._load_from_local()

    def _load_from_hub(self):
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError:
            raise ImportError(
                "Please install the datasets library to load from HuggingFace Hub:\n"
                "  pip install datasets"
            )

        from pathlib import Path

        root_path = Path(self.root)
        print(f"Attempt to use cached dataset from: {self.root}")
        if root_path.exists() and (root_path / "dataset_dict.json").exists():
            print("=" * 60)
            print(f"[ok] Using cached dataset from: {self.root}")
            print("=" * 60)

            hf_split = "validation" if self.split == "valid" else self.split
            dataset = load_from_disk(self.root)

            if hf_split == "all":
                all_data = []
                for split_name in dataset.keys():
                    all_data.extend(list(dataset[split_name]))
                self.data = all_data
            else:
                self.data = list(dataset[hf_split])
            print(f"[ok] Loaded {len(self.data)} images from cached '{hf_split}' split")
            return

        print("=" * 60)
        print(f"[download] HuggingFace dataset: {self.repo_name}")
        print("=" * 60)

        hf_split = "validation" if self.split == "valid" else self.split
        cache_dir = None
        if self.root:
            os.makedirs(self.root, exist_ok=True)
            cache_dir = self.root

        if hf_split == "all":
            self.dataset = load_dataset(self.repo_name, cache_dir=cache_dir)
            all_data = []
            for split_name in self.dataset.keys():
                all_data.extend(list(self.dataset[split_name]))
            self.data = all_data
        else:
            self.dataset = load_dataset(self.repo_name, split=hf_split, cache_dir=cache_dir)
            self.data = list(self.dataset)

        print(f"Loaded {len(self.data)} images from {hf_split} split")

    def _load_from_local(self):
        from pathlib import Path

        if self._try_load_from_saved_dataset():
            return

        split_dir = "validation" if self.split == "valid" else self.split
        if self.split == "all":
            train_path = Path(self.root) / "train"
            val_path = Path(self.root) / "validation"
            self.data = []
            if train_path.exists():
                self.data.extend(self._load_split_data(train_path))
            if val_path.exists():
                self.data.extend(self._load_split_data(val_path))
        else:
            split_path = Path(self.root) / split_dir
            self.data = self._load_split_data(split_path)
        print(f"Loaded {len(self.data)} images from local directory")

    def _try_load_from_saved_dataset(self) -> bool:
        from pathlib import Path

        root_path = Path(self.root)
        if not root_path.exists():
            return False
        if not (root_path / "dataset_info.json").exists():
            return False
        try:
            from datasets import load_from_disk
        except ImportError:
            return False

        hf_split = "validation" if self.split == "valid" else self.split
        dataset = load_from_disk(self.root)
        if hf_split == "all":
            all_data = []
            for split_name in dataset.keys():
                all_data.extend(list(dataset[split_name]))
            self.data = all_data
        else:
            self.data = list(dataset[hf_split])
        print(f"Loaded {len(self.data)} images from {hf_split} split")
        return True

    def _load_split_data(self, split_path):
        from pathlib import Path

        images_dir = split_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                "Please download the dataset first."
            )

        image_files = sorted(images_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(images_dir.glob("*.jpg"))

        data = []
        for img_path in image_files:
            data.append({"image": str(img_path), "image_id": img_path.name})
        return data

    def _build_transforms(self) -> Callable:
        transform_list = []
        if self.augment and self.split == "train":
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        return transforms.Compose(transform_list)

    def _single_edge(self, image: Image.Image, method: str) -> Image.Image:
        m = method.lower()
        if m == "canny":
            out = canny_edges(image)
            return out.resize(image.size, Image.BILINEAR) if out.size != image.size else out
        if m == "xdog":
            out = xdog_edges(image)
            return out.resize(image.size, Image.BILINEAR) if out.size != image.size else out
        if m == "hed":
            out = hed_edges(
                image,
                detect_resolution=self.edge_mix_hed_detect_resolution,
                image_resolution=self.edge_mix_hed_image_resolution,
            )
            return out.resize(image.size, Image.BILINEAR) if out.size != image.size else out
        if m == "pidinet":
            out = pidinet_edges(
                image,
                threshold=self.edge_mix_pidinet_threshold,
                use_binary=self.edge_mix_pidinet_binary,
            )
            return out.resize(image.size, Image.BILINEAR) if out.size != image.size else out
        raise ValueError(f"Unsupported edge method: {method}")

    def _apply_mushy_aug(self, edge01: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for mixed edge augmentations. Install with: pip install opencv-python-headless"
            ) from exc

        x = edge01.astype(np.float32)
        if np.random.rand() < self.edge_mix_morph_prob:
            k = np.random.choice([1, 2])
            kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
            if np.random.rand() < 0.5:
                x = cv2.dilate(x, kernel, iterations=1)
            else:
                x = cv2.erode(x, kernel, iterations=1)
        if np.random.rand() < self.edge_mix_blur_prob:
            sigma = float(np.random.uniform(0.4, 1.4))
            x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if np.random.rand() < self.edge_mix_dropout_prob:
            keep = np.random.rand(*x.shape) > self.edge_mix_dropout_prob
            x = x * keep.astype(np.float32)
        if np.random.rand() < self.edge_mix_binary_prob:
            thr = float(np.random.uniform(0.35, 0.65))
            x = (x >= thr).astype(np.float32)
        return np.clip(x, 0.0, 1.0)

    def _mixed_edges(self, image: Image.Image) -> Image.Image:
        maps = []
        for method in self.edge_mix_methods:
            emap = self._single_edge(image, method)
            gray = np.array(emap.convert("L"), dtype=np.float32) / 255.0
            maps.append(gray)
        stacked = np.stack(maps, axis=0)  # (K, H, W)

        alpha = np.full((len(self.edge_mix_methods),), self.edge_mix_alpha, dtype=np.float64)
        weights = np.random.dirichlet(alpha).astype(np.float32)
        mix = np.tensordot(weights, stacked, axes=(0, 0))
        mix = self._apply_mushy_aug(mix)
        mix_u8 = (mix * 255.0).astype(np.uint8)
        mix_u8 = _ensure_white_edges_on_black(mix_u8)
        return _gray_to_rgb_pil(mix_u8)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        item = self.data[idx]

        if self.from_hub:
            image = item["image"].convert("RGB")
        else:
            image = Image.open(item["image"]).convert("RGB")

        if self.conditional:
            if self.edge_method == "mixed":
                edge = self._mixed_edges(image)
            else:
                edge = self._single_edge(image, self.edge_method)

            if self.augment and self.split == "train" and torch.rand(()) < 0.5:
                image = TF.hflip(image)
                edge = TF.hflip(edge)
            return self.base_transform(image), self.base_transform(edge)

        if self.transform:
            image = self.transform(image)
        return image


def create_dataloader(
    root: str = "./data/celeba-subset",
    split: str = "train",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
    conditional: bool = False,
    edge_method: str = "xdog",
    shuffle: Optional[bool] = None,
    drop_last: bool = True,
    from_hub: bool = False,
    repo_name: str = "electronickale/cmu-10799-celeba64-subset",
    edge_mix_methods: Optional[List[str]] = None,
    edge_mix_alpha: float = 1.0,
    edge_mix_binary_prob: float = 0.35,
    edge_mix_dropout_prob: float = 0.05,
    edge_mix_blur_prob: float = 0.20,
    edge_mix_morph_prob: float = 0.20,
    edge_mix_hed_detect_resolution: Optional[int] = None,
    edge_mix_hed_image_resolution: Optional[int] = None,
    edge_mix_pidinet_threshold: float = 0.5,
    edge_mix_pidinet_binary: bool = False,
) -> DataLoader:
    dataset = CelebADataset(
        root=root,
        split=split,
        image_size=image_size,
        augment=augment,
        conditional=conditional,
        edge_method=edge_method,
        from_hub=from_hub,
        repo_name=repo_name,
        edge_mix_methods=edge_mix_methods,
        edge_mix_alpha=edge_mix_alpha,
        edge_mix_binary_prob=edge_mix_binary_prob,
        edge_mix_dropout_prob=edge_mix_dropout_prob,
        edge_mix_blur_prob=edge_mix_blur_prob,
        edge_mix_morph_prob=edge_mix_morph_prob,
        edge_mix_hed_detect_resolution=edge_mix_hed_detect_resolution,
        edge_mix_hed_image_resolution=edge_mix_hed_image_resolution,
        edge_mix_pidinet_threshold=edge_mix_pidinet_threshold,
        edge_mix_pidinet_binary=edge_mix_pidinet_binary,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dataloader_from_config(config: dict, split: str = "train") -> DataLoader:
    data_config = config["data"]
    training_config = config["training"]

    return create_dataloader(
        root=data_config.get("root", "./data/celeba-subset"),
        split=split,
        image_size=data_config["image_size"],
        batch_size=training_config["batch_size"],
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        augment=(split == "train" and data_config.get("augment", True)),
        conditional=data_config.get("conditional", False),
        edge_method=data_config.get("edge_method", "xdog"),
        from_hub=data_config.get("from_hub", False),
        repo_name=data_config.get("repo_name", "electronickale/cmu-10799-celeba64-subset"),
        edge_mix_methods=data_config.get("edge_mix_methods", None),
        edge_mix_alpha=float(data_config.get("edge_mix_alpha", 1.0)),
        edge_mix_binary_prob=float(data_config.get("edge_mix_binary_prob", 0.35)),
        edge_mix_dropout_prob=float(data_config.get("edge_mix_dropout_prob", 0.05)),
        edge_mix_blur_prob=float(data_config.get("edge_mix_blur_prob", 0.20)),
        edge_mix_morph_prob=float(data_config.get("edge_mix_morph_prob", 0.20)),
        edge_mix_hed_detect_resolution=data_config.get("edge_mix_hed_detect_resolution", None),
        edge_mix_hed_image_resolution=data_config.get("edge_mix_hed_image_resolution", None),
        edge_mix_pidinet_threshold=float(data_config.get("edge_mix_pidinet_threshold", 0.5)),
        edge_mix_pidinet_binary=bool(data_config.get("edge_mix_pidinet_binary", False)),
    )


def unnormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1]."""
    return (images + 1.0) / 2.0


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [0, 1] to [-1, 1]."""
    return images * 2.0 - 1.0


def make_grid(images: torch.Tensor, nrow: int = 8, **kwargs) -> torch.Tensor:
    """Create a grid of images."""
    return torch_make_grid(images, nrow=nrow, **kwargs)


def save_image(images: torch.Tensor, path: str, nrow: int = 8, **kwargs):
    """Save a batch of images as a grid."""
    torch_save_image(images, path, nrow=nrow, **kwargs)
