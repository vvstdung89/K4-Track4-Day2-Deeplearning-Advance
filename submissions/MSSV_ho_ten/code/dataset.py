"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (giữ nguyên như bộ khung):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Thêm so với khung:
    ImageCache: nạp trước toàn bộ 17.509 ảnh (uint8, 256x256) vào một mảng numpy dùng chung
    (memmap .npy) để tránh nghẽn giải mã JPEG trên Colab (chỉ 2 vCPU).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mọi backbone timm dùng ở đây đều dùng mean/std ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TOTAL = 17509
RAW_SIZE = 256


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1), không sửa gì."""
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Các kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Lỗi thì raise ngay."""
    images_dir = Path(images_dir)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    # 1. số ảnh mỗi tập và mỗi lớp
    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    ratio = {k: v / total for k, v in n.items()}
    per_class = pd.DataFrame({k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, v in splits.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)
    for k, df in splits.items():
        assert not df["Filename"].duplicated().any(), f"{k}: có Filename bị trùng trong cùng một tập"
        assert df["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: nhãn ngoài 0..8"

    # 2. giao từng cặp rỗng
    names = {k: set(v["Filename"]) for k, v in splits.items()}
    overlap = {
        "train∩val": len(names["train"] & names["val"]),
        "train∩test": len(names["train"] & names["test"]),
        "val∩test": len(names["val"] & names["test"]),
    }
    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"

    # 3. hợp ba tập = 17.509 ảnh
    union = len(names["train"] | names["val"] | names["test"])
    assert union == EXPECTED_TOTAL, f"hợp ba tập có {union} ảnh, kỳ vọng {EXPECTED_TOTAL}"

    # 4. mọi file đều tồn tại
    on_disk = {p.name for p in images_dir.iterdir()} if images_dir.is_dir() else set()
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)
    assert not missing, f"{len(missing)} file không có trong {images_dir}, ví dụ {missing[:3]}"

    for k, r in ratio.items():
        if abs(r - {"train": 0.6, "val": 0.2, "test": 0.2}[k]) > 0.01:
            print(f"CẢNH BÁO: tỉ lệ {k} = {r:.3f}, lệch > 1 điểm % so với 60/20/20")

    out = {"n": n, "ratio": {k: round(v, 4) for k, v in ratio.items()}, "union": union,
           "overlap": overlap, "missing_files": len(missing), "per_class": per_class}
    print("Số ảnh:", n, "| tỉ lệ:", out["ratio"], "| hợp:", union, "| giao:", overlap,
          "| thiếu file:", len(missing))
    print(per_class.to_string())
    return out


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic", eval_mode: str = "center"):
    """Transform trên tensor uint8 CHW (ảnh từ ImageCache) -> tensor float đã chuẩn hoá.

    aug (trục B): "basic" = RandomResizedCrop + lật ngang; "color" = basic + ColorJitter;
    "trivial" = basic + TrivialAugmentWide; "randaug" = basic + RandAugment(2, 9);
    "flipv" = basic + lật dọc (ảnh chụp từ trên xuống nên lật dọc vẫn là ảnh hợp lệ).
    Val/test (eval_mode): "center" = CenterCrop(img_size) từ ảnh gốc 256 (mặc định, công thức nền);
    "resize" = resize cả ảnh về img_size (dùng cho dò độ phân giải). Không có phép ngẫu nhiên.
    """
    norm = [T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        if eval_mode == "center" and img_size <= RAW_SIZE:
            geo = [T.CenterCrop(img_size)]
        else:
            geo = [T.Resize((img_size, img_size), antialias=True)]
        return T.Compose(geo + norm)

    ops = [T.RandomResizedCrop(img_size, antialias=True), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "flipv":
        ops.append(T.RandomVerticalFlip())
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose(ops + norm)


class ImageCache:
    """Toàn bộ ảnh DeepWeeds trong một mảng uint8 (N, 256, 256, 3), lưu thành .npy để dùng lại.

    Mở bằng mmap_mode="r" nên các worker của DataLoader dùng chung trang bộ nhớ, không sao chép.
    """

    def __init__(self, images_dir: str | Path, cache_path: str | Path | None = None,
                 filenames: list[str] | None = None):
        images_dir = Path(images_dir)
        if filenames is None:
            filenames = sorted(p.name for p in images_dir.glob("*.jpg"))
        self.filenames = list(filenames)
        self.index = {f: i for i, f in enumerate(self.filenames)}
        cache_path = Path(cache_path) if cache_path else None
        if cache_path and cache_path.exists():
            self.array = np.load(cache_path, mmap_mode="r")
            assert len(self.array) == len(self.filenames), "cache không khớp danh sách ảnh"
            return
        arr = np.empty((len(self.filenames), RAW_SIZE, RAW_SIZE, 3), dtype=np.uint8)
        for i, f in enumerate(self.filenames):
            with Image.open(images_dir / f) as im:
                im = im.convert("RGB")
                if im.size != (RAW_SIZE, RAW_SIZE):
                    im = im.resize((RAW_SIZE, RAW_SIZE), Image.BILINEAR)
                arr[i] = np.asarray(im)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, arr)
            del arr
            self.array = np.load(cache_path, mmap_mode="r")
        else:
            self.array = arr

    def get(self, filename: str) -> torch.Tensor:
        a = np.array(self.array[self.index[filename]], copy=True)  # bản sao ghi được (memmap chỉ đọc)
        return torch.from_numpy(a).permute(2, 0, 1)  # uint8 CHW


_CACHES: dict[str, ImageCache] = {}


def get_cache(images_dir: str | Path, cache_path: str | Path | None = None) -> ImageCache:
    key = str(images_dir)
    if key not in _CACHES:
        _CACHES[key] = ImageCache(images_dir, cache_path)
    return _CACHES[key]


class DeepWeedsDataset(Dataset):
    """Dataset theo DataFrame (Filename, Label). __getitem__ -> (ảnh đã transform, nhãn, tên file)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 cache: ImageCache | None = None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache = cache

    def __len__(self) -> int:
        return len(self.filenames)

    def _load(self, f: str) -> torch.Tensor:
        if self.cache is not None:
            return self.cache.get(f)
        with Image.open(self.images_dir / f) as im:
            a = np.asarray(im.convert("RGB"))
        return torch.from_numpy(a.copy()).permute(2, 0, 1)

    def __getitem__(self, i: int):
        f = self.filenames[i]
        x = self._load(f)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i], f


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                cache: ImageCache | None = None, seed: int = 0):
    """DataLoader. Train: shuffle (hoặc sampler cân bằng), drop_last. Eval: giữ đúng thứ tự df."""
    ds = DeepWeedsDataset(df, images_dir, transform, cache)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        counts = np.bincount(df["Label"].to_numpy(), minlength=NUM_CLASSES)
        w = 1.0 / counts[df["Label"].to_numpy()]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df),
                                    replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(train and smp is None), sampler=smp,
        drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0,
    )
