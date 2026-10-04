"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "eval.py").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py ở thư mục cha")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
from eval import compute_metrics, save_predictions  # noqa: E402  (eval.py gốc, không sửa)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import dataset as D  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng đặt tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | flipv
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    grad_clip: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache_path: str | None = None     # .npy chứa toàn bộ ảnh uint8 (dataset.ImageCache)
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU, CUDA). cudnn.benchmark=True để nhanh: kết quả giữa hai
    lần chạy cùng seed chỉ khớp xấp xỉ (thuật toán cuDNN không tất định), ghi rõ trong báo cáo."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD momentum 0.9) với các nhóm tham số của model.param_groups."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(cfg.optimizer)


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số LR: warmup tuyến tính (1/w -> 1) rồi cosine từ 1 về 0 ở bước cuối."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (scheduler.step() sau mỗi iteration)."""
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warm))


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W sau mỗi bước tối ưu (slide trang 56).

    Bản sao riêng (`self.module`) dùng để đánh giá. Decay khởi động d_t = min(d, (1+t)/(10+t)) để
    những bước đầu không bị kẹt ở trọng số ban đầu. Buffer (running_mean/var của BN) được SAO CHÉP
    từ model đang train (thống kê BN đi theo trọng số hiện tại, cách làm của timm ModelEmaV2).
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.param_names = {k for k, _ in self.module.named_parameters()}

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if k in self.param_names:
                v.mul_(d).add_(src, alpha=1.0 - d)
            else:
                v.copy_(src)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def _to_device(x, device, cfg: Config):
    x = x.to(device, non_blocking=True)
    return x.contiguous(memory_format=torch.channels_last) if str(device).startswith("cuda") else x


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, rng: np.random.Generator | None = None) -> dict:
    """Một epoch: train mode (giữ backbone đóng băng ở eval), Mixup/CutMix nếu có, AMP,
    clip gradient nếu cfg.grad_clip, optimizer.step(), scheduler.step() theo bước, EMA."""
    M.set_train_mode(model)
    use_amp = cfg.amp and str(device).startswith("cuda")
    tot_loss, tot_n, correct, lrs = 0.0, 0, 0, []
    for x, y, _ in loader:
        x = _to_device(x, device, cfg)
        y = y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng=rng)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = L.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn)")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n = y.size(0)
        tot_loss += loss.item() * n
        tot_n += n
        correct += (logits.argmax(1) == y).sum().item()  # vô nghĩa khi trộn mẫu, chỉ để tham khảo
    return {"train_loss": tot_loss / tot_n, "train_acc": correct / tot_n, "lr_steps": lrs}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = True):
    """Eval mode, không gradient. Trả về (filenames, y_true[N], logits[N, 9], loss)."""
    model.eval()
    use_amp = amp and str(device).startswith("cuda")
    names, ys, outs = [], [], []
    for x, y, f in loader:
        x = _to_device(x, device, None)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        outs.append(out.float().cpu())
        ys.append(torch.as_tensor(y))
        names.extend(f)
    logits, y = torch.cat(outs), torch.cat(ys)
    loss = float(criterion(logits, y)) if criterion is not None else float("nan")
    return names, y.numpy(), logits.numpy(), loss


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics_from_logits(y, logits) -> dict:
    p = softmax(logits)
    return compute_metrics(np.asarray(y), p.argmax(1), p)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps: list[float] | None = None) -> None:
    """curves/<exp_id>_<mota>.png: loss train/val, macro-F1 + top-1 val theo epoch, LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    ncol = 3 if lr_steps else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    if "val_macro_f1_raw" in history[0]:
        ax[1].plot(ep, [h["val_macro_f1_raw"] for h in history], "x--", label="val macro-F1 (raw, no EMA)")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ax[1].axvline(best["epoch"], color="gray", ls=":", label=f"best epoch {best['epoch']}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].set_title("Val metrics"); ax[1].legend()
    for a in ax[:2]:
        a.grid(alpha=0.3)
    if lr_steps:
        ax[2].plot(lr_steps)
        ax[2].set_xlabel("step"); ax[2].set_ylabel("LR (group 0)"); ax[2].set_title("LR schedule")
        ax[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def versions() -> dict:
    import timm
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def build_loss(cfg: Config, train_df):
    kw = {}
    if cfg.loss == "ls":
        kw["smoothing"] = cfg.label_smoothing or 0.1
    elif cfg.loss == "focal":
        kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=D.NUM_CLASSES)
        kw["weight"] = L.class_weights(counts, cfg.class_weight_beta or 0.0)
    return L.build_criterion(cfg.loss, **kw)


def load_model_from_run(cfg: Config, device, which: str = "best") -> nn.Module:
    """Dựng lại model và nạp checkpoint <run_dir>/best.pt (trọng số dùng để đánh giá: EMA nếu có)."""
    model = M.build_model(cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate)
    state = torch.load(run_dir(cfg) / f"{which}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ. Bỏ qua nếu <run_dir>/summary.json đã có (chạy tiếp được
    khi Colab bị ngắt). Chọn checkpoint theo MACRO-F1 VAL (hoà lấy epoch sớm hơn). Test chỉ được
    đánh giá khi cfg.save_test_predictions=True, đúng một lần, bằng checkpoint đã chọn trên val."""
    rd = run_dir(cfg)
    if (rd / "summary.json").exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã có kết quả, bỏ qua")
        return json.loads((rd / "summary.json").read_text())
    set_seed(cfg.seed)
    rd.mkdir(parents=True, exist_ok=True)
    meta = {"config": dataclasses.asdict(cfg), "versions": versions()}

    # 2. dữ liệu + kiểm tra chia
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir)
    cache = D.get_cache(cfg.images_dir, cfg.cache_path) if cfg.cache_path else None

    # 3. loader
    tr_tf = D.build_transforms(True, cfg.img_size, cfg.aug)
    ev_tf = D.build_transforms(False, cfg.img_size)
    train_loader = D.make_loader(train_df, cfg.images_dir, tr_tf, cfg.batch_size, True, cfg.sampler,
                                 cfg.num_workers, cache, seed=cfg.seed)
    val_loader = D.make_loader(val_df, cfg.images_dir, ev_tf, 128, False, None, cfg.num_workers, cache)

    # 4. model, loss, optimizer, scheduler, scaler, EMA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    meta["weight_tag"] = model.weight_tag
    meta["params_M"] = M.count_params(model)
    meta["trainable_params_M"] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    meta["gmacs"] = M.count_gmacs(model, cfg.img_size)
    criterion = build_loss(cfg, train_df).to(device)
    val_criterion = nn.CrossEntropyLoss()  # val loss luôn là CE thường để so được giữa các loss
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    rng = np.random.default_rng(cfg.seed)
    (rd / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))

    # 5. vòng epoch
    history, lr_steps, best, best_state, epoch_times = [], [], None, None, []
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema, rng)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        lr_steps.extend(tr.pop("lr_steps"))
        eval_model = ema.module if ema else model
        _, yv, zv, vloss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
        mv = metrics_from_logits(yv, zv)
        row = {"epoch": epoch, **tr, "val_loss": vloss, "val_macro_f1": mv["macro_f1"],
               "val_top1": mv["top1"], "val_bal_acc": mv["balanced_acc"], "val_ece": mv["ece"],
               "lr_end": optimizer.param_groups[0]["lr"], "epoch_time_s": epoch_times[-1]}
        if ema:
            _, _, zr, _ = evaluate(model, val_loader, None, device, cfg.amp)
            row["val_macro_f1_raw"] = metrics_from_logits(yv, zr)["macro_f1"]
        history.append(row)
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch:2d} train_loss {tr['train_loss']:.4f} "
              f"val_loss {vloss:.4f} val_F1 {mv['macro_f1']:.4f} val_top1 {mv['top1']:.4f} "
              f"({epoch_times[-1]:.0f}s)", flush=True)
        if best is None or mv["macro_f1"] > best["val_macro_f1"]:  # hoà: giữ epoch sớm hơn
            best = row
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in eval_model.state_dict().items()}

    # 6. nạp checkpoint tốt nhất, lưu val logits + predictions val
    model_eval = M.build_model(cfg.backbone, False, D.NUM_CLASSES, cfg.drop_rate).to(device)
    model_eval.load_state_dict(best_state)
    if device.type == "cuda":
        model_eval = model_eval.to(memory_format=torch.channels_last)
    torch.save(best_state, rd / "best.pt")
    fv, yv, zv, _ = evaluate(model_eval, val_loader, None, device, cfg.amp)
    np.save(rd / "val_logits.npy", zv)
    np.save(rd / "val_labels.npy", yv)
    (rd / "val_filenames.txt").write_text("\n".join(fv))
    save_predictions(pred_path(cfg, "val"), fv, yv, softmax(zv))
    mv = metrics_from_logits(yv, zv)

    # 7. test: chỉ ở Bước 4, đúng một lần
    if cfg.save_test_predictions:
        evaluate_split(cfg, "test", model_eval, device)

    # 8. history, curves, latency sơ bộ, summary
    import pandas as pd
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_steps, dtype=np.float32))
    name = f"{cfg.exp_id}_{cfg.desc or cfg.backbone}"
    plot_curves(history, Path(cfg.curves_dir) / f"{name}.png",
                f"{cfg.exp_id} | {cfg.backbone} | {cfg.desc} | seed {cfg.seed}", lr_steps)
    lat = None
    if device.type == "cuda":
        from benchmark import latency_report
        lat = latency_report(model_eval, 1, cfg.img_size, "fp32", "cuda", warmup=10, iters=50)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "desc": cfg.desc,
        "weight_tag": meta["weight_tag"], "params_M": meta["params_M"], "gmacs": meta["gmacs"],
        "best_epoch": best["epoch"], "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"],
        "val_bal_acc": mv["balanced_acc"], "val_ece": mv["ece"],
        "val_f1_per_class": mv["f1"].tolist(), "val_recall_per_class": mv["recall"].tolist(),
        "train_time_per_epoch_s": float(np.mean(epoch_times)), "epochs": cfg.epochs,
        "latency_b1_fp32_p50_ms": lat["p50"] if lat else None,
        "latency_b1_fp32_p95_ms": lat["p95"] if lat else None,
        "init_loss_hint": history[0]["train_loss"],
        "config": dataclasses.asdict(cfg), "versions": meta["versions"],
    }
    if ema:
        best_raw = max(h["val_macro_f1_raw"] for h in history)
        summary["val_macro_f1_raw_best"] = best_raw
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    del model, model_eval, optimizer, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def evaluate_split(cfg: Config, split: str, model=None, device=None) -> dict:
    """Đánh giá checkpoint đã chọn trên `split` (1-view, công thức đánh giá gốc), lưu logits và
    predictions/<exp_id>_seed<k>_<split>.csv. Với split="test": chỉ gọi ở Bước 4, mỗi seed một lần;
    hàm từ chối chạy lại nếu logit test của lần chạy này đã tồn tại."""
    rd = run_dir(cfg)
    if split == "test" and (rd / "test_logits.npy").exists():
        raise RuntimeError(f"{rd}: test đã được đánh giá một lần, không chạy lại (quy tắc S4)")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model_from_run(cfg, device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
    dfs = dict(zip(("train", "val", "test"), D.load_split(cfg.labels_dir, cfg.fold)))
    cache = D.get_cache(cfg.images_dir, cfg.cache_path) if cfg.cache_path else None
    loader = D.make_loader(dfs[split], cfg.images_dir, D.build_transforms(False, cfg.img_size), 128,
                           False, None, cfg.num_workers, cache)
    f, y, z, _ = evaluate(model, loader, None, device, cfg.amp)
    np.save(rd / f"{split}_logits.npy", z)
    np.save(rd / f"{split}_labels.npy", y)
    (rd / f"{split}_filenames.txt").write_text("\n".join(f))
    save_predictions(pred_path(cfg, split), f, y, softmax(z))
    return metrics_from_logits(y, z)


def _cast(value: str, type_str: str):
    v = value.strip()
    t = type_str.replace(" ", "")
    if v.lower() in ("none", "null") and "None" in t:
        return None
    base = t.replace("|None", "").replace("None|", "")
    if base == "bool":
        if v.lower() in ("1", "true", "yes", "y"):
            return True
        if v.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"không đọc được bool: {value}")
    if base == "int":
        return int(v)
    if base == "float":
        return float(v)
    return v


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' không có dạng KEY=VALUE")
        k, v = pair.split("=", 1)
        k = k.strip()
        if k not in types:
            raise KeyError(f"Config không có trường '{k}'. Các trường: {sorted(types)}")
        out[k] = _cast(v, types[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    summary = run(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "versions")}, indent=2,
                     default=str))


if __name__ == "__main__":
    main()
