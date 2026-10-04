"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện (giữ nguyên như bộ khung):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
Thêm: set_train_mode(model) (giữ phần đóng băng ở eval), weight_tag(model).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Model phân loại 9 lớp. init: "scratch" | "frozen" | "finetune" (trục A)."""
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    timm_name = SUGGESTED_BACKBONES.get(name, name)
    use_pretrained = pretrained and init != "scratch"
    model = timm.create_model(timm_name, pretrained=use_pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)
    model.init_mode = init
    model.weight_tag = weight_tag(model) if use_pretrained else f"{timm_name} (random init)"
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    """Tag trọng số timm thực sự được tải, ví dụ 'resnet50.a1_in1k'."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", "")
    tag = cfg.get("tag", "")
    return f"{arch}.{tag}" if tag else arch


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head. BN của backbone giữ ở eval (xem set_train_mode)."""
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True
    set_train_mode(model)


def set_train_mode(model) -> None:
    """model.train(); nếu backbone bị đóng băng thì đưa mọi module ngoài head về eval
    (BatchNorm không cập nhật running stats, dropout trong backbone tắt)."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = model.get_classifier()
        head_modules = set(head.modules())
        for m in model.modules():
            if m not in head_modules and m is not model:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Nhóm tham số theo slide trang 52:
      - backbone, ndim > 1           : lr_backbone, weight_decay
      - backbone norm/bias (ndim<=1) : lr_backbone, weight_decay = 0
      - head weight                  : lr_head,     weight_decay
      - head bias                    : lr_head,     weight_decay = 0  (tách riêng để bias không bị decay)
    Bỏ qua tham số requires_grad == False. Nhóm rỗng bị loại.
    """
    head = _head_param_ids(model)
    groups = {
        "backbone_decay": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr_head, "weight_decay": weight_decay},
        "head_no_decay": {"params": [], "lr": lr_head, "weight_decay": 0.0},
    }
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head else "backbone"
        kind = "decay" if p.ndim > 1 else "no_decay"
        groups[f"{part}_{kind}"]["params"].append(p)
    out = []
    for name, g in groups.items():
        if g["params"]:
            g["name"] = name
            out.append(g)
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Dùng torch.utils.flop_counter.FlopCounterMode (đếm ở mức toán tử aten, gồm cả matmul của
    attention); MAC = FLOP / 2. Không đếm phép cộng/kích hoạt/chuẩn hoá (giống fvcore).
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    if was_training:
        model.train()
    return counter.get_total_flops() / 2 / 1e9
