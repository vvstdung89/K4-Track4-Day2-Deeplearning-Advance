"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
Mọi hàm chạy ở eval, không gradient. Chọn phương pháp CHỈ trên val; T khớp trên VAL.

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views)      -> (filenames, y_true, [logits_view_k])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.inference_mode()
def predict_views(model, loader, device, views, amp: bool = True):
    """Một lượt qua loader, chạy model trên từng view của mỗi batch. Trả về list logit theo view.

    Mỗi phần tử của `views` là hàm batch -> batch, hoặc -> list các batch (multi-crop: logit của
    các batch con được trung bình cộng theo không gian logit trong view đó).
    """
    model.eval()
    names, ys = [], []
    outs = [[] for _ in views]
    use_amp = amp and str(device).startswith("cuda")
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        for k, view in enumerate(views):
            xv = view(x) if view is not None else x
            batches = xv if isinstance(xv, (list, tuple)) else [xv]
            acc = []
            for b in batches:
                b = b.contiguous(memory_format=torch.channels_last) if use_amp else b
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    acc.append(model(b).float())
            outs[k].append(torch.stack(acc).mean(0).cpu())
        names.extend(f)
        ys.append(torch.as_tensor(y))
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader, giữ đúng thứ tự file. Trả về (filenames, y_true, logits) dạng numpy."""
    names, y, outs = predict_views(model, loader, device, [view], amp=amp)
    return names, y, outs[0]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng."""
    return torch.flip(x, dims=[3])


def view_center(x, crop: int = 224):
    """Center crop `crop` (dùng khi loader trả ảnh gốc 256): tương đương CenterCrop lúc val."""
    h, w = x.shape[-2:]
    t, l = (h - crop) // 2, (w - crop) // 2
    return x[..., t:t + crop, l:l + crop]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; flip=True thêm bản lật của cả 5 (10 crop)."""
    h, w = x.shape[-2:]
    tops = [0, 0, h - crop, h - crop, (h - crop) // 2]
    lefts = [0, w - crop, 0, w - crop, (w - crop) // 2]
    crops = [x[..., t:t + crop, l:l + crop] for t, l in zip(tops, lefts)]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). CNN có global pooling
    nhận được mọi kích thước; ViT/Swin cần nội suy position embedding/cửa sổ (không dùng ở đây)."""
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear",
                                                     align_corners=False, antialias=True)
            for s in sizes]


def make_view(name: str, crop: int = 224):
    """Tạo hàm view theo tên, cho loader trả ảnh gốc 256 đã chuẩn hoá (không crop)."""
    if name == "center":
        return partial(view_center, crop=crop)
    if name == "center_flip":
        return lambda x: view_hflip(view_center(x, crop))
    if name == "five_crop":
        return partial(views_multicrop, crop=crop, flip=False)
    if name == "ten_crop":
        return partial(views_multicrop, crop=crop, flip=True)
    if name.startswith("full"):  # full<size>: resize cả ảnh 256 về size
        size = int(name[4:])
        return lambda x: views_multiscale(x, [size])[0]
    raise ValueError(name)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax."""
    L = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        p = np.mean([softmax_np(l) for l in L], axis=0)
    elif space == "logit":
        p = softmax_np(L.mean(0))
    else:
        raise ValueError(space)
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    P = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    if P.ndim != 3:
        raise ValueError("cần list các mảng (N, K) cùng kích thước")
    p = P.mean(0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T). Tìm lưới thô trên log T rồi LBFGS."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = torch.linspace(np.log(0.05), np.log(20.0), 200, dtype=torch.float64)
    nll = torch.stack([F.cross_entropy(z / g.exp(), y) for g in grid])
    log_t = grid[nll.argmin()].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode).to(conv.weight.device, conv.weight.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight if bn.affine else torch.ones_like(std)
    beta = bn.bias if bn.affine else torch.zeros_like(std)
    scale = gamma / std
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(beta + (b - bn.running_mean) * scale)
    return fused


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (BN + drop + act) -> giữ lại drop + act."""
    act, drop = getattr(bn, "act", None), getattr(bn, "drop", None)
    if act is None and drop is None:
        return nn.Identity()
    return nn.Sequential(drop or nn.Identity(), act or nn.Identity())


@torch.no_grad()
def fuse_conv_bn(model, check_input: torch.Tensor | None = None):
    """Gộp mọi cặp (Conv2d, BatchNorm2d) là hai module con LIỀN KỀ trong cùng một cha (đúng thứ tự
    forward của ResNet/ResNeXt/EfficientNet/MobileNet trong timm). Trả về bản sao đã gộp.

    Nếu có `check_input`, in sai số lớn nhất giữa đầu ra trước/sau (kỳ vọng ~1e-5).
    Mạng không có BN (ViT, Swin, ConvNeXt dùng LayerNorm): không có gì để gộp, trả về n_fused = 0.
    """
    model.eval()
    fused_model = copy.deepcopy(model).eval()
    n_fused = 0
    for parent in fused_model.modules():
        children = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(children, children[1:]):
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d) and m2.track_running_stats:
                setattr(parent, n1, _fuse_pair(m1, m2))
                setattr(parent, n2, _bn_replacement(m2))
                n_fused += 1
    fused_model.n_fused = n_fused
    if check_input is not None:
        ref = model(check_input).float()
        out = fused_model(check_input).float()
        fused_model.max_abs_err = float((ref - out).abs().max())
        print(f"fuse_conv_bn: gộp {n_fused} cặp, sai số lớn nhất {fused_model.max_abs_err:.2e}")
    return fused_model
