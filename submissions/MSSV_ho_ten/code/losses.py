"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện (giữ nguyên như bộ khung):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """kind: "ce" | "ls" (label smoothing) | "focal" | "ce_weighted".

    kw: smoothing (ls), gamma và alpha (focal), weight (ce_weighted, tensor 9 phần tử).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight=class_weights(...)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """CE với nhãn mềm q'(k) = (1 - eps) * 1[k == y] + eps / K (tự cài đặt, slide trang 56).

    loss = (1 - eps) * NLL(y) + eps * mean_k(-log p_k). eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), lấy trung bình theo batch (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp. gamma = 0 và alpha = None cho đúng CE.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp(min=0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của TRAIN.

    beta = 0 : w_c ∝ 1 / n_c, chuẩn hoá về trung bình 1.
    beta > 0 : class-balanced, w_c = (1 - beta) / (1 - beta^n_c), chuẩn hoá tổng = số lớp.
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh trong train")
    if beta == 0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)


def rand_box(h: int, w: int, lam: float, rng: np.random.Generator):
    """Hộp CutMix có diện tích danh nghĩa (1 - lam) * H * W, tâm ngẫu nhiên, cắt theo biên ảnh."""
    cut_rat = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut_rat), int(w * cut_rat)
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn batch: trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].

    mixup : x_mix = lam * x + (1 - lam) * x[perm]
    cutmix: dán hộp từ x[perm] vào x; lam = 1 - diện tích THỰC của hộp (sau khi cắt biên) / (H*W).
    """
    rng = rng if rng is not None else np.random.default_rng()
    lam = float(rng.beta(alpha, alpha))
    perm = torch.from_numpy(rng.permutation(x.size(0))).to(x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_box(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
