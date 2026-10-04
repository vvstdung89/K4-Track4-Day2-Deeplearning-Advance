"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo đã cài đặt:
  - warmup: bỏ `warmup` (mặc định 10) lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - `iters` (mặc định 100, >= 50) lần đo, báo cáo p50, p95, p99 và mean
  - ghi GPU, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (đầu vào là tensor đã nằm sẵn trên GPU): chỉ đo forward của model
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo `fn()` (không tham số), trả về mili-giây: {"p50","p95","p99","mean","n"}."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, bn_fused: bool = False, views: int = 1) -> dict:
    """Độ trễ forward với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half(), dùng bản sao để không đổi model gốc).
    views: số lượt forward liên tiếp mỗi lần đo (TTA K view, đo thật K lượt).
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    m = model
    if dtype == "fp16":
        m = copy.deepcopy(model).half()
    m = m.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        x = x.half()
    if device == "cuda":
        m = m.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    sync = torch.cuda.synchronize if device == "cuda" else None

    def fn():
        with torch.inference_mode():
            if dtype == "amp":
                with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
                    for _ in range(views):
                        m(x)
            else:
                for _ in range(views):
                    m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    return {"gpu": gpu, "dtype": dtype, "batch": batch_size, "img_size": img_size, "views": views,
            "bn_fused": bn_fused, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
            "n": r["n"], "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__,
            "preprocessing_included": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (K lượt forward thật trong mỗi lần đo), kèm so sánh với K * p50 của 1 view."""
    one = latency_report(model, views=1, **kw)
    k = latency_report(model, views=k_views, **kw)
    k["k_times_single_p50"] = k_views * one["p50"]
    k["ratio_vs_single"] = k["p50"] / one["p50"]
    return k
