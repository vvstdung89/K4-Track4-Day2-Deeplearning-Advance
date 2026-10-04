"""run_experiments.py - điều phối mọi thí nghiệm của lab (gọi train.run cho mọi cấu hình).

Các stage (chạy theo thứ tự, mỗi stage chạy tiếp được nếu Colab bị ngắt):
    python run_experiments.py --stage eda
    python run_experiments.py --stage sanity
    python run_experiments.py --stage backbones
    python run_experiments.py --stage training  --backbone convnext_tiny
    python run_experiments.py --stage combo     --backbone convnext_tiny --combo "mix=cutmix loss=ls"
    python run_experiments.py --stage inference --backbone convnext_tiny
    python run_experiments.py --stage final     --backbone convnext_tiny --combo "..." [--method ...]

Đường dẫn lấy từ biến môi trường:
    LAB_SUB   thư mục bài nộp (runs/, predictions/, curves/, figures/, tables/ nằm trong đó)
    LAB_DATA  thư mục dữ liệu (images/, labels/, cache.npy)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark as B
import dataset as D
import inference as I
import model as M
import train as TR
from train import Config, compute_metrics, save_predictions

SUB = Path(os.environ.get("LAB_SUB", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("LAB_DATA", "data"))
EPOCHS = int(os.environ.get("LAB_EPOCHS", 10))
TABLES = SUB / "tables"
FIGS = SUB / "figures"

# Bước 1: >= 5 backbone (ResNet, ConvNeXt, transformer, 2 mạng nhẹ)
BACKBONES = [
    ("B01", "resnet50"),
    ("B02", "convnext_tiny"),
    ("B03", "deit_small"),
    ("B04", "efficientnet_b0"),
    ("B05", "mobilenetv3"),
]

# Bước 2: mỗi thí nghiệm khác T00 đúng MỘT yếu tố: (exp_id, trục, mô tả, overrides)
TRAINING = [
    ("T01", "A. Khởi tạo", "scratch", {"init": "scratch"}),
    ("T02", "A. Khởi tạo", "frozen", {"init": "frozen"}),
    ("T03", "B. Augmentation", "trivialaug", {"aug": "trivial"}),
    ("T04", "B. Augmentation", "cutmix", {"mix": "cutmix", "mix_alpha": 1.0}),
    ("T05", "C. Loss", "labelsmooth", {"loss": "ls", "label_smoothing": 0.1}),
    ("T06", "C. Loss", "focal", {"loss": "focal", "focal_gamma": 2.0}),
    ("T07", "C. Loss", "ce_weighted", {"loss": "ce_weighted", "class_weight_beta": 0.0}),
    ("T08", "F. Chính quy hoá", "ema", {"ema_decay": 0.999}),
    ("T09", "D. Cân bằng mẫu", "balanced_sampler", {"sampler": "balanced"}),
]


def base_cfg(**kw) -> Config:
    c = Config(epochs=EPOCHS, images_dir=str(DATA / "images"), labels_dir=str(DATA / "labels"),
               cache_path=str(DATA / "cache.npy"), out_dir=str(SUB / "runs"),
               pred_dir=str(SUB / "predictions"), curves_dir=str(SUB / "curves"), num_workers=2)
    return dataclasses.replace(c, **kw)


IDENTITY_FIELDS = ("exp_id", "desc", "seed", "save_test_predictions")


def same_recipe(a: Config, b: Config) -> bool:
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    return all(da[k] == db[k] for k in da if k not in IDENTITY_FIELDS)


def alias_run(src: Config, dst: Config, note: str) -> dict:
    """dst có cùng công thức và seed với src đã chạy: sao chép kết quả thay vì train lại (tiết kiệm
    GPU). Ghi rõ `alias_of` trong summary. Không sao chép kết quả test."""
    s_dir, d_dir = TR.run_dir(src), TR.run_dir(dst)
    if (d_dir / "summary.json").exists():
        return json.loads((d_dir / "summary.json").read_text(encoding="utf-8"))
    assert same_recipe(src, dst) and src.seed == dst.seed
    d_dir.mkdir(parents=True, exist_ok=True)
    for f in s_dir.iterdir():
        if not f.name.startswith("test_") and f.name != "summary.json":
            shutil.copy2(f, d_dir / f.name)
    summ = json.loads((s_dir / "summary.json").read_text(encoding="utf-8"))
    summ.update(exp_id=dst.exp_id, desc=dst.desc, alias_of=f"{src.exp_id}/seed{src.seed}", note=note,
                config=dataclasses.asdict(dst))
    hist = pd.read_csv(d_dir / "history.csv").to_dict("records")
    lr = np.load(d_dir / "lr_steps.npy").tolist()
    TR.plot_curves(hist, Path(dst.curves_dir) / f"{dst.exp_id}_{dst.desc}.png",
                   f"{dst.exp_id} | {dst.backbone} | {dst.desc} | seed {dst.seed} (= {src.exp_id})", lr)
    shutil.copy2(TR.pred_path(src, "val"), TR.pred_path(dst, "val"))
    (d_dir / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summ


def parse_combo(s: str | None) -> dict:
    return TR.parse_overrides(s.split()) if s else {}


# --------------------------------------------------------------------------- #
# Bước 0
# --------------------------------------------------------------------------- #
def stage_eda():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tr, va, te = D.load_split(DATA / "labels", 0)
    info = D.check_split(tr, va, te, DATA / "images")
    pc = info["per_class"]
    paper = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
    pc["paper_table1"] = paper
    pc["diff_vs_paper"] = pc["total"] - pc["paper_table1"]
    pc.to_csv(TABLES / "eda_per_class.csv")
    json.dump({k: v for k, v in info.items() if k != "per_class"} |
              {"imbalance_ratio_max_min": float(pc["total"].max() / pc["total"].min())},
              open(TABLES / "split_check.json", "w", encoding="utf-8"), indent=2)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(9)
    for i, s in enumerate(("train", "val", "test")):
        ax.bar(x + (i - 1) * 0.27, pc[s], width=0.27, label=s)
    ax.set_xticks(x, D.CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("số ảnh"); ax.set_yscale("log")
    ax.set_title("DeepWeeds fold 0: số ảnh theo lớp (trục log)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIGS / "eda_class_distribution.png", dpi=110); plt.close(fig)

    labels = pd.read_csv(DATA / "labels" / "labels.csv")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(9, 4, figsize=(8, 18))
    for c in range(9):
        files = labels[labels.Label == c].Filename.to_numpy()
        for j, f in enumerate(rng.choice(files, 4, replace=False)):
            axes[c, j].imshow(Image.open(DATA / "images" / f)); axes[c, j].axis("off")
        axes[c, 0].set_title(D.CLASS_NAMES[c], loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "eda_samples.png", dpi=80); plt.close(fig)

    sizes = {Image.open(DATA / "images" / f).size for f in labels.Filename.sample(min(300, len(labels)), random_state=0)}
    print("kích thước ảnh (mẫu 300):", sizes)
    D.get_cache(DATA / "images", DATA / "cache.npy")  # tạo cache .npy cho các stage sau
    print(pc.to_string())


def stage_sanity(backbone: str = "resnet50"):
    """Kiểm tra pipeline (GUIDE 1.3): loss ban đầu ≈ ln 9, overfit 1 batch nhỏ, ảnh sau augmentation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TR.set_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = D.load_split(DATA / "labels", 0)
    cache = D.get_cache(DATA / "images", DATA / "cache.npy")
    loader = D.make_loader(tr, DATA / "images", D.build_transforms(True, 224, "basic"), 64, True,
                           None, 2, cache, seed=0)
    x, y, f = next(iter(loader))
    m = M.build_model(backbone, True, 9).to(dev)
    m.eval()
    with torch.no_grad():
        init_loss = torch.nn.functional.cross_entropy(m(x.to(dev)), y.to(dev)).item()

    xs, ys = x[:16].to(dev), y[:16].to(dev)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    curve = []
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(m(xs), ys)
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(loss.item())
    res = {"backbone": backbone, "initial_loss": init_loss, "ln9": float(np.log(9)),
           "overfit_16_imgs_final_loss": curve[-1], "overfit_curve": curve}
    TABLES.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(TABLES / "sanity_checks.json", "w", encoding="utf-8"), indent=2)
    print(f"loss ban đầu {init_loss:.3f} (ln 9 = {np.log(9):.3f}); overfit 16 ảnh: {curve[0]:.3f} -> {curve[-1]:.4f}")

    FIGS.mkdir(parents=True, exist_ok=True)
    mean, std = torch.tensor(D.IMAGENET_MEAN)[:, None, None], torch.tensor(D.IMAGENET_STD)[:, None, None]
    for aug in ("basic", "trivial"):
        tf = D.build_transforms(True, 224, aug)
        fig, axes = plt.subplots(3, 6, figsize=(13, 7))
        for k, ax in enumerate(axes.flat):
            i = k % 6
            img = tf(cache.get(tr.Filename.iloc[i]))
            ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()); ax.axis("off")
            ax.set_title(D.CLASS_NAMES[tr.Label.iloc[i]], fontsize=8)
        fig.suptitle(f"Ảnh sau augmentation '{aug}' (đã giải chuẩn hoá), 3 lần lấy mẫu cho 6 ảnh")
        fig.tight_layout(); fig.savefig(FIGS / f"aug_check_{aug}.png", dpi=80); plt.close(fig)
    from losses import mix_batch
    xm, (ya, yb, lam) = mix_batch(x[:6], y[:6], 1.0, "cutmix", np.random.default_rng(0))
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    for i, ax in enumerate(axes):
        ax.imshow((xm[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"{D.CLASS_NAMES[ya[i]][:10]}/{D.CLASS_NAMES[yb[i]][:10]}", fontsize=7)
    fig.suptitle(f"CutMix, lam (diện tích thực) = {lam:.3f}")
    fig.tight_layout(); fig.savefig(FIGS / "aug_check_cutmix.png", dpi=80); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(curve); ax.set_xlabel("bước"); ax.set_ylabel("CE loss"); ax.set_yscale("log")
    ax.set_title(f"Overfit 16 ảnh ({backbone})"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "sanity_overfit.png", dpi=100); plt.close(fig)


# --------------------------------------------------------------------------- #
# Bước 1, 2
# --------------------------------------------------------------------------- #
def stage_backbones(only: list[str] | None = None):
    out = []
    for eid, bb in BACKBONES:
        if only and eid not in only:
            continue
        out.append(TR.run(base_cfg(exp_id=eid, backbone=bb, desc=bb, seed=0)))
    return out


def baseline_cfg(backbone: str, seed: int = 0) -> Config:
    return base_cfg(exp_id="T00", backbone=backbone, desc="baseline", seed=seed)


def ensure_t00(backbone: str, seed: int = 0) -> dict:
    t00 = baseline_cfg(backbone, seed)
    src = next((base_cfg(exp_id=e, backbone=b, desc=b, seed=seed) for e, b in BACKBONES if b == backbone), None)
    if seed == 0 and src is not None and (TR.run_dir(src) / "summary.json").exists():
        return alias_run(src, t00, "T00 seed0 trùng cấu hình với lần chạy backbone tương ứng")
    return TR.run(t00)


def stage_training(backbone: str, only: list[str] | None = None):
    ensure_t00(backbone, 0)
    for eid, axis, desc, ov in TRAINING:
        if only and eid not in only:
            continue
        TR.run(base_cfg(exp_id=eid, backbone=backbone, desc=desc, seed=0, **ov))


def stage_combo(backbone: str, combo: dict, exp_id: str = "T10"):
    desc = "combo_" + "_".join(f"{k}-{v}" for k, v in combo.items() if k not in ("mix_alpha",))
    return TR.run(base_cfg(exp_id=exp_id, backbone=backbone, desc=desc, seed=0, **combo))


# --------------------------------------------------------------------------- #
# Bước 3: suy luận (chỉ trên VAL)
# --------------------------------------------------------------------------- #
VIEW_SETS = {
    # tên phương pháp: danh sách view (loader trả ảnh gốc 256 đã chuẩn hoá)
    "I00_1view": ["center"],
    "I01_hflip": ["center", "center_flip"],
    "I02_5crop": ["five_crop"],
    "I02_10crop": ["ten_crop"],
    "I02_3scale": ["full224", "full256", "full288"],
    "I04_res224full": ["full224"],
    "I04_res256": ["full256"],
    "I04_res288": ["full288"],
    "I04_res320": ["full320"],
}
VIEW_COST = {"center": (1, 224), "center_flip": (1, 224), "five_crop": (5, 224), "ten_crop": (10, 224),
             "full224": (1, 224), "full256": (1, 256), "full288": (1, 288), "full320": (1, 320)}


def full_loader(df, cfg: Config):
    cache = D.get_cache(cfg.images_dir, cfg.cache_path)
    return D.make_loader(df, cfg.images_dir, D.build_transforms(False, 256, eval_mode="resize"), 64,
                         False, None, 2, cache)


def view_logits(model, df, cfg, view_names, dev):
    names, y, outs = I.predict_views(model, full_loader(df, cfg), dev, [I.make_view(v) for v in view_names])
    return names, y, dict(zip(view_names, outs))


def method_latency(model, views: list[str], dev, iters=100) -> dict:
    """Độ trễ batch 1, FP32, K lượt forward thật (mỗi view ở đúng độ phân giải của nó)."""
    model = model.to(dev).eval().to(memory_format=torch.channels_last)
    inputs = []
    for v in views:
        k, s = VIEW_COST[v]
        inputs += [torch.randn(1, 3, s, s, device=dev).contiguous(memory_format=torch.channels_last)] * k

    def fn():
        with torch.inference_mode():
            for x in inputs:
                model(x)
    return B.bench(fn, 10, iters, torch.cuda.synchronize if dev.type == "cuda" else None)


def metrics_row(y, probs) -> dict:
    m = compute_metrics(y, probs.argmax(1), probs)
    return {"val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
            "val_f1_chinee": m["f1"][0], "val_f1_snake": m["f1"][7]}


def stage_inference(backbone: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TABLES.mkdir(parents=True, exist_ok=True)
    cfg = baseline_cfg(backbone, 0)
    _, va, _ = D.load_split(cfg.labels_dir, 0)
    model = TR.load_model_from_run(cfg, dev).to(memory_format=torch.channels_last)
    all_views = sorted({v for vs in VIEW_SETS.values() for v in vs})
    names, y, L = view_logits(model, va, cfg, all_views, dev)
    ref = (TR.run_dir(cfg) / "val_filenames.txt").read_text(encoding="utf-8").split("\n")
    assert names == ref, "thứ tự file val không khớp"
    rows, lat_cache = [], {}

    def lat(views):
        key = tuple(views)
        if key not in lat_cache:
            lat_cache[key] = method_latency(model, views, dev) if dev.type == "cuda" else {}
        return lat_cache[key]

    for meth, views in VIEW_SETS.items():
        K = sum(VIEW_COST[v][0] for v in views)
        # nhiều view: so sánh gộp xác suất và gộp logit (I03). 5/10-crop: các crop được gộp
        # trong predict_views bằng trung bình logit.
        spaces = ("prob", "logit") if len(views) > 1 else ("logit" if K > 1 else "-",)
        for space in spaces:
            p = I.aggregate_views([L[v] for v in views], "logit" if space == "-" else space)
            label = f" [{space}]" if len(views) > 1 else (" [logit-avg crops]" if K > 1 else "")
            rows.append({"exp_id": meth.split("_")[0], "method": meth + label, "views": "+".join(views),
                         "aggregate": space, "K_forward": K, "model": f"T00/seed0 ({backbone})",
                         **metrics_row(y, p), **_lat_cols(lat(views))})

    # I03: so sánh gộp xác suất vs logit đã có trong các dòng [prob]/[logit] ở trên.
    # I05: ensemble các backbone (1-view, logit val đã lưu)
    summ = {e: json.loads((TR.run_dir(base_cfg(exp_id=e, seed=0)) / "summary.json").read_text(encoding="utf-8"))
            for e, _ in BACKBONES if (TR.run_dir(base_cfg(exp_id=e, seed=0)) / "summary.json").exists()}
    ranked = sorted(summ, key=lambda e: -summ[e]["val_macro_f1"])
    for k in (2, 3):
        ids = ranked[:k]
        probs = [TR.softmax(np.load(TR.run_dir(base_cfg(exp_id=e, seed=0)) / "val_logits.npy")) for e in ids]
        p = I.ensemble_probs(probs)
        yl = np.load(TR.run_dir(base_cfg(exp_id=ids[0], seed=0)) / "val_labels.npy")
        lats = {}
        if dev.type == "cuda":
            ts = []
            for e in ids:
                mm = M.build_model(summ[e]["backbone"], False).to(dev).eval().to(memory_format=torch.channels_last)
                ts.append(mm)
            x = torch.randn(1, 3, 224, 224, device=dev).contiguous(memory_format=torch.channels_last)

            def fn():
                with torch.inference_mode():
                    for mm in ts:
                        mm(x)
            lats = B.bench(fn, 10, 100, torch.cuda.synchronize)
            del ts
        rows.append({"exp_id": "I05", "method": f"I05_ensemble_top{k} [prob]",
                     "views": "center", "aggregate": "prob", "K_forward": k,
                     "model": "+".join(f"{e}:{summ[e]['backbone']}" for e in ids),
                     **metrics_row(yl, p), **_lat_cols(lats)})

    # I06: trọng số EMA (T08) so với trọng số thường của cùng lần chạy
    t08 = TR.run_dir(base_cfg(exp_id="T08", backbone=backbone, seed=0)) / "summary.json"
    if t08.exists():
        s = json.loads(t08.read_text(encoding="utf-8"))
        rows.append({"exp_id": "I06", "method": "I06_EMA_weights (T08)", "views": "center", "aggregate": "-",
                     "K_forward": 1, "model": "T08/seed0 EMA", "val_macro_f1": s["val_macro_f1"],
                     "val_top1": s["val_top1"], "val_ece": s["val_ece"],
                     "note": f"cùng lần chạy, trọng số thường (epoch tốt nhất): macro-F1 {s.get('val_macro_f1_raw_best')}",
                     **_lat_cols(lat(["center"]))})

    # I07: temperature scaling trên logit 1-view; T khớp trên val; ECE kiểm chéo 2 nửa val
    z0 = L["center"]
    T = I.fit_temperature(z0, y)
    p_before, p_after = TR.softmax(z0), I.apply_temperature(z0, T)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y)); a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    cross = []
    for fit, ev in ((a, b), (b, a)):
        t = I.fit_temperature(z0[fit], y[fit])
        cross.append((compute_metrics(y[ev], z0[ev].argmax(1), TR.softmax(z0[ev]))["ece"],
                      compute_metrics(y[ev], z0[ev].argmax(1), I.apply_temperature(z0[ev], t))["ece"]))
    cross = np.mean(cross, 0)
    calib = {"T": T, "ece_before": metrics_row(y, p_before)["val_ece"], "ece_after": metrics_row(y, p_after)["val_ece"],
             "nll_before": compute_metrics(y, p_before.argmax(1), p_before)["nll"],
             "nll_after": compute_metrics(y, p_after.argmax(1), p_after)["nll"],
             "crossfit_ece_before": float(cross[0]), "crossfit_ece_after": float(cross[1])}
    rows.append({"exp_id": "I07", "method": f"I07_temperature (T={T:.3f})", "views": "center", "aggregate": "-",
                 "K_forward": 1, "model": f"T00/seed0 ({backbone})", **metrics_row(y, p_after),
                 "note": f"ECE val {calib['ece_before']:.4f} -> {calib['ece_after']:.4f}; kiểm chéo 2 nửa val "
                         f"{calib['crossfit_ece_before']:.4f} -> {calib['crossfit_ece_after']:.4f}",
                 **_lat_cols(lat(["center"]))})

    # I08: FP16 / AMP (model đã chọn) và gộp BN (ResNet-50 B01: ConvNeXt/ViT không có BN)
    lat_rows = []
    vl = full_loader(va, cfg)
    for dtype in ("fp32", "amp", "fp16"):
        mm = model if dtype != "fp16" else __import__("copy").deepcopy(model).half()
        if dtype != "fp32":
            if dtype == "fp16":
                _, _, outs = _predict_half(mm, vl, dev)
            else:
                _, _, outs = I.predict_views(mm, vl, dev, [I.make_view("center")], amp=True)
                outs = outs[0]
            p = TR.softmax(outs)
            rows.append({"exp_id": "I08", "method": f"I08_{dtype}", "views": "center", "aggregate": "-",
                         "K_forward": 1, "model": f"T00/seed0 ({backbone})", **metrics_row(y, p),
                         **_lat_cols(B.latency_report(model, 1, 224, dtype, iters=100) if dev.type == "cuda" else {}),
                         "note": f"max |Δlogit| vs FP32 = {np.abs(outs - z0_fp32(model, vl, dev)).max():.2e}"})
        for bs in (1, 32):
            if dev.type == "cuda":
                lat_rows.append({"config": f"T00 {backbone}", **B.latency_report(model, bs, 224, dtype, iters=100)})
    bb_of = dict(BACKBONES)
    for e in ("B01", "B04"):  # ResNet-50 và EfficientNet-B0: hai mạng có BatchNorm
        if e not in bb_of:
            continue
        bb = bb_of[e]
        c = base_cfg(exp_id=e, backbone=bb, seed=0)
        if not (TR.run_dir(c) / "best.pt").exists():
            continue
        m0 = TR.load_model_from_run(c, dev)
        fused = I.fuse_conv_bn(m0, torch.randn(2, 3, 224, 224, device=dev))
        _, yb, ob = I.predict_views(fused, vl, dev, [I.make_view("center")], amp=False)
        pb = TR.softmax(ob[0])
        _, _, o0 = I.predict_views(m0, vl, dev, [I.make_view("center")], amp=False)
        rows.append({"exp_id": "I08", "method": f"I08_fuseBN ({bb})", "views": "center", "aggregate": "-",
                     "K_forward": 1, "model": f"{e}/seed0 ({bb})", **metrics_row(yb, pb),
                     "note": f"gộp {fused.n_fused} cặp conv+BN; max |Δlogit| = {np.abs(ob[0] - o0[0]).max():.2e}; "
                             f"macro-F1 trước gộp {metrics_row(yb, TR.softmax(o0[0]))['val_macro_f1']:.4f}",
                     **_lat_cols(B.latency_report(fused, 1, 224, 'fp32', iters=100) if dev.type == "cuda" else {})})
        if dev.type == "cuda":
            for mm, fz in ((m0, False), (fused, True)):
                for bs in (1, 32):
                    lat_rows.append({"config": f"{e} {bb}", **B.latency_report(mm, bs, 224, "fp32", iters=100, bn_fused=fz)})
        del m0, fused

    df = pd.DataFrame(rows)
    base = df[df.method == "I00_1view"].iloc[0]
    df["delta_f1_vs_I00"] = df["val_macro_f1"] - base["val_macro_f1"]
    if "lat_p50_ms" in df:
        df["rel_cost_vs_I00"] = df["lat_p50_ms"] / base["lat_p50_ms"]
    df.to_csv(TABLES / "inference.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(TABLES / "latency.csv", index=False)
    json.dump(calib, open(TABLES / "calibration_val.json", "w", encoding="utf-8"), indent=2)
    # chọn phương pháp suy luận cho chung kết: macro-F1 val cao nhất trong các phương pháp 1 model
    cand = df[df.exp_id.isin(["I00", "I01", "I02", "I04"])].sort_values(["val_macro_f1", "K_forward"],
                                                                         ascending=[False, True])
    choice = cand.iloc[0]
    sel = {"method": choice["method"], "views": choice["views"].split("+"), "aggregate": choice["aggregate"],
           "val_macro_f1": float(choice["val_macro_f1"]), "I00_val_macro_f1": float(base["val_macro_f1"]),
           "rule": "macro-F1 val cao nhất trong I00/I01/I02/I04 trên model T00 seed0; hoà thì ít forward hơn"}
    json.dump(sel, open(TABLES / "inference_choice.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(df.to_string())
    print("Chọn:", sel)


_FP32_CACHE = {}


def z0_fp32(model, loader, dev):
    k = id(model)
    if k not in _FP32_CACHE:
        _, _, o = I.predict_views(model, loader, dev, [I.make_view("center")], amp=False)
        _FP32_CACHE[k] = o[0]
    return _FP32_CACHE[k]


@torch.inference_mode()
def _predict_half(model, loader, dev):
    model.eval()
    outs, ys, names = [], [], []
    view = I.make_view("center")
    for x, y, f in loader:
        x = view(x.to(dev)).half().contiguous(memory_format=torch.channels_last)
        outs.append(model(x).float().cpu())
        ys.append(torch.as_tensor(y)); names += list(f)
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy()


def _lat_cols(r: dict) -> dict:
    if not r:
        return {}
    return {"lat_p50_ms": r["p50"], "lat_p95_ms": r["p95"], "lat_p99_ms": r["p99"],
            "throughput_img_s_b1": 1000.0 / r["p50"]}


# --------------------------------------------------------------------------- #
# Bước 4: chung kết (>= 3 seed), test MỘT lần mỗi seed
# --------------------------------------------------------------------------- #
def stage_final(backbone: str, combo: dict, seeds=(0, 1, 2), method: dict | None = None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = method or json.loads((TABLES / "inference_choice.json").read_text(encoding="utf-8"))
    views, space = method["views"], method["aggregate"]
    if space not in ("prob", "logit"):
        space = "prob"
    desc = "final"
    combo_src = base_cfg(exp_id="T10", backbone=backbone, seed=0, **combo)
    rows = []
    for seed in seeds:
        # mốc T00 (+ I00): 1-view, không temperature scaling
        t00 = baseline_cfg(backbone, seed)
        ensure_t00(backbone, seed)
        if not (TR.run_dir(t00) / "test_logits.npy").exists():
            TR.evaluate_split(t00, "test")
        # chung kết F01: công thức kết hợp + suy luận đã chọn + temperature scaling (T khớp trên val)
        f01 = base_cfg(exp_id="F01", backbone=backbone, desc=desc, seed=seed, **combo)
        if seed == 0 and (TR.run_dir(combo_src) / "summary.json").exists() and same_recipe(combo_src, f01):
            alias_run(combo_src, f01, "F01 seed0 trùng cấu hình với T10 (kết hợp) seed0")
        else:
            TR.run(f01)
        rd = TR.run_dir(f01)
        out = rd / "final_probs.npz"
        if out.exists():
            continue
        model = TR.load_model_from_run(f01, dev).to(memory_format=torch.channels_last)
        tr, va, te = D.load_split(f01.labels_dir, 0)
        nv, yv, Lv = view_logits(model, va, f01, views, dev)
        zv = _agg_logits(Lv, views, space)
        T = I.fit_temperature(zv, yv)                      # chỉ VAL
        assert not (rd / "test_logits.npy").exists(), "test đã chạy cho seed này"
        nt, yt, Lt = view_logits(model, te, f01, views, dev)  # test: đúng một lần
        zt = _agg_logits(Lt, views, space)
        np.save(rd / "test_logits.npy", zt)
        p_val, p_test = I.apply_temperature(zv, T), I.apply_temperature(zt, T)
        save_predictions(TR.pred_path(f01, "val"), nv, yv, p_val)
        save_predictions(TR.pred_path(f01, "test"), nt, yt, p_test)
        save_predictions(SUB / "predictions" / f"F01uncal_seed{seed}_test.csv", nt, yt, TR.softmax(zt))
        save_predictions(SUB / "predictions" / f"F01uncal_seed{seed}_val.csv", nv, yv, TR.softmax(zv))
        np.savez(out, p_val=p_val, p_test=p_test, T=T)
        rows.append({"seed": seed, "T": T})
        del model
    json.dump({"method": method, "combo": combo, "backbone": backbone, "temps": rows},
              open(TABLES / "final_setup.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)


def _agg_logits(L: dict, views: list[str], space: str) -> np.ndarray:
    """Logit 'hiệu dụng' sau gộp view: logit -> trung bình logit; prob -> log của trung bình xác suất
    (để temperature scaling áp dụng được trên cùng một đại lượng)."""
    if len(views) == 1:
        return L[views[0]]
    if space == "logit":
        return np.mean([L[v] for v in views], 0)
    return np.log(np.clip(I.aggregate_views([L[v] for v in views], "prob"), 1e-12, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["eda", "sanity", "backbones", "training", "combo", "inference", "final"])
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--only", default="")
    ap.add_argument("--combo", default="")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    only = [s for s in a.only.split(",") if s] or None
    t0 = time.time()
    if a.stage == "eda":
        stage_eda()
    elif a.stage == "sanity":
        stage_sanity(a.backbone)
    elif a.stage == "backbones":
        stage_backbones(only)
    elif a.stage == "training":
        stage_training(a.backbone, only)
    elif a.stage == "combo":
        stage_combo(a.backbone, parse_combo(a.combo))
    elif a.stage == "inference":
        stage_inference(a.backbone)
    elif a.stage == "final":
        stage_final(a.backbone, parse_combo(a.combo), tuple(int(s) for s in a.seeds.split(",")))
    print(f"stage {a.stage} xong sau {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
