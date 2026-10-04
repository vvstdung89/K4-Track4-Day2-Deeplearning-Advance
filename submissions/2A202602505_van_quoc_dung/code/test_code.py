"""test_code.py - kiểm tra tự viết cho các phần dễ sai (RUBRIC mục C, H). Chạy được trên CPU:

    python -m unittest test_code -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark as B
import dataset as D
import inference as I
import losses as L
import model as M
import train as TR


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.z = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = L.FocalLoss(gamma=0.0)(self.z, self.y)
        ce = F.cross_entropy(self.z, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(L.FocalLoss(2.0)(self.z, self.y).item(), F.cross_entropy(self.z, self.y).item())

    def test_label_smoothing_matches_torch(self):
        for eps in (0.0, 0.1, 0.3):
            ours = L.LabelSmoothingCE(eps)(self.z, self.y)
            ref = F.cross_entropy(self.z, self.y, label_smoothing=eps)
            self.assertLess(abs(ours.item() - ref.item()), 1e-6)

    def test_class_weights(self):
        counts = [100, 200, 400]
        w = L.class_weights(counts, 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertAlmostEqual((w[0] / w[2]).item(), 4.0, places=5)
        wcb = L.class_weights(counts, 0.999)
        self.assertAlmostEqual(wcb.sum().item(), 3.0, places=5)
        self.assertTrue(wcb[0] > wcb[1] > wcb[2])

    def test_cutmix_lambda_is_true_area(self):
        rng = np.random.default_rng(0)
        x = torch.zeros(8, 3, 32, 32)
        x[4:] = 1.0  # nửa batch toàn 1
        for _ in range(50):
            xm, (ya, yb, lam) = L.mix_batch(x, torch.arange(8), 1.0, "cutmix", rng)
            # với ảnh nền 0, tỉ lệ pixel được dán = 1 - lam nếu ảnh nguồn là 1
            perm_src = yb
            for i in range(8):
                frac_pasted = (xm[i] != x[i]).float().mean().item()
                if (x[i, 0, 0, 0] != x[perm_src[i], 0, 0, 0]):
                    self.assertAlmostEqual(frac_pasted, 1 - lam, places=5)
            self.assertTrue(0.0 <= lam <= 1.0)

    def test_mixup_and_mixed_loss(self):
        rng = np.random.default_rng(1)
        x = torch.randn(8, 3, 4, 4)
        y = torch.arange(8)
        xm, (ya, yb, lam) = L.mix_batch(x, y, 0.4, "mixup", rng)
        perm = [int(torch.nonzero(y == b)[0]) for b in yb]
        torch.testing.assert_close(xm, lam * x + (1 - lam) * x[perm])
        ce = nn.CrossEntropyLoss()
        z = torch.randn(8, 9)
        ml = L.mixed_loss(ce, z, (ya, yb, lam))
        self.assertAlmostEqual(ml.item(), (lam * ce(z, ya) + (1 - lam) * ce(z, yb)).item(), places=5)


class TestModel(unittest.TestCase):
    def test_param_groups_no_decay_on_norm_bias(self):
        m = M.build_model("resnet18", pretrained=False)
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        for g in groups:
            for p in g["params"]:
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0)
        head = {id(p) for p in m.get_classifier().parameters()}
        for g in groups:
            is_head = g["name"].startswith("head")
            self.assertEqual(g["lr"], 1e-3 if is_head else 1e-4)
            self.assertTrue(all((id(p) in head) == is_head for p in g["params"]))
        n = sum(p.numel() for g in groups for p in g["params"])
        self.assertEqual(n, sum(p.numel() for p in m.parameters()))

    def test_frozen_keeps_bn_eval_and_only_head_trains(self):
        m = M.build_model("resnet18", pretrained=False, init="finetune")
        M.freeze_backbone(m)
        M.set_train_mode(m)
        bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm2d)]
        self.assertTrue(all(not b.training for b in bns))
        self.assertTrue(m.get_classifier().training)
        trainable = [p for p in m.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 2)  # fc.weight, fc.bias
        before = bns[0].running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        torch.testing.assert_close(bns[0].running_mean, before)

    def test_count_gmacs_resnet50(self):
        m = M.build_model("resnet50", pretrained=False)
        self.assertAlmostEqual(M.count_gmacs(m, 224), 4.1, delta=0.15)
        self.assertAlmostEqual(M.count_params(m), 23.5, delta=0.2)  # head 9 lớp thay cho 1000 lớp

    def test_initial_loss_near_ln9(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False).eval()
        with torch.no_grad():
            z = m(torch.randn(32, 3, 64, 64))
        loss = F.cross_entropy(z, torch.randint(0, 9, (32,))).item()
        self.assertLess(abs(loss - math.log(9)), 0.5)


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        for name in ("resnet18", "efficientnet_b0", "mobilenetv3_large_100"):
            torch.manual_seed(0)
            m = M.build_model(name, pretrained=False)
            # thống kê BN khác mặc định để phép gộp có ý nghĩa
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.running_mean.uniform_(-0.5, 0.5)
                    mod.running_var.uniform_(0.5, 2.0)
                    mod.weight.data.uniform_(0.5, 1.5)
                    mod.bias.data.uniform_(-0.2, 0.2)
            m.eval()
            x = torch.randn(2, 3, 96, 96)
            fused = I.fuse_conv_bn(m, x)
            self.assertGreater(fused.n_fused, 10, name)
            left = sum(isinstance(mod, nn.BatchNorm2d) for mod in fused.modules())
            self.assertEqual(left, 0, f"{name}: còn {left} BN chưa gộp")
            self.assertLess(fused.max_abs_err, 1e-3, name)

    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        n, k = 4000, 9
        true = rng.normal(size=(n, k)) * 2
        y = np.array([rng.choice(k, p=p) for p in I.softmax_np(true)])
        T = I.fit_temperature(true * 2.5, y)  # logit bị phóng 2.5 lần -> T ≈ 2.5
        self.assertAlmostEqual(T, 2.5, delta=0.25)
        p = I.apply_temperature(true * 2.5, T)
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-9)
        np.testing.assert_array_equal(p.argmax(1), true.argmax(1))  # accuracy không đổi

    def test_aggregate_and_ensemble(self):
        a = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 0.0], [0.0, 3.0]])
        pp = I.aggregate_views([a, b], "prob")
        pl = I.aggregate_views([a, b], "logit")
        np.testing.assert_allclose(pp, (I.softmax_np(a) + I.softmax_np(b)) / 2)
        np.testing.assert_allclose(pl, I.softmax_np((a + b) / 2))
        np.testing.assert_allclose(I.ensemble_probs([I.softmax_np(a), I.softmax_np(b)]), pp)

    def test_views(self):
        x = torch.arange(2 * 3 * 256 * 256, dtype=torch.float32).reshape(2, 3, 256, 256)
        self.assertEqual(I.view_center(x).shape[-1], 224)
        torch.testing.assert_close(I.view_hflip(I.view_hflip(x)), x)
        crops = I.views_multicrop(x, 224, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape[-2:] == (224, 224) for c in crops))
        torch.testing.assert_close(crops[4], I.view_center(x))
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, [224, 288])], [224, 288])

    def test_center_view_matches_eval_transform(self):
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        a = D.build_transforms(False, 224)(img)
        b = I.view_center(D.build_transforms(False, 256, eval_mode="resize")(img)[None])[0]
        torch.testing.assert_close(a, b)


class TestTrain(unittest.TestCase):
    def test_lr_schedule(self):
        f = [TR.lr_factor(s, 100, 10) for s in range(100)]
        self.assertAlmostEqual(f[0], 0.1)
        self.assertAlmostEqual(f[9], 1.0)
        self.assertAlmostEqual(f[10], 1.0)
        self.assertLess(f[-1], 0.01)
        self.assertTrue(all(a >= b for a, b in zip(f[10:], f[11:])))

    def test_parse_overrides(self):
        d = TR.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false",
                                "label_smoothing=0.1", "mix=cutmix", "class_weight_beta=0.999"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False,
                             "label_smoothing": 0.1, "mix": "cutmix", "class_weight_beta": 0.999})
        with self.assertRaises(KeyError):
            TR.parse_overrides(["nope=1"])

    def test_ema(self):
        m = nn.Linear(2, 2)
        ema = TR.EMA(m, 0.5)
        w0 = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        d = min(0.5, 2 / 11)
        torch.testing.assert_close(ema.module.weight, d * w0 + (1 - d) * m.weight)

    def test_overfit_one_batch(self):
        torch.manual_seed(0)
        m = M.build_model("resnet18", pretrained=False)
        x, y = torch.randn(8, 3, 64, 64), torch.arange(8) % 9
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        m.train()
        for _ in range(60):
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        self.assertLess(loss.item(), 0.05)


class TestDataAndBench(unittest.TestCase):
    def test_transforms_shapes(self):
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        for aug in ("basic", "color", "trivial", "randaug", "flipv"):
            out = D.build_transforms(True, 224, aug)(img)
            self.assertEqual(tuple(out.shape), (3, 224, 224))
            self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(D.build_transforms(False, 288)(img).shape), (3, 288, 288))

    def test_bench_percentiles(self):
        r = B.bench(lambda: sum(range(1000)), warmup=3, iters=60)
        self.assertEqual(r["n"], 60)
        self.assertTrue(r["p50"] <= r["p95"] <= r["p99"])


if __name__ == "__main__":
    unittest.main()
