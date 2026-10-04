# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

*Fold 0 chia sẵn · Google Colab T4 · mọi con số truy ngược được tới `results.xlsx`, `runs/*/summary.json`,
`tables/*` và kết quả `eval.py` trong `eval_out/`.*

## 1. Tóm tắt

Bài toán: phân loại 9 lớp ảnh cỏ dại DeepWeeds (17.509 ảnh, `Negatives` chiếm 52%), chỉ số chính macro-F1.
Đã chạy **5 backbone** (ResNet-50, ConvNeXt-T, DeiT-S, EfficientNet-B0, MobileNetV3-L), **9 ablation** trên **5 trục**
công thức huấn luyện (khởi tạo, augmentation, loss, cân bằng mẫu, EMA) cộng 1 kết hợp, **8 nhóm phương pháp suy luận**
(TTA lật, multi-crop/multi-scale, gộp xác suất/logit, độ phân giải kiểm tra, ensemble, EMA, temperature scaling,
FP16/AMP/gộp BN) với độ trễ p50/p95/p99 đo đúng cách, và một chẩn đoán thống kê BatchNorm. Mọi lựa chọn dựa trên val;
test chạy một lần mỗi seed.
**Cấu hình tốt nhất:** ConvNeXt-T + CutMix, suy luận 1 lượt ở 288 px trên ảnh đầy đủ, temperature scaling →
**macro-F1 test 0,9750 ± 0,0026, top-1 97,92% ± 0,20%** (3 seed), recall Chinee apple / Snake weed 95,1% / 94,6%,
ECE 0,007, p95 17,7 ms (batch 1, T4). Mốc T00 + 1-view: 0,9706 ± 0,0020 → cải thiện **+0,0044** (1,7 lần std).
**Kết luận chính:** khởi tạo tiền huấn luyện và backbone quyết định gần như toàn bộ chất lượng; độ phân giải kiểm tra
là cải thiện rẻ nhất; các thay đổi công thức (augmentation, loss, EMA) chỉ cỡ nhiễu seed và không cộng dồn.

## 2. Dữ liệu và thiết lập

**Dataset.** DeepWeeds (Olsen et al., 2019), 17.509 ảnh RGB 256×256, 9 lớp. Ảnh tải từ Zenodo (MD5
`b7b30f96d466fba86016aa5a26606e0f` khớp), nhãn và file chia từ GitHub của tác giả. Dùng **fold 0** nguyên bản
(`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`), không sửa, không lọc.

**Kiểm tra chia dữ liệu** (`tables/split_check.json`, `tables/eda_per_class.csv`):

| | train | val | test | tổng |
|---|---|---|---|---|
| Số ảnh | 10.501 | 3.501 | 3.507 | 17.509 |
| Tỉ lệ | 59,97% | 20,00% | 20,03% | |

Giao từng cặp (train∩val, train∩test, val∩test) theo tên file đều **rỗng**; hợp ba tập đúng **17.509** ảnh;
**không thiếu file** nào trong thư mục ảnh. Mọi ảnh 256×256 RGB.

**EDA** (`figures/eda_class_distribution.png`, `figures/eda_samples.png`). Số ảnh đếm được khớp Table 1 của bài báo ở
7/9 lớp; lệch 1 ảnh ở Chinee apple (1.126 so với 1.125) và Lantana (1.063 so với 1.064), tổng vẫn 17.509 (có lẽ một ảnh
được gán lại nhãn trong bản phát hành). `Negatives` chiếm 9.106 ảnh (52,0%), các loài cỏ 1.009–1.126 ảnh; tỉ lệ lớp
lớn nhất / nhỏ nhất ≈ 9,0. Vì vậy top-1 bị `Negatives` kéo cao, **macro-F1 (9 lớp) là chỉ số chính**. Nhìn ảnh mẫu:
Chinee apple và Snake weed đều là lá xanh nhỏ trên nền cỏ/đất lẫn lộn, khó phân biệt bằng mắt; `Negatives` rất đa dạng
(cỏ khác, đất, đá, cây bụi), nên là lớp "phần còn lại" khó mô hình hoá.

**Kiểm tra pipeline trước khi chạy thật** (`tables/sanity_checks.json`, `figures/sanity_overfit.png`,
`figures/aug_check_*.png`): loss CE ban đầu của ResNet-50 với head mới = **2,187** (−ln(1/9) = 2,197); overfit 16 ảnh
trong 60 bước: loss 2,2 → **1,4·10⁻⁴**; ảnh sau augmentation (đã giải chuẩn hoá) và CutMix hiển thị đúng ảnh/nhãn.
Ngoài ra 21 kiểm tra tự viết (`code/test_code.py`) đều qua: focal γ=0 ≡ CE (sai số < 1e-6), label smoothing khớp
`torch`, λ của CutMix bằng diện tích thực, mixup trộn cả nhãn, weight decay = 0 cho norm/bias, BN đóng băng không cập nhật
thống kê, gộp BN sai số ~1e-6, T của temperature scaling khôi phục đúng giá trị đã biết, lịch LR warmup + cosine.

**Công thức nền T00** (giống nhau cho mọi backbone): trọng số ImageNet của timm, head mới 9 lớp, tinh chỉnh toàn bộ;
train `RandomResizedCrop(224)` + lật ngang; val/test `CenterCrop(224)` từ ảnh gốc 256 + chuẩn hoá ImageNet; AdamW,
LR backbone 1e-4 / head 1e-3; weight decay 0,05 (0 cho norm và bias, 4 nhóm tham số); warmup tuyến tính 1 epoch rồi
cosine về 0, cập nhật theo bước; CE; batch 64; AMP FP16; channels_last; **10 epoch** (ngân sách Colab miễn phí, GUIDE
cho phép 10–15); chọn checkpoint theo macro-F1 val cao nhất (hoà lấy epoch sớm hơn). Val loss luôn là CE thường để
so được giữa các loss.

**Phần cứng và phiên bản.** Google Colab miễn phí: Tesla T4 15 GB, 2 vCPU, 12 GB RAM; torch 2.11.0+cu130,
torchvision 0.26.0, timm 1.0.29, numpy 2.1.3. Ảnh được giải mã một lần vào mảng uint8 (memmap `.npy`) để 2 vCPU không
nghẽn giải mã JPEG.

**Seed.** Sàng backbone và ablation: seed 0 (cùng seed cho mọi cấu hình). Chung kết F01 và mốc T00: seed 0, 1, 2.
`cudnn.benchmark=True` nên tái lập ở mức xấp xỉ.

**Quy tắc val/test.** Mọi lựa chọn (backbone, công thức, phương pháp suy luận, checkpoint, nhiệt độ T) chỉ dùng val.
Test chỉ được đánh giá ở Bước 4, một lần cho mỗi seed; `train.evaluate_split` và `stage_final` từ chối chạy lại nếu
logit test của lần chạy đó đã tồn tại.

## 3. So sánh backbone (Bước 1)

Cùng công thức nền T00, cùng split, seed 0, 10 epoch (sheet `Backbones`, `curves/B0x_*.png`,
`figures/backbones_tradeoff.png`). Độ trễ ở đây là đo sơ bộ (batch 1, FP32, 50 lần, T4); đo kỹ ở mục 5.

| exp_id | Backbone (tag timm) | #params (M) | GMAC | macro-F1 val | top-1 val | F1 Chinee / Snake | s/epoch | trễ b1 p50 (ms) |
|---|---|---|---|---|---|---|---|---|
| B01 | ResNet-50 (`resnet50.a1_in1k`) | 23,5 | 4,09 | 0,7850 | 0,8446 | 0,604 / 0,715 | 34,3 | 7,7 |
| **B02** | **ConvNeXt-T (`convnext_tiny.in12k_ft_in1k`)** | 27,8 | 4,45 | **0,9676** | **0,9760** | 0,932 / 0,927 | 48,2 | 6,1 |
| B03 | DeiT-S (`deit_small_patch16_224.fb_in1k`) | 21,7 | 4,60 | 0,9460 | 0,9617 | 0,873 / 0,889 | 32,2 | 5,0 |
| B04 | EfficientNet-B0 (`efficientnet_b0.ra_in1k`) | 4,0 | 0,38 | 0,7238 | 0,8021 | 0,619 / 0,691 | 30,5 | 9,1 |
| B05 | MobileNetV3-L (`mobilenetv3_large_100.ra_in1k`) | 4,2 | 0,22 | 0,5997 | 0,7207 | 0,624 / 0,490 | 23,9 | 7,2 |

(#params tính với head 9 lớp; GMAC đếm bằng `torch.utils.flop_counter`, MAC = FLOP/2.)

**Nhận xét.**

- Khoảng cách rất lớn và **không theo thứ hạng ImageNet**: hai mạng dùng LayerNorm (ConvNeXt-T, DeiT-S) đạt
  0,95–0,97, ba mạng dùng BatchNorm (ResNet-50, EfficientNet-B0, MobileNetV3) chỉ 0,60–0,79, dù loss train của chúng
  vẫn giảm đều (đường cong `B01/B04/B05`: train loss 0,26–0,41 nhưng val loss 0,46–0,84, khoảng cách train/val lớn
  ngay từ đầu, không phải quá khớp muộn).
- **Chẩn đoán** (`tables/bn_recalibration.csv`, stage `bnrecal`): giả thuyết là thống kê BN được học trên ảnh
  `RandomResizedCrop` (scale 0,08–1, tức ảnh bị phóng to mạnh) không khớp ảnh center-crop lúc đánh giá. Ước lượng lại
  running mean/var của mọi lớp BN bằng ảnh **train** đi qua transform đánh giá (không gradient, không dùng val/test),
  giữ nguyên trọng số:

  | Backbone | macro-F1 val trước | sau khi ước lượng lại BN | Δ |
  |---|---|---|---|
  | ResNet-50 | 0,7856 | 0,8221 | +0,037 |
  | EfficientNet-B0 | 0,7232 | 0,8583 | **+0,135** |
  | MobileNetV3-L | 0,5998 | 0,8704 | **+0,271** |

  Giả thuyết được xác nhận một phần lớn với hai mạng nhẹ: phần lớn khoảng cách đến từ **lệch thống kê BN giữa
  train-augmentation và eval**, không phải do kiến trúc kém. Với ResNet-50 (tag `a1_in1k`, huấn luyện bằng công thức
  BCE + LAMB của "ResNet strikes back"), phần còn lại có lẽ do trọng số này cần LR/epoch lớn hơn để tinh chỉnh; 10 epoch
  với LR 1e-4 là chưa đủ (val F1 vẫn đang tăng chậm ở epoch 10). Đây đúng là câu hỏi 1 của GUIDE mục 9: chênh lệch
  ResNet-50 ↔ ConvNeXt-T ở đây **đến từ công thức (cả của trọng số tiền huấn luyện lẫn của ta), không chỉ kiến trúc**.
  Không sửa công thức nền cho riêng mạng BN để giữ so sánh công bằng (N1); ghi nhận là hạn chế.
- **FLOPs không phải độ trễ** (slide trang 43): EfficientNet-B0/MobileNetV3 ít hơn ConvNeXt-T 10–20 lần về GMAC nhưng ở
  batch 1 trên T4 lại không nhanh hơn (7–9 ms so với 6 ms): với ảnh nhỏ, GPU bị giới hạn bởi số kernel launch/độ sâu
  mạng chứ không bởi phép nhân. Ở batch 32 thì EfficientNet-B0 đạt 415 ảnh/s so với 113 của ConvNeXt-T (sheet Latency).
  Thời gian train/epoch cũng không tỉ lệ với GMAC (bị chặn bởi nạp dữ liệu trên 2 vCPU).
- **Chọn backbone đi tiếp: ConvNeXt-T.** macro-F1 val cao nhất (0,9676, hơn DeiT-S 0,022 — chênh lệch lớn hơn nhiều
  so với nhiễu seed ước lượng ở mục 4), F1 hai lớp khó cao nhất, độ trễ batch 1 ~6 ms (đủ thời gian thực), chỉ chậm hơn
  DeiT-S ~50% khi train. DeiT-S là lựa chọn dự phòng nếu cần train nhanh hơn.

## 4. Công thức huấn luyện (Bước 2, ConvNeXt-T)

Mỗi lần chạy khác T00 đúng **một** yếu tố, seed 0, 10 epoch (sheet `Training`, `figures/training_ablation.png`,
`curves/T0x_*.png`). T00 seed 0 trùng cấu hình với B02 nên dùng lại kết quả (ghi `alias_of`). **Nhiễu seed** của T00
đo từ 3 seed (0, 1, 2) ở Bước 4: std macro-F1 val = **0,0011 (val, 3 seed); độ lệch chuẩn của *hiệu* giữa hai lần chạy 1 seed ≈ √2 · 0,0011 ≈ 0,0015**.

| exp_id | Trục | Khác T00 ở điểm | macro-F1 val | Δ so với T00 | Δ / std | F1 Chinee / Snake |
|---|---|---|---|---|---|---|
| T00 | — | (công thức nền) | 0,9676 | — | — | 0,932 / 0,927 |
| T01 | A. Khởi tạo | từ đầu (không tiền huấn luyện) | 0,2999 | −0,668 | −611 | 0,210 / 0,259 |
| T02 | A. Khởi tạo | đóng băng backbone, chỉ train head | 0,8433 | −0,124 | −114 | 0,817 / 0,774 |
| T03 | B. Augmentation | + TrivialAugmentWide | 0,9694 | +0,0018 | +1,7 | 0,942 / 0,929 |
| T04 | B. Augmentation | + CutMix (α = 1) | 0,9703 | +0,0027 | +2,5 | 0,956 / 0,942 |
| T05 | C. Loss | label smoothing ε = 0,1 | 0,9687 | +0,0011 | +1,1 | 0,938 / 0,934 |
| T06 | C. Loss | focal γ = 2 | 0,9694 | +0,0018 | +1,7 | 0,943 / 0,931 |
| T07 | C. Loss | CE trọng số 1/n_c (chuẩn hoá mean 1) | 0,9641 | −0,0035 | −3,2 | 0,949 / 0,915 |
| T08 | F. Chính quy hoá | EMA trọng số (d = 0,999) | 0,9701 | +0,0025 | +2,3 | 0,940 / 0,937 |
| T09 | D. Cân bằng mẫu | WeightedRandomSampler 1/n_c | 0,9679 | +0,0003 | +0,3 | 0,941 / 0,936 |
| T10 | Kết hợp | CutMix + focal + EMA | 0,9651 | −0,0025 | −2,3 | 0,940 / 0,912 |

**Nhận xét.**

- **Khởi tạo là yếu tố lớn nhất, áp đảo mọi yếu tố khác.** Từ đầu, 10 epoch, ConvNeXt-T chỉ đạt 0,30 (gần như đoán
  `Negatives` và vài lớp dễ): ~10k ảnh và 10 epoch không đủ để học đặc trưng từ đầu cho một mạng 28M tham số. Đóng băng
  backbone (linear probe trên đặc trưng ImageNet) đạt 0,84: đặc trưng ImageNet hữu ích nhưng chưa đủ cho bài toán cỏ dại
  hạt mịn; tinh chỉnh toàn bộ thêm +0,12. Đường cong T02 phẳng sớm (train loss ≈ val loss ≈ 0,38): thiếu dung lượng,
  không phải quá khớp.
- **Augmentation, loss, EMA, sampler: mọi chênh lệch đều trong khoảng ±0,004.** So với độ lệch chuẩn của hiệu hai lần
  chạy 1 seed (≈ 0,0015): TrivialAugment (+0,0018), label smoothing (+0,0011), focal (+0,0019), sampler cân bằng (+0,0003)
  **không phân biệt được** với T00; CutMix (+0,0027) và EMA (+0,0025) dương cỡ 1,7–1,8 lần nhiễu, CE có trọng số
  (−0,0035) âm cỡ 2,3 lần nhiễu — có xu hướng nhưng chưa chắc chắn với 1 seed. Đáng chú ý: CutMix cho F1 hai lớp khó cao nhất (0,956 / 0,942),
  phù hợp kỳ vọng rằng trộn vùng ảnh buộc mô hình nhìn nhiều phần lá hơn. Cùng chiều, ở chung kết 3 seed CutMix
  cũng làm tăng recall Chinee apple trên test (mục 6).
- **Loss cho lớp hiếm:** CE có trọng số (và sampler cân bằng) không giúp: ở DeepWeeds mỗi loài vẫn có ~600 ảnh train,
  mất cân bằng chủ yếu là `Negatives` ↔ phần còn lại; tăng trọng số các loài làm F1 Snake weed giảm (0,915 so với
  0,927) dù Chinee apple tăng — giả thuyết: mô hình đoán loài nhiều hơn trên ảnh `Negatives` nên precision các loài
  giảm (chưa kiểm chứng riêng). Focal/LS cho kết quả tương đương CE.
- **EMA:** macro-F1 val bằng trọng số EMA (0,9701) gần như bằng trọng số thường tốt nhất của cùng lần chạy (0,9705):
  với lịch cosine về 0, trọng số cuối đã "mượt" sẵn nên EMA không thêm gì đáng kể (I06).
- **Kết hợp (cách tham lam theo trục):** lấy giá trị tốt nhất của mỗi trục có Δ > 0 trên val: B = CutMix, C = focal,
  F = EMA → T10. Kết quả: T10 = 0,9651, **thấp hơn** cả T00 (−0,0025) và từng yếu tố riêng lẻ: các hiệu ứng nhỏ **không cộng dồn mà triệt tiêu**. Hai quan sát: (i) trọng số thường của T10 đạt 0,9683 ở epoch tốt nhất, cao hơn bản EMA 0,9651 — với CutMix + focal mô hình hội tụ chậm hơn nên EMA (d = 0,999, ~1000 bước) kéo về trọng số cũ còn kém; (ii) CutMix (nhãn mềm) và focal (giảm trọng số mẫu dễ) cùng làm giảm độ tự tin, tác dụng chồng lên nhau thay vì bổ sung. Cả hai là giả thuyết, chưa tách riêng được với 1 seed. Vì vậy **cấu hình chung kết dùng công thức tốt nhất trên val là T04 (CutMix một mình)**, không dùng T10.
- Thứ tự tham lam và việc chỉ dùng 1 seed cho ablation là hạn chế đã nêu; kết luận chắc chắn chỉ rút ra ở Bước 4 (3 seed).

## 5. Suy luận (Bước 3, mô hình T00 seed 0, chỉ trên val)

Sheet `Inference` và `Latency`, `figures/inference_tradeoff.png`. Độ trễ: batch 1, FP32, warmup 10 lần, `torch.cuda.synchronize()`
trước và sau, 100 lần đo, Tesla T4, chỉ forward (không tính tiền xử lý, ảnh đã nằm trên GPU), channels_last.
Loader trả ảnh gốc 256 đã chuẩn hoá; các view được tạo trên GPU.

| exp_id | Phương pháp | K forward | macro-F1 val | top-1 val | ECE val | p50 / p95 (ms) | chi phí so với I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view, center crop 224 | 1 | 0,9676 | 0,9760 | 0,0092 | 11,8 / 12,5 | 1,0× |
| I01 | TTA lật ngang, gộp xác suất / logit | 2 | 0,9682 / 0,9679 | 0,9763 / 0,9760 | 0,0097 | 23,9 / 25,0 | 2,0× |
| I02 | 5 crop 224 (trung bình logit) | 5 | 0,9691 | 0,9769 | 0,0089 | 60,2 / 80,8 | 5,1× |
| I02 | 10 crop (5 crop + lật) | 10 | 0,9684 | 0,9766 | 0,0095 | 120,2 / 156,7 | 10,2× |
| I02 | 3 tỉ lệ (ảnh đủ 224/256/288), gộp xác suất / logit | 3 | 0,9745 / 0,9752 | 0,9806 / 0,9809 | 0,0075 / 0,0092 | 43,3 / 59,6 | 3,7× |
| I04 | độ phân giải kiểm tra: ảnh đủ 224 | 1 | 0,9701 | 0,9769 | 0,0125 | 11,8 / 13,6 | 1,0× |
| I04 | ảnh đủ 256 (gốc) | 1 | 0,9707 | 0,9780 | 0,0074 | 13,3 / 15,9 | 1,1× |
| **I04** | **ảnh đủ, resize 288** | 1 | **0,9756** | **0,9806** | 0,0084 | 16,8 / 17,7 | 1,4× |
| I04 | ảnh đủ, resize 320 | 1 | 0,9738 | 0,9789 | 0,0093 | 21,0 / 21,4 | 1,8× |
| I05 | ensemble ConvNeXt-T + DeiT-S | 2 | 0,9694 | 0,9774 | 0,0125 | 27,9 / 37,4 | 2,4× |
| I05 | ensemble top-3 (+ ResNet-50) | 3 | 0,9664 | 0,9751 | 0,0719 | 38,9 / 45,0 | 3,3× |
| I06 | trọng số EMA (T08) | 1 | 0,9701 | 0,9766 | 0,0104 | = I00 | 1,0× |
| I07 | temperature scaling (T = 1,322, khớp trên val) | 1 | 0,9676 | 0,9760 | **0,0057** | = I00 | 1,0× |
| I08 | AMP (autocast FP16) | 1 | 0,9676 | 0,9760 | 0,0092 | 20,4 / 30,0 | 1,7× |
| I08 | FP16 (model.half()) | 1 | 0,9676 | 0,9760 | 0,0092 | 10,3 / 11,3 | 0,9× |
| I08 | gộp BN vào conv (ResNet-50 B01) | 1 | 0,7838 (= trước gộp) | 0,8435 | 0,0227 | 15,2 / 17,8 | — |

**Nhận xét.**

- **Độ phân giải kiểm tra là phương pháp đáng giá nhất**: dùng ảnh đầy đủ thay vì center crop và tăng lên 288 cho +0,008
  macro-F1 với chi phí 1,4×, hơn mọi TTA. Đây đúng hiệu ứng FixRes (slide trang 68): `RandomResizedCrop` lúc train làm
  vật thể trông to hơn so với center crop 224 lúc test; phóng ảnh test lên bù lại. Quá 288 (320) bắt đầu giảm.
  3 tỉ lệ (224/256/288) cho kết quả tương đương 288 đơn lẻ nhưng tốn 3 lần: không đáng.
- **TTA lật / nhiều crop gần như không giúp** (+0,0006 đến +0,0015, dưới nhiễu) mà tốn 2–10 lần: crop góc bỏ mất phần
  ảnh và lật ngang đã có trong augmentation train. **Gộp xác suất vs logit (I03)**: chênh ≤ 0,0007, không phân biệt được;
  chọn gộp xác suất cho lật, logit cho multi-crop.
- **Ensemble** không giúp vì các mô hình còn lại yếu hơn nhiều (DeiT-S 0,946, ResNet-50 0,785) và kém hiệu chuẩn
  (ECE top-3 0,072): trung bình với mô hình kém kéo xuống.
- **Hiệu chuẩn (I07):** T = 1,32 > 1 nghĩa là mô hình hơi quá tự tin; ECE val 0,0092 → 0,0057, NLL 0,0914 → 0,0847,
  accuracy không đổi. Vì T khớp và đo trên cùng tập val, kiểm chéo hai nửa val (khớp nửa này, đo nửa kia) cho mức giảm
  thật thận trọng hơn: 0,0101 → 0,0093.
- **FP16/AMP/gộp BN:** FP16 (`model.half()`) cho macro-F1 giống hệt FP32 (sai khác logit lớn nhất 0,032, không đổi nhãn nào ảnh hưởng chỉ số) và nhanh nhất ở batch 1
  (10,3 ms) cũng như batch 32 (386 ảnh/s so với 113). **AMP ở batch 1 lại chậm hơn FP32** (14,2–20 ms so với 11,7 ms)
  do chi phí ép kiểu từng lớp — đúng cảnh báo slide trang 73; ở batch 32 AMP nhanh gấp 2,7 lần. Gộp BN chính xác
  (sai số logit lớn nhất 7·10⁻⁵ với ResNet-50, 4·10⁻⁵ với EfficientNet-B0; F1 không đổi: 0,7838 trước và sau); ở batch 32 nhanh hơn ~2% (248 so với 253 ms), ở batch 1 khác biệt nằm trong
  dao động đo. ConvNeXt-T dùng LayerNorm nên không có BN để gộp.
- **Ngoại tuyến vs thời gian thực:** mọi phương pháp 1 model đều nằm xa dưới ngân sách 30–100 ms/khung trên T4. Phương
  pháp chọn cho chung kết (theo luật định trước: macro-F1 val cao nhất trong I00/I01/I02/I04, hoà thì ít forward hơn):
  **I04 ảnh đủ 288**, p95 = 17,7 ms — dùng được cả cho robot. TTA 10 crop hay ensemble chỉ hợp ngoại tuyến và ở đây
  thậm chí không tốt hơn.

## 6. Cấu hình tốt nhất và kết quả test (Bước 4)

**Cấu hình chung kết F01** (chốt hoàn toàn trên val):

- Backbone ConvNeXt-T (`convnext_tiny.in12k_ft_in1k`), tinh chỉnh toàn bộ, công thức nền T00 **+ CutMix (α = 1)**
  (T04 — công thức tốt nhất trên val; T10 kết hợp kém hơn, xem mục 4), 10 epoch, checkpoint theo macro-F1 val.
- Suy luận **I04: ảnh đầy đủ 256 → resize 288**, 1 lượt forward (tốt nhất trên val trong I00/I01/I02/I04, mục 5).
- **Temperature scaling** khớp trên val của từng seed: T = 0,701 / 0,683 / 0,705 (seed 0/1/2). T < 1: mô hình
  CutMix **thiếu tự tin** (nhãn mềm khi train) nên T làm sắc xác suất; accuracy không đổi.
- 3 seed (0, 1, 2). F01 seed 0 trùng cấu hình T04 seed 0 nên dùng lại checkpoint đó (`alias_of`).

**Mốc so sánh**: T00 (công thức nền) + I00 (1-view center crop 224), không temperature scaling, cùng 3 seed.

Test chạy **đúng một lần cho mỗi seed** của cấu hình chung kết và của mốc, trên toàn bộ 3.507 ảnh test (một lần
chạy test của cấu hình kết hợp đã loại được nêu rõ ở mục 8). Mọi số dưới đây là kết quả của
`python eval.py score` / `grade` (thư mục `eval_out/`), tính lại từ `predictions/*.csv`:

| | F01 (chung kết) | T00 + I00 (mốc) | Δ |
|---|---|---|---|
| macro-F1 val | 0,9719 ± 0,0004 | 0,9684 ± 0,0011 | +0,0035 |
| **macro-F1 test** | **0,9750 ± 0,0026** | 0,9706 ± 0,0020 | **+0,0044** |
| top-1 test | **97,92% ± 0,20** | 97,65% ± 0,23 | +0,27 điểm |
| balanced accuracy test | 0,9660 ± 0,0034 | 0,9710 ± 0,0032 | −0,0050 |
| ECE test (15 bin) | 0,0072 ± 0,0015 (chưa TS: 0,0344 ± 0,0041) | 0,0097 ± 0,0005 | |
| recall Chinee apple | 95,1% ± 1,5 | 92,9% ± 2,0 | +2,2 điểm |
| recall Snake weed | 94,6% ± 0,5 | 95,1% ± 1,0 | −0,5 điểm |
| độ trễ batch 1 FP32, T4 (p50 / p95) | 16,8 / 17,7 ms | 11,8 / 12,5 ms | 1,4× |

(mean ± std mẫu ddof = 1 qua 3 seed; F01 từng seed: macro-F1 test 0,9764 / 0,9720 / 0,9765.)

**Δ macro-F1 test = +0,0044, lớn hơn std lớn hơn trong hai nhóm (0,0026) 1,7 lần** nhưng chưa đến 2 lần và dưới 0,01:
cải thiện là thật nhưng **nhỏ**. `eval.py grade` (đề xuất): I1 = 7/7, I2 = 4/5, I3 = 4/4, I4 = 2/2, I5 = 2/2 →
**19/20**. Chênh val/test của F01 là 0,0031 (< 0,02), không có dấu hiệu chọn quá khớp val.

**So với bài báo** (trích dẫn, README 2.3): ResNet-50 95,7%, Inception-v3 95,1% (weighted average accuracy, 5 fold,
~100 epoch); recall Chinee apple 88,5%, Snake weed 88,8%. Cấu hình của chúng tôi đạt top-1 97,9% và recall hai lớp khó
~95% chỉ với 10 epoch, chủ yếu nhờ trọng số tiền huấn luyện mạnh hơn (ImageNet-12k → 1k) và kiến trúc hiện đại; định
nghĩa accuracy và số fold khác nhau nên so sánh chỉ mang tính tham khảo.

**Theo lớp** (sheet `PerClass`, F01 mean qua 3 seed): F1 thấp nhất là Snake weed 0,960, Prickly acacia 0,961,
Chinee apple 0,966; cao nhất Parkinsonia 0,986 và Negatives 0,984.

**Ma trận nhầm lẫn và phân tích lỗi** (`figures/confusion_F01.png`, `figures/confusion_T00.png`, cộng 3 seed;
`figures/errors_chinee_snake.png`):

| Nhầm lẫn (tổng 3 seed) | F01 | T00 |
|---|---|---|
| loài cỏ → Negatives (bỏ sót cỏ) | 148 | 87 |
| Negatives → một loài cỏ (báo nhầm) | 28 | 91 |
| Chinee apple → Snake weed | 9 | 20 |
| Snake weed → Chinee apple | 5 | 3 |
| Prickly acacia → Parkinsonia | 6 | 8 |

- **Lỗi còn lại chủ yếu là bỏ sót cỏ** (đoán `Negatives`): 23–27 ảnh (cộng 3 seed) ở mỗi lớp Chinee apple, Prickly
  acacia, Rubber vine, Siam weed, Snake weed; ít hơn ở Lantana (17), Parthenium (10), Parkinsonia (1). So với mốc, F01 **dịch
  ngưỡng về phía `Negatives`**: báo nhầm giảm 3 lần (91 → 28) nhưng bỏ sót tăng 1,7 lần (87 → 148). Vì vậy
  macro-F1 và top-1 tăng (precision các loài tăng mạnh: ví dụ Lantana 0,959 → 0,997) trong khi balanced accuracy (trung
  bình recall) giảm 0,005. Giả thuyết: CutMix dán vùng ảnh nền vào ảnh cỏ và ngược lại; khi mảnh cỏ nhỏ, nhãn trộn vẫn
  ghi trọng số lớn cho loài cỏ, nhưng ảnh 288 toàn khung chứa nhiều nền hơn crop 224, làm mô hình thận trọng hơn với ảnh
  mà cỏ chỉ chiếm phần nhỏ.
- **Chinee apple ↔ Snake weed** (cặp khó trong bài báo): giảm từ 23 xuống 14 ảnh (3 seed). Các ảnh còn nhầm của seed 0
  (`errors_chinee_snake.png`) đều là ảnh tối/bóng râm, lá nhỏ chỉ lộ một phần giữa thân cành khô hoặc cỏ khác; độ tin cậy
  thấp (p ≈ 0,55) trừ một ảnh Snake weed bị đoán Chinee với p = 0,96. Hai loài đều có lá xanh hình bầu dục cỡ nhỏ; ở
  256×256 khi lá chỉ chiếm vài chục pixel, thông tin hình dạng mép lá/gân lá — đặc trưng phân biệt chính — bị mất.
- Parkinsonia ↔ Prickly acacia (bài báo nêu 1,3%): còn 6 + 0 ảnh, đều là cây họ đậu lá kép nhỏ.

## 7. Kết luận và khuyến nghị

**Cấu hình nào tốt nhất?** ConvNeXt-T tinh chỉnh toàn bộ + CutMix, suy luận 1 lượt ở độ phân giải 288 trên ảnh đầy đủ,
temperature scaling: macro-F1 test **0,9750 ± 0,0026**, top-1 **97,92% ± 0,20%**, ECE 0,007, p95 17,7 ms/ảnh trên T4.
Hơn mốc T00 + I00 **+0,0044 macro-F1** (1,7 lần std): vượt nhiễu nhưng không nhiều, và đổi lại bỏ sót cỏ nhiều hơn.

**Yếu tố nào đóng góp nhiều nhất?** Theo thứ tự (đều đo trên val, cùng công thức trừ yếu tố đang xét):

1. **Khởi tạo tiền huấn luyện**: từ đầu 0,30 → tinh chỉnh 0,97 (+0,67). Đóng băng chỉ đạt 0,84.
2. **Backbone (gắn với tag trọng số)**: ConvNeXt-T 0,968 so với ResNet-50 0,785, EfficientNet-B0 0,724 (+0,18 đến
   +0,24); phần lớn khoảng cách của mạng BN là do lệch thống kê BN với công thức nền này (mục 3), không thuần do kiến trúc.
3. **Suy luận — độ phân giải kiểm tra**: +0,008 (I04, 288) với chi phí 1,4×; TTA/ensemble không giúp.
4. **Công thức huấn luyện (augmentation, loss, EMA, sampler)**: mọi thay đổi trong ±0,004, cỡ nhiễu seed; kết hợp các
   yếu tố "tốt" còn làm giảm. Với backbone tiền huấn luyện mạnh và 10 epoch, công thức nền đã gần bão hoà.

Như vậy ở bài toán này **backbone + khởi tạo** quyết định gần như toàn bộ chất lượng; suy luận ở đúng độ phân giải là
cải thiện rẻ nhất tiếp theo; tinh chỉnh công thức chỉ cho lợi ích biên.

**Triển khai trên robot (ngân sách 30–100 ms/khung):** chọn chính F01 — một lượt forward ConvNeXt-T 288, p95 17,7 ms
trên T4 ở FP32 (FP16 còn nhanh hơn, cùng độ chính xác), không cần TTA/ensemble. Nếu phần cứng yếu hơn T4 (Jetson), dùng
FP16 + TensorRT và đo lại; nếu muốn giảm bỏ sót cỏ, có thể dùng mốc T00 (balanced accuracy cao hơn) hoặc hạ ngưỡng xác
suất `Negatives` — nhưng ngưỡng đó phải được chọn trên val. Không dùng ResNet-50/EfficientNet/MobileNetV3 với công thức
này trừ khi ước lượng lại thống kê BN (mục 3) hoặc đổi augmentation.

## 8. Hạn chế và việc tiếp theo

- **Số seed:** sàng backbone và ablation chỉ 1 seed (nhiễu ước lượng từ 3 seed của T00: std 0,0011 trên val); chỉ chung
  kết và mốc có 3 seed. Các kết luận về CutMix/EMA/focal ở mục 4 vì vậy chỉ là xu hướng.
- **Một fold**, chia ngẫu nhiên theo ảnh, **không theo địa điểm**: ảnh cùng địa điểm/cùng ngày có thể nằm ở cả train và
  test, nên điểm test có thể **lạc quan** so với khi robot gặp cánh đồng, mùa, ánh sáng mới. Temperature T khớp trên
  val cùng phân phối cũng có thể không còn đúng khi lệch miền.
- **Giảm bớt do ngân sách GPU (Colab miễn phí, T4):** 10 epoch thay vì 12–15; ablation chỉ trên ConvNeXt-T; không thử trục
  E (LR/optimizer) và G (độ phân giải train, số epoch); không làm lại ablation cho mạng BN.
- **Công thức nền không công bằng hoàn toàn với mạng BN**: RandomResizedCrop scale 0,08 khiến thống kê BN lệch (mục 3);
  một công thức khác (scale ≥ 0,3, hoặc ước lượng lại BN) có thể đổi thứ hạng backbone. ResNet-50 `a1_in1k` có lẽ cần
  LR/epoch lớn hơn.
- **Thứ tự tham lam và một lần chạy test bị loại (đã xử lý).** Cấu hình kết hợp T10 được định nghĩa trước khi biết kết
  quả của nó, và stage chung kết đã được xếp hàng tự động ngay sau T10 với giả định dùng T10. Dòng thời gian (giờ file
  trên Google Drive và giờ commit git, UTC+7, 05/10/2026):

  | Thời điểm | Sự kiện |
  |---|---|
  | 00:45 | T10 xong, `runs/T10/seed0/summary.json`: macro-F1 **val** 0,9651 (< T00 0,9676 < T04 0,9703) |
  | 00:46 | script xếp hàng sẵn tự chạy test cho F01-kết hợp seed 0 (`F01_seed0_test.csv`, ...) |
  | 00:54 | dừng stage chung kết; chuyển nguyên các file đó vào `discarded_F01_combo/` |
  | 00:55 | commit `7c1f9ca` ghi lý do đổi: T10 kém hơn T00 **trên val** → chung kết dùng T04 |

  Quyết định chỉ dựa trên macro-F1 val; các file test bị loại được giữ nguyên, **không tính chỉ số, không mở**, để
  giảng viên kiểm tra nếu cần (sha256 `F01_seed0_test.csv` = `43897454…6264af41`, `F01uncal_seed0_test.csv` = `3808b1a6…7d02d954e`; giá trị đầy đủ trong `tables/discarded_sha256.txt`). Để lỗi này không thể lặp lại, `run_experiments.py --stage final --combo auto` giờ tự
  chọn công thức có macro-F1 val cao nhất trong T00…T10 và **từ chối chạy (kể cả test) nếu còn thiếu bất kỳ kết quả val
  nào** (`select_final_recipe`, ghi `tables/final_recipe_choice.json`); với dữ liệu hiện có nó chọn đúng T04 (CutMix).
- Thí nghiệm thất bại/không giúp: huấn luyện từ đầu, ensemble, TTA nhiều crop, kết hợp T10, CE có trọng số.
- **Việc tiếp theo** nếu có thêm một ngày: chạy 3–5 fold (theo địa điểm nếu có metadata); 3 seed cho các ablation hứa
  hẹn (CutMix, EMA); công thức riêng cho mạng BN; chưng cất ConvNeXt-T sang MobileNetV3 cho thiết bị biên; đánh giá trên
  ảnh làm tối/mờ để đo độ bền và hiệu chuẩn khi lệch miền.

## 9. Phụ lục

- Danh sách `exp_id` và cấu hình đầy đủ: sheet `Backbones`, `Training`, `Inference`, `Final` của `results.xlsx`;
  cấu hình từng lần chạy (gồm tag trọng số, phiên bản thư viện): `runs/<exp_id>/seed<k>/config.json` và `summary.json`
  (trên Google Drive, không commit vì có checkpoint).
- Ảnh đường cong: `curves/<exp_id>_<mô tả>.png` (T00 và F01 có một ảnh mỗi seed: `_seed0/1/2`).
- Notebook: `code/lab_day2.ipynb` (sạch, chạy lại được) và `code/lab_day2_run.ipynb` (bản đã chạy trên Colab).
- Kết quả `eval.py`: `eval_out/score_F01.txt`, `score_T00.txt`, `score_F01uncal.txt`, `grade.txt`.
