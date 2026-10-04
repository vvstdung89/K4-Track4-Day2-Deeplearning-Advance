# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận

Bài nộp của **MSSV_ho_ten** (đổi tên thư mục thành `<mssv>_<ho_ten_khong_dau>` trước khi nộp).

| Sản phẩm | Vị trí |
|---|---|
| Bảng so sánh mọi thí nghiệm | [`results.xlsx`](results.xlsx) (Summary, Backbones, Training, Inference, Final, PerClass, Latency) |
| Báo cáo | [`report.md`](report.md) |
| Đường cong training (mỗi `exp_id` một ảnh) | [`curves/`](curves) |
| Dự đoán test/val (định dạng `eval.py`) | [`predictions/`](predictions) |
| Hình cho báo cáo (EDA, kiểm tra pipeline, đánh đổi, ma trận nhầm lẫn, ảnh lỗi) | [`figures/`](figures) |
| Bảng trung gian (kiểm tra chia, suy luận, độ trễ, hiệu chuẩn) | [`tables/`](tables) |
| Kết quả `eval.py score/grade` | [`eval_out/`](eval_out) |
| Code | [`code/`](code) |

## Chạy lại

**Notebook:** [`code/lab_day2.ipynb`](code/lab_day2.ipynb), mở bằng Google Colab (Runtime → GPU T4).
Notebook giả định repo nằm ở `MyDrive/VinAI/K4-Track4-Day2-Deeplearning-Advance` trên Google Drive
(sửa biến `REPO` ở ô đầu nếu khác). Bản notebook đã chạy (có output) nằm cạnh nó: `code/lab_day2_run.ipynb`.

Thứ tự chạy (mỗi lệnh là một stage của `code/run_experiments.py`; các lần chạy đã có `summary.json` được bỏ qua,
nên có thể chạy tiếp sau khi Colab bị ngắt):

```bash
export LAB_SUB=<thư mục bài nộp> LAB_DATA=/content/data LAB_EPOCHS=10
cd code
python -m unittest test_code                                   # 21 kiểm tra tự viết (CPU được)
python run_experiments.py --stage eda                          # kiểm tra chia fold 0 + EDA + cache ảnh
python run_experiments.py --stage sanity --backbone resnet50   # loss ban đầu, overfit 1 batch, ảnh augmentation
python run_experiments.py --stage backbones                    # B01..B05
python run_experiments.py --stage bnrecal                      # chẩn đoán BN (mục 3 báo cáo)
python run_experiments.py --stage training --backbone convnext_tiny            # T00..T09
python run_experiments.py --stage combo    --backbone convnext_tiny --combo "mix=cutmix loss=focal ema_decay=0.999"  # T10
python run_experiments.py --stage inference --backbone convnext_tiny           # I00..I08 + độ trễ
python run_experiments.py --stage final    --backbone convnext_tiny --combo "mix=cutmix" --seeds 0,1,2   # F01 = T04 + I04 288 + TS
python make_results.py --backbone convnext_tiny                # results.xlsx + hình
```

Chạy một cấu hình đơn lẻ: `python train.py --set exp_id=B01 backbone=resnet50 seed=0 epochs=10`.

## Môi trường đã dùng

Google Colab (bản miễn phí), GPU **Tesla T4 15 GB**, 2 vCPU, 12 GB RAM · Python 3.13.15 · torch 2.11.0+cu130 ·
torchvision 0.26.0+cu130 · timm 1.0.29 · numpy 2.1.3 · pandas 2.2.3 · matplotlib 3.10.0 · openpyxl 3.1.5.
Tag trọng số timm của từng backbone được ghi trong `results.xlsx` (sheet Backbones) và `runs/*/seed*/config.json`.

## Seed và tái lập

- Quét backbone và ablation: seed 0. Chung kết F01 và mốc T00: seed 0, 1, 2.
- `set_seed` cố định `random`, `numpy`, `torch` (CPU + CUDA), thứ tự batch (generator của DataLoader) và seed worker.
  `cudnn.benchmark=True` để nhanh, nên hai lần chạy cùng seed khớp xấp xỉ chứ không bit-by-bit.
- Chia dữ liệu: fold 0 nguyên bản của tác giả (`train/val/test_subset0.csv`), không sửa; seed không đổi cách chia.
- Checkpoint (`runs/`, ~100 MB mỗi lần chạy) không commit; nằm trên Google Drive của tác giả bài nộp.
