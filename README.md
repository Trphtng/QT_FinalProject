# Quản lý Danh mục Đầu tư bằng Deep Reinforcement Learning (PPO)

Hệ thống đầy đủ cho bài toán phân bổ danh mục đa tài sản, sử dụng kiến trúc **PPO Actor-Critic** với PyTorch. Tải dữ liệu thị trường thực tế từ Yahoo Finance hoặc thị trường Việt Nam (Vnstock), tự động tính toán các chỉ số kỹ thuật, huấn luyện chính sách danh mục, và so sánh với các chiến lược cơ bản (baseline).

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc mô hình](#2-kiến-trúc-mô-hình)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Cấu hình hệ thống](#5-cấu-hình-hệ-thống)
6. [Hướng dẫn Huấn luyện](#6-hướng-dẫn-huấn-luyện)
7. [Hướng dẫn Đánh giá](#7-hướng-dẫn-đánh-giá)
8. [Đánh giá toàn bộ Fold](#8-đánh-giá-toàn-bộ-fold)
9. [Tạo báo cáo HTML](#9-tạo-báo-cáo-html)
10. [Danh sách tài sản mặc định](#10-danh-sách-tài-sản-mặc-định)
11. [Hỗ trợ thị trường Việt Nam](#11-hỗ-trợ-thị-trường-việt-nam)
12. [Các chỉ số đánh giá](#12-các-chỉ-số-đánh-giá)
13. [Biểu đồ xuất ra](#13-biểu-đồ-xuất-ra)
14. [Xử lý lỗi thường gặp](#14-xử-lý-lỗi-thường-gặp)
15. [Quy trình làm việc đầy đủ](#15-quy-trình-làm-việc-đầy-đủ)
16. [Tóm tắt lệnh](#16-tóm-tắt-lệnh)

---

## 1. Tổng quan dự án

Dự án xây dựng quy trình DRL hoàn chỉnh cho bài toán phân bổ danh mục:

- **Dữ liệu thị trường**: mô phỏng từ dữ liệu OHLCV lịch sử thực tế
- **Feature Engineering**: log return, RSI, MACD, Bollinger Bands, SMA/EMA, volatility, momentum, EMA Sharpe, drawdown, trend signal
- **Backbone encoder**: LSTM / CNN1D / Transformer (chọn qua config)
- **Thuật toán**: PPO clipped Actor-Critic — ổn định và phù hợp với dữ liệu tài chính
- **Reward function (V3)**: tích hợp log-return, EMA Sharpe bonus, momentum bonus, penalty turnover và drawdown
- **Walk-Forward Cross-Validation**: 18 folds, mỗi fold 126 ngày test
- **Baselines**: Buy & Hold Equal Weight, Markowitz (MVO), Random Allocation

---

## 2. Kiến trúc mô hình

### State (Đầu vào)

Tại mỗi bước thời gian, state gồm:

| Thành phần | Mô tả |
|---|---|
| Market features | Cửa sổ lookback × số tài sản × số feature |
| Portfolio weights | Tỷ trọng hiện tại của từng tài sản |
| Cash ratio | Tỷ lệ tiền mặt |
| Rolling volatility | Biến động ngắn hạn |
| EMA Sharpe | Sharpe ratio theo EMA (anti-noise) |
| Current drawdown | Mức độ suy giảm từ đỉnh |
| Trend signal | Tín hiệu xu hướng thị trường |

> ⚠️ **Quan trọng**: `portfolio_state_dim = 26` khi `include_prev_weights: true`. Các checkpoint cũ (dim=22) sẽ không tương thích với code hiện tại.

### Encoder

Chọn loại encoder trong config:

```yaml
model:
  encoder_type: "transformer"   # Khuyến nghị: tốt nhất trên time-series tài chính
  # encoder_type: "lstm"
  # encoder_type: "cnn"
```

### Actor

- Shared latent representation từ encoder
- **Phân phối Dirichlet** cho portfolio weights
- Tổng weights luôn = 1 (bao gồm cash)
- Inference deterministic: dùng mean của Dirichlet

### Critic

- Scalar state-value head (ước tính V(s))

### Reward Function (Version 3 — hiện tại)

```
R_t = λ_return × log(1 + r_t)
    + λ_sharpe  × EMA_Sharpe_bonus
    + λ_momo    × Momentum_bonus
    - λ_turn    × Turnover_penalty
    - λ_dd      × Drawdown_penalty
    - λ_var     × Downside_variance_penalty
```

> ✅ **Điểm mạnh**: EMA Sharpe loại bỏ nhiễu của rolling window. Log-return phản ánh đúng lãi kép.
> ⚠️ **Reward được clip về [-0.1, 0.1]** và normalize bằng RunningMeanStd để ổn định gradient.

---

## 3. Cấu trúc thư mục

```text
QT_FinalProject/
│── configs/
│   ├── config.yaml              ← Config chính (US tickers, Yahoo Finance)
│   └── config_vn30.yaml         ← Config thị trường Việt Nam (Vnstock)
│
│── data/
│   ├── raw/                     ← Cache CSV thô (tự động tạo)
│   └── processed/               ← Cache tensor đã xử lý (tự động tạo)
│
│── src/
│   ├── data_loader.py           ← Tải dữ liệu Yahoo / Vnstock
│   ├── feature_engineering.py  ← Tính RSI, MACD, Bollinger, volatility...
│   ├── env/
│   │   └── portfolio_env.py    ← ⭐ Môi trường RL (reward V3, state augmented)
│   ├── models/
│   │   ├── encoders.py          ← LSTM / CNN / Transformer
│   │   ├── actor.py             ← Dirichlet policy head
│   │   ├── critic.py            ← Value head
│   │   └── actor_critic.py      ← Model tổng hợp
│   ├── agents/
│   │   └── trainer.py           ← PPO training loop, GAE, rollout
│   ├── baselines/
│   │   ├── buy_hold.py          ← Buy & Hold Equal Weight
│   │   ├── markowitz.py         ← Mean-Variance Optimization
│   │   └── random_strategy.py   ← Random Allocation
│   └── utils/
│       ├── metrics.py           ← Sharpe, Sortino, MDD, Calmar...
│       ├── plotting.py          ← Equity curve, heatmap, drawdown chart
│       ├── logger.py            ← Logger chuẩn
│       └── seed.py              ← Reproducibility
│
│── outputs/
│   ├── models/                  ← Checkpoint (.pt) của từng fold
│   ├── figures/                 ← Biểu đồ PNG
│   ├── reports/                 ← JSON metrics, walk-forward summary
│   └── tensorboard/             ← TensorBoard logs
│
│── train.py                     ← ⭐ Script huấn luyện chính
│── evaluate.py                  ← ⭐ Script đánh giá + so sánh baseline
│── evaluate_all_folds.py        ← Đánh giá toàn bộ 18 fold, chọn best
│── generate_report.py           ← Tạo báo cáo HTML đẹp
│── requirements.txt
└── README.md
```

---

## 4. Cài đặt môi trường

### Yêu cầu

- Python 3.10+
- CUDA (khuyến nghị) — model tự detect GPU nếu có

### Cài đặt

```bash
pip install -r requirements.txt
```

### Kiểm tra GPU

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

> 💡 Nếu không có GPU, model vẫn chạy được trên CPU nhưng sẽ chậm hơn ~5–10x.

---

## 5. Cấu hình hệ thống

File config chính: `configs/config.yaml`

Các section quan trọng:

### `data` — Dữ liệu

```yaml
data:
  provider: "yahoo"         # "yahoo" hoặc "vnstock"
  tickers: [AAPL, MSFT, ...]
  start_date: "2018-01-01"
  lookback_window: 30       # ⭐ Số ngày lịch sử đầu vào
  walk_forward:
    enabled: true
    train_window: 756       # ~3 năm training
    val_window: 126         # ~6 tháng validation
    test_window: 126        # ~6 tháng test
    step_size: 63           # Bước trượt ~3 tháng
```

### `environment` — Môi trường RL

```yaml
environment:
  initial_cash: 1000000.0
  fee_rate: 0.001           # Phí giao dịch 0.1%
  slippage_rate: 0.0005     # Slippage 0.05%
  lambda_var: 0.5           # Penalty biến động
  lambda_turnover: 0.01     # Penalty giao dịch nhiều
  lambda_drawdown: 0.05     # Penalty drawdown
  lambda_return_bonus: 3.0  # Thưởng lợi nhuận
  lambda_sharpe_bonus: 0.3  # Thưởng EMA Sharpe
  lambda_momentum_bonus: 0.1 # Thưởng xu hướng
  rebalance_frequency: 5    # Rebalance mỗi 5 ngày
  rebalance_alpha: 0.5      # Tốc độ chuyển vị trí
```

> ⚠️ **Quan trọng**: Các hệ số lambda được calibrate theo scale log-return (~1e-3). Không nên thay đổi quá nhiều cùng lúc.

### `training` — Huấn luyện PPO

```yaml
training:
  total_epochs: 50
  learning_rate: 0.00005    # LR nhỏ = ổn định hơn
  gamma: 0.995              # Discount factor cao cho long-horizon
  clip_epsilon: 0.15        # PPO clip range
  entropy_coef_start: 0.01  # Exploration ban đầu
  weight_decay: 0.0001      # Regularization
  max_grad_norm: 0.5        # Gradient clipping
```

### Model Selection Score

```python
# Score dùng để chọn best fold trong walk-forward:
score = Sharpe + 2.0 × CAGR − 0.5 × |MDD|
```

---

## 6. Hướng dẫn Huấn luyện

### Huấn luyện cơ bản (US tickers)

```bash
python train.py
```

### Huấn luyện với dữ liệu Việt Nam

```bash
python train.py --config configs/config_vn30.yaml
```

### Quy trình huấn luyện tự động

1. Tải/load cache dữ liệu OHLCV từ Yahoo Finance hoặc Vnstock
2. Tính toán 27 features kỹ thuật và covariance matrices
3. Lưu cache vào `data/processed/` (tái sử dụng cho lần sau)
4. Chạy Walk-Forward: 18 folds × 126 ngày test
5. Mỗi fold: train PPO → validate → lưu checkpoint tốt nhất
6. Lưu `outputs/reports/walk_forward_summary.json`

### Theo dõi bằng TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

Các metric được log:
- `reward/raw_reward` — Raw reward mỗi step
- `reward/sharpe_bonus` — Thành phần EMA Sharpe bonus
- `reward/momentum_bonus` — Thành phần momentum
- `loss/actor_loss`, `loss/critic_loss`
- `metrics/sharpe_val` — Sharpe trên validation

> 💡 Nếu `reward_sharpe_bonus` chiếm ưu thế quá mức, giảm `lambda_sharpe_bonus` xuống 0.2.

---

## 7. Hướng dẫn Đánh giá

### Đánh giá tự động (chọn best fold)

```bash
python evaluate.py
```

### Đánh giá một fold cụ thể

```bash
# Xem danh sách tất cả fold có sẵn
python evaluate.py --list-folds

# Đánh giá fold 7 (best test return)
python evaluate.py --fold 7

# Đánh giá fold 16 (beat cả Markowitz + BuyHold)
python evaluate.py --fold 16

# Đánh giá với config VN30
python evaluate.py --config configs/config_vn30.yaml --fold 5
```

### Đánh giá chỉ PPO (không so baseline)

```bash
python evaluate.py --rl-only
python evaluate.py --config configs/config_vn30.yaml --rl-only
```

### Output của evaluate.py

- Bảng so sánh: PPO vs BuyHold vs Markowitz vs Random
- File: `outputs/reports/metrics_summary.json`
- Biểu đồ: `outputs/figures/us/`

---

## 8. Đánh giá toàn bộ Fold

Script `evaluate_all_folds.py` đánh giá **tất cả 18 fold** trên tập test riêng của từng fold (không phải validation), xếp hạng và chọn best model thực sự.

```bash
# Đánh giá tất cả fold và chọn best
python evaluate_all_folds.py --set-best

# Chỉ hiển thị top 5
python evaluate_all_folds.py --top 5

# Dùng config khác
python evaluate_all_folds.py --config configs/config_vn30.yaml
```

> ⭐ **Khuyến nghị**: Luôn chạy `evaluate_all_folds.py --set-best` sau khi train xong để chọn model tốt nhất theo test performance thực sự (không phải validation score).

Kết quả lưu vào: `outputs/reports/fold_evaluation.json`

**Composite score** dùng để xếp hạng:
```
Test Score = Sharpe + 2 × CAGR − 0.5 × |Max Drawdown|
```

---

## 9. Tạo báo cáo HTML

```bash
python generate_report.py
```

Báo cáo được lưu tại: `outputs/reports/evaluation_report.html`

Mở trực tiếp trong trình duyệt để xem bảng xếp hạng, KPI cards, phân tích best model và so sánh baseline.

---

## 10. Danh sách tài sản mặc định

### US Tickers (`configs/config.yaml`)

```yaml
data:
  provider: "yahoo"
  tickers:
    - AAPL    # Apple
    - MSFT    # Microsoft
    - GOOGL   # Alphabet
    - AMZN    # Amazon
    - TSLA    # Tesla
    - META    # Meta Platforms
    - NVDA    # NVIDIA
    - JPM     # JPMorgan Chase
    - XOM     # ExxonMobil
    - NFLX    # Netflix
```

Có thể thay/thêm bất kỳ ticker nào hợp lệ trên Yahoo Finance.

---

## 11. Hỗ trợ thị trường Việt Nam

Config: `configs/config_vn30.yaml`

```yaml
data:
  provider: "vnstock"
  vn_source: "VCI"          # Nguồn dữ liệu (VCI hoặc KBS)
  tickers:
    - FPT
    - VCB
    - HPG
    - MBB
    - ACB
    - SSI
    - MWG
    - VNM
    - TCB
    - VPB
```

**Lưu ý quan trọng:**
- `vn_source: "VCI"` là nguồn mặc định, giàu dữ liệu nhất
- Nếu VCI bị block, thử đổi sang `vn_source: "KBS"`
- Có thể thêm ETF như `E1VFVN30` hoặc index `VNINDEX`, `VN30`
- Cache riêng biệt: `data/raw_vn30/` và `data/processed_vn30/`

Cài đặt thư viện vnstock:

```bash
pip install vnstock
```

---

## 12. Các chỉ số đánh giá

| Chỉ số | Ý nghĩa | Mục tiêu |
|---|---|---|
| **Total Return %** | Tổng lợi nhuận | Càng cao càng tốt |
| **Sharpe Ratio** | Lợi nhuận / rủi ro | > 1.0 |
| **Max Drawdown** | Sụt giảm tối đa từ đỉnh | < -15% |
| **Volatility** | Biến động hàng năm | Càng thấp càng tốt |
| **Calmar Ratio** | CAGR / |MDD| | > 0.5 |
| **Turnover** | Tần suất giao dịch | < 2% |


---

## 13. Biểu đồ xuất ra

Lưu tại `outputs/figures/us/` (US) hoặc `outputs/figures/vn30/` (VN):

- `equity_curve_test.png` — Đường tăng trưởng danh mục
- `drawdown_chart.png` — Biểu đồ drawdown theo thời gian
- `weight_heatmap.png` — Heatmap phân bổ tài sản
- `baseline_comparison.png` — So sánh với các chiến lược cơ bản
- `rolling_sharpe.png` — Sharpe ratio trượt 63 ngày
- `training_reward_curve.png` — Đường reward trong quá trình train
- `loss_curve.png` — Actor/Critic loss

---

## 14. Xử lý lỗi thường gặp

### Lỗi tải dữ liệu Yahoo Finance

```bash
# Xóa cache và tải lại từ đầu
Remove-Item -Recurse data\raw\*
python train.py
```

Kiểm tra:
- Kết nối internet
- Ticker hợp lệ trên Yahoo Finance (vd: `AAPL`, không phải `aapl`)
- Giảm `data.max_workers` nếu bị rate-limit

### Lỗi tải dữ liệu Vnstock

```yaml
# Thử đổi nguồn trong config_vn30.yaml
data:
  vn_source: "KBS"    # thay vì "VCI"
```

Xóa cache VN và tải lại:
```bash
Remove-Item -Recurse data\raw_vn30\*
python train.py --config configs/config_vn30.yaml
```

### Lỗi mismatch checkpoint dimension

```
RuntimeError: size mismatch for portfolio_proj.0.weight
```

> Nguyên nhân: checkpoint cũ có `portfolio_state_dim=22`, code mới dùng `dim=26`.
> Giải pháp: Chạy lại `python train.py` để tạo checkpoint mới.

### Lỗi CUDA out of memory

```yaml
# Giảm batch size trong config.yaml
training:
  minibatch_size: 128    # giảm từ 256 xuống 128
```

---

## 15. Quy trình làm việc đầy đủ

### Quy trình chuẩn (US tickers)

```bash
# Bước 1: Cài đặt
pip install -r requirements.txt

# Bước 2: Huấn luyện (tất cả 18 fold)
python train.py

# Bước 3: Đánh giá toàn bộ fold, chọn best
python evaluate_all_folds.py --set-best

# Bước 4: Đánh giá chi tiết best fold
python evaluate.py

# Bước 5: Xem đánh giá fold cụ thể
python evaluate.py --list-folds
python evaluate.py --fold 7

# Bước 6 (tuỳ chọn): Tạo báo cáo HTML
python generate_report.py

# Bước 7 (tuỳ chọn): Theo dõi training
tensorboard --logdir outputs/tensorboard
```

### Quy trình thị trường Việt Nam

```bash
pip install -r requirements.txt
pip install vnstock

python train.py --config configs/config_vn30.yaml
python evaluate_all_folds.py --config configs/config_vn30.yaml --set-best
python evaluate.py --config configs/config_vn30.yaml
```

### Quy trình đánh giá hai dataset

```bash
# Dataset 1: US
python train.py --config configs/config.yaml
python evaluate_all_folds.py --set-best
python evaluate.py --fold 7

# Dataset 2: VN30
python train.py --config configs/config_vn30.yaml
python evaluate_all_folds.py --config configs/config_vn30.yaml --set-best
python evaluate.py --config configs/config_vn30.yaml

# Tạo báo cáo tổng hợp
python generate_report.py
```

---

## 16. Tóm tắt lệnh

```bash
# ── CÀI ĐẶT ──────────────────────────────────────────────────────────────
pip install -r requirements.txt
pip install vnstock                          # Chỉ cần nếu dùng thị trường VN

# ── HUẤN LUYỆN ───────────────────────────────────────────────────────────
python train.py                              # US tickers (mặc định)
python train.py --config configs/config_vn30.yaml   # Thị trường VN

# ── ĐÁNH GIÁ ─────────────────────────────────────────────────────────────
python evaluate.py                           # Đánh giá best fold tự động
python evaluate.py --list-folds              # Xem danh sách tất cả fold
python evaluate.py --fold 7                  # Đánh giá fold 7
python evaluate.py --fold 16                 # Đánh giá fold 16 (khuyến nghị)
python evaluate.py --rl-only                 # Chỉ PPO, không so baseline
python evaluate.py --config configs/config_vn30.yaml --fold 5

# ── ĐÁNH GIÁ TOÀN BỘ FOLD ────────────────────────────────────────────────
python evaluate_all_folds.py                 # Đánh giá tất cả 18 fold
python evaluate_all_folds.py --set-best      # Tự động chọn và set best fold
python evaluate_all_folds.py --top 5         # Chỉ hiển thị top 5

# ── BÁO CÁO ──────────────────────────────────────────────────────────────
python generate_report.py                    # Tạo báo cáo HTML tại outputs/reports/

# ── TENSORBOARD ──────────────────────────────────────────────────────────
tensorboard --logdir outputs/tensorboard     # Theo dõi training real-time

# ── LÀM SẠCH CACHE ───────────────────────────────────────────────────────
Remove-Item -Recurse data\raw\*              # Xóa cache raw US
Remove-Item -Recurse data\processed\*        # Xóa cache processed US
Remove-Item -Recurse data\raw_vn30\*         # Xóa cache raw VN
Remove-Item -Recurse data\processed_vn30\*   # Xóa cache processed VN
```

---

## Ghi chú kỹ thuật quan trọng

> ⭐ **Walk-Forward Selection**: Sau khi train, chạy `evaluate_all_folds.py --set-best` để chọn best model dựa trên *test performance* thực sự (không phải validation score). Điều này tránh selection bias.

> ⚠️ **Không dùng `use_cache: true` khi thay đổi `features.include_columns`**. Phải xóa `data/processed/` để feature engineering chạy lại.

> 💡 **Về hiệu suất mô hình**: PPO thua Random Allocation trong giai đoạn bull market cực mạnh là bình thường. Điểm mạnh của PPO nằm ở **risk-adjusted return** (Sharpe, Calmar), không phải absolute return.

> ⚙️ **Thay đổi số lượng tài sản**: Khi thêm/bớt ticker, phải xóa toàn bộ checkpoint cũ và train lại từ đầu vì `n_assets` ảnh hưởng đến kích thước model.
