# Bài kiểm tra thực hành cuối kỳ — Làm sạch dữ liệu & Phân loại

Bộ dữ liệu: [`Maternal Health Risk Data Set - Modi.csv`](../data-cleaning-ml/test/Maternal%20Health%20Risk%20Data%20Set%20-%20Modi.csv)

## Cấu trúc thư mục

```
bai-kiem-tra-cuoi-ky/
├── cau1_lam_sach_du_lieu.py        # Câu 1: trực quan + làm sạch dữ liệu
├── cau2_phan_loai_4_mo_hinh.py      # Câu 2: phân loại với 4 mô hình
├── chay_toan_bo.py                 # Chạy tuần tự cả Câu 1 và Câu 2
└── output/                         # Toàn bộ kết quả sinh ra
    ├── cau1_histogram_6_cot_so.png
    ├── cau1_ma_tran_tuong_quan.png
    ├── cau1_truoc_sau_xu_ly_ngoai_le.png
    ├── cau1_bao_cao_lam_sach.txt
    ├── Maternal_Health_Risk_cleaned.csv
    ├── cau2_confusion_matrix_4_mo_hinh.png
    ├── cau2_duong_roc_4_mo_hinh.png
    ├── cau2_so_sanh_do_do_4_mo_hinh.png
    ├── cau2_bang_tong_hop_hieu_suat.csv
    └── cau2_ket_luan.txt
```

## Cách chạy

Dùng virtual environment sẵn có trong `data-cleaning-ml/.venv`:

```powershell
# Chạy cả hai câu
.\data-cleaning-ml\.venv\Scripts\python.exe bai-kiem-tra-cuoi-ky\chay_toan_bo.py

# Hoặc chạy từng câu
.\data-cleaning-ml\.venv\Scripts\python.exe bai-kiem-tra-cuoi-ky\cau1_lam_sach_du_lieu.py
.\data-cleaning-ml\.venv\Scripts\python.exe bai-kiem-tra-cuoi-ky\cau2_phan_loai_4_mo_hinh.py
```

> Cần cài thêm `matplotlib` và `seaborn` cho phần trực quan:
> `.\data-cleaning-ml\.venv\Scripts\python.exe -m pip install matplotlib seaborn`

---

## CÂU 1 — Trực quan và làm sạch dữ liệu

### 1. Trực quan histogram

Vẽ histogram cho **cả 6 cột số** (đề yêu cầu tối thiểu 2 cột): `Age`,
`SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`. Mỗi biểu đồ có
histogram + đường KDE.

**Kết quả quan sát được:**

| Cột | Hình dạng phân phối |
|-----|--------------------|
| `Age` | Lệch phải (right-skewed), đa số 10–40 tuổi |
| `SystolicBP` | Lệch trái, đỉnh nhọn tại 120 mmHg |
| `DiastolicBP` | Đa đỉnh, đỉnh rõ ở 60 và 80 mmHg |
| `BS` | Lệch phải mạnh, đa số 6–9 mmol/L, vân đuôi tới 19 |
| `BodyTemp` | Rất tập trung tại 98 °F (804/1014 mẫu) |
| `HeartRate` | Tập trung 70–80 bpm, có đuôi trái do giá trị 7 bpm |

### 2. Ma trận tương quan

Ma trận tương quan Pearson của **tất cả 6 cột numeric**.

**Kết quả quan sát được:**

- `SystolicBP` ↔ `DiastolicBP`: **0.787** — cặp tương quan mạnh nhất, hợp lý về
  mặt sinh lý (huyết áp tâm thu và tâm trương thường thay đổi cùng nhau).
- `Age` ↔ `BS` (0.473), `Age` ↔ `SystolicBP` (0.416), `BS` ↔ `SystolicBP` (0.425).
- `BodyTemp` có tương quan **âm** nhẹ với `Age` (−0.255) và với huyết áp (−0.287).
- `HeartRate` gần như **độc lập** với các biến còn lại (|r| ≤ 0.14) → là đặc
  trưng bổ sung giá trị cho mô hình cây.

Hai cột `SystolicBP`/`DiastolicBP` tương quan cao nên có thể xem xét đa cộng
tuyến khi dùng hồi quy tuyến tính; với 4 mô hình ở Câu 2 thì vấn đề này không
ảnh hưởng nghiêm trọng.

### 3. Làm sạch dữ liệu

#### (1) Thay thế giá trị trống/NaN bằng kỹ thuật Rolling Mean

Hàm `rolling_mean_impute()` triển khai đúng nguyên tắc **cửa sổ lùi**:

- Chỉ dùng tối đa **3 dòng đã quan sát** trước ô đang trống.
- **Không** dùng giá trị tương lai.
- **Không** dùng giá trị đã điền trước đó làm lịch sử (đọc lại từ series gốc).
- Nếu cửa sổ chưa có dữ liệu quan sát → dùng **fallback** (thống kê fit trên
  toàn bộ tập dữ liệu).
- Với cột phân loại → **rolling mode** (mode của tối đa 3 dòng trước).

> **Lưu ý về dữ liệu thật:** bảng CSV gốc **không có ô trống/NaN nào** (tổng
> số ô trống = 0). Vì vậy bước này được triển khai đầy đủ nhưng không ghi đè
> giá trị nào của dữ liệu thật. Script có thêm phần *minh hoạ* trên một **bản
> sao** cố ý chừa trống 9 ô để chứng minh cơ chế hoạt động đúng:
> 6 ô được điền bằng rolling mean, 3 ô dùng fallback.

#### (2) Chuyển cột kiểu Text sang chữ thường

Cột `RiskLevel` được chuyển sang chữ thường bằng `.str.strip()` + `.str.lower()`.

> Cột `RiskLevel` vốn đã viết thường (`high risk`, `low risk`, `mid risk`) nên
> phép chuyển đổi là **idempotent** — giá trị không đổi. Hàm vẫn được áp dụng
> đầy đủ cho mọi cột kiểu Text.

### 4. Mod / Mean / Var / Quartile và xử lý ngoại lệ

Cột được chọn: **`HeartRate`** (có ngoại lệ rõ ràng và dễ giải thích).

| Chỉ số | Giá trị |
|--------|---------|
| Count | 1014 |
| **Mod (mode)** | 70 (tần suất 271) |
| **Mean** | 74.3018 |
| **Var** (mẫu, `ddof=1`) | 65.4271 |
| Std | 8.0887 |
| Min / Max | 7 / 90 |
| **Quartile** | Q1 = 70, Q2 = 76, Q3 = 80, IQR = 10 |

**Phát hiện ngoại lệ bằng IQR:**

$$
\text{lower} = Q_1 - 1.5 \times IQR = 70 - 15 = 55
\qquad
\text{upper} = Q_3 + 1.5 \times IQR = 80 + 15 = 95
$$

→ **2 giá trị ngoại lệ**, cả hai đều bằng `7` bpm (trong khi mode là 70 bpm),
nằm ở index **499** và **908**.

**Xử lý (phương án 2 trong đề):** thay thế ngoại lệ bằng **rolling mean** —
giá trị mới là trung bình của các giá trị quan sát hợp lệ trong 3 dòng trước.

| Index | Giá trị cũ | Giá trị mới (rolling mean) |
|-------|-----------|------------------------------|
| 499 | 7 | 74.3333 |
| 908 | 7 | 74.3333 |

→ **0 ngoại lệ còn lại** sau khi xử lý.

> Phương án (1) trong đề (xoá hàng chứa ngoại lệ) cũng được hỗ trợ qua
> `find_iqr_outliers()` trả về đúng các dòng chứa ngoại lệ kèm index để có
> thể xoá thủ công.

### Lưu ý quan trọng về dữ liệu

Bảng gốc có **562 dòng trùng hoàn toàn** (55% số dòng) do thu thập lặp.
Theo yêu cầu đề bài, các dòng này được **giữ nguyên** để không thay đổi dữ
liệu gốc, nhưng cần lưu ý khi diễn giải điểm số mô hình ở Câu 2.

---

## CÂU 2 — Phân loại với 4 mô hình

### Chuẩn bị dữ liệu

| Thành phần | Lựa chọn | Lý do |
|------------|----------|-------|
| Nhãn | `RiskLevel` (3 lớp) | Cột phân loại duy nhất, lớp lặp lại đủ nhiều |
| Đặc trưng | 6 cột số (`Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`) | Toàn bộ biến lâm sàng có ý nghĩa dự đoán |
| Mã hoá nhãn | `LabelEncoder` | `high risk`=0, `low risk`=1, `mid risk`=2 |
| Chia dữ liệu | `train_test_split(test_size=0.25, stratify=y, random_state=42)` | 760 train / 254 test, giữ đúng tỉ lệ lớp |
| Chuẩn hoá | `StandardScaler` (KNN, Tree, AdaBoost)<br>`MinMaxScaler` (Naive Bayes) | KNN dựa trên khoảng cách → cần cùng thang đo.<br>`MultinomialNB` yêu cầu đặc trưng **không âm** → `MinMaxScaler` |

Scaler được đặt **trong `Pipeline`** nên được fit trên tập train → không rò rỉ
dữ liệu test.

### Kết quả đánh giá (tập test, 254 mẫu)

| Mô hình | Accuracy | Precision | Recall/Sensitivity | F1 (macro) | AUC (macro OVR) |
|---------|----------|-----------|--------------------|------------|-----------------|
| **KNN** (k=5) | 0.6890 | 0.7120 | 0.6893 | 0.6982 | 0.8644 |
| **Naive Bayes** (MultinomialNB) | 0.5315 | 0.6290 | 0.4914 | 0.4780 | 0.7109 |
| **Decision Tree** (max_depth=6) | **0.7205** | **0.7601** | **0.7152** | **0.7139** | **0.8695** |
| **AdaBoost** (200 bước, lr=0.5) | 0.7008 | 0.7210 | 0.6961 | 0.6859 | 0.8334 |

### Confusion matrix

**KNN**

| | high risk | low risk | mid risk |
|---|---|---|---|
| **high risk** | 50 | 7 | 11 |
| **low risk** | 3 | 74 | 25 |
| **mid risk** | 4 | 29 | 51 |

**Naive Bayes**

| | high risk | low risk | mid risk |
|---|---|---|---|
| **high risk** | 23 | 29 | 16 |
| **low risk** | 0 | 94 | 8 |
| **mid risk** | 1 | 65 | 18 |

**Decision Tree** (tốt nhất)

| | high risk | low risk | mid risk |
|---|---|---|---|
| **high risk** | 58 | 4 | 6 |
| **low risk** | 0 | 93 | 9 |
| **mid risk** | 2 | 50 | 32 |

**AdaBoost**

| | high risk | low risk | mid risk |
|---|---|---|---|
| **high risk** | 58 | 3 | 7 |
| **low risk** | 0 | 92 | 10 |
| **mid risk** | 6 | 50 | 28 |

### Phân tích lỗi nhầm lẫn

Lỗi tập trung ở một chỗ: **nhầm `mid risk` thành `low risk`** (50 mẫu với
Decision Tree, 50 mẫu với AdaBoost, 65 mẫu với Naive Bayes). Nguyên nhân là
`mid risk` là lớp trung gian, ranh giới với `low risk` không rõ nét trên thang
đo huyết áp/đường huyết. Ngược lại, lớp `high risk` được nhận dạng rất tốt
(precision 0.97 với Decision Tree, AUC 0.94–0.97 ở cả 4 mô hình) — đây là
kết quả quan trọng về mặt lâm sàng vì bỏ sót nguy cơ cao là lỗi nghiêm trọng.

### Kết luận

> **Decision Tree** là mô hình tốt nhất (Macro F1 = 0.7139, AUC = 0.8695) và
> được kiến nghị nộp bài: vừa có điểm F1 cao nhất vừa cho phép giải thích
> được bằng các biến lâm sàng quen thuộc. **Naive Bayes** đứng cuối bảng
> (Macro F1 = 0.4780) vì giả định độc lập giữa các đặc trưng không còn đúng
> khi `SystolicBP` và `DiastolicBP` tương quan 0.787.

Bản kết luận đầy đủ do script tự sinh nằm ở
[`output/cau2_ket_luan.txt`](./output/cau2_ket_luan.txt).

---

## Ghi chú kỹ thuật

- Phiên bản: Python 3.13, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1,
  matplotlib 3.11.2, seaborn 0.13.2.
- `matplotlib.use("Agg")` để script chạy được cả khi không mở cửa sổ đồ họa.
- Console Windows mặc định dùng `cp1252` nên cả 3 script đều ép
  `stdout.reconfigure(encoding="utf-8")` để in được tiếng Việt có dấu.
