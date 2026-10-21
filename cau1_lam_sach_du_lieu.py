# -*- coding: utf-8 -*-
r"""
CÂU 1 (1.0 điểm) — TRỰC QUAN VÀ LÀM SẠCH DỮ LIỆU
=================================================
Bộ dữ liệu: "Maternal Health Risk Data Set - Modi.csv"
          (data-cleaning-ml/test/Maternal Health Risk Data Set - Modi.csv)

Nội dung thực hiện trong script này:
  1. Trực quan histogram tối thiểu 2 cột dữ liệu (thực tế: 6 cột số).
  2. Trực quan ma trận tương quan của TẤT CẢ các cột kiểu numeric.
  3. Làm sạch dữ liệu:
       (3.1) Thay thế giá trị trống/NaN bằng giá trị khác dựa trên kỹ thuật
             Rolling Mean (cửa sổ lùi tối đa 3 dòng đã quan sát).
       (3.2) Chuyển tối thiểu 1 cột kiểu Text sang chữ thường (lower case).
  4. Tính Mod / Mean / Var / Quartile cho cột numeric, rồi xử lý giá trị
     ngoại lệ (IQR): thay thế ngoại lệ bằng giá trị khác theo kỹ thuật
     Rolling Mean (phương án (2); phương án (1) xoá hàng cũng có sẵn).

Cách chạy:
    .venv\Scripts\python.exe bai-kiem-tra-cuoi-ky\cau1_lam_sach_du_lieu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bảo đảm in được tiếng Việt ra console Windows (mặc định là cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

# Dùng backend "Agg" để script chạy được cả khi không mở cửa sổ đồ họa.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 0. CẤU HÌNH ĐƯỜNG DẪN VÀ HẰNG SỐ
# ---------------------------------------------------------------------------

# Thư mục chứa script hiện tại (dùng để dựng đường dẫn tương đối ổn định).
BASE_DIR = Path(__file__).resolve().parent

# Thư mục gốc của workspace (nơi chứa thư mục "data-cleaning-ml").
WORKSPACE_DIR = BASE_DIR.parent

# Đường dẫn tới file CSV đầu vào.
CSV_PATH = (
    WORKSPACE_DIR
    / "data-cleaning-ml"
    / "test"
    / "Maternal Health Risk Data Set - Modi.csv"
)

# Thư mục lưu kết quả (ảnh biểu đồ, CSV đã làm sạch, báo cáo thống kê).
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kích thước cửa sổ rolling: tối đa 3 dòng đã quan sát trước ô đang trống.
ROLLING_WINDOW = 3

# Cột numeric được chọn để tính Mod/Mean/Var/Quartile và xử lý ngoại lệ.
# HeartRate có ngoại lệ rõ ràng (7 bpm trong khi mode = 70 bpm) -> rất dễ
# trực quan và giải thích khi thay bằng Rolling Mean.
STAT_COLUMN = "HeartRate"

# Cột kiểu Text được chuyển sang chữ thường (yêu cầu 3.2).
TEXT_COLUMN = "RiskLevel"

# Bật/tắt phần minh hoạ kỹ thuật Rolling Mean (xem giải thích trong hàm).
DEMONSTRATE_ROLLING = True


# ---------------------------------------------------------------------------
# 1. CÁC HÀM TIỆN ÍCH
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> pd.DataFrame:
    """Đọc file CSV (có BOM UTF-8) và trả về DataFrame thô."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    # Chuẩn hóa tên cột: bỏ khoảng trắng thừa ở hai đầu.
    frame.columns = [str(column).strip() for column in frame.columns]
    print(f"[Đọc dữ liệu] {path.name}")
    print(f"           Số dòng: {len(frame)} | Số cột: {len(frame.columns)}")
    print(f"           Kiểu dữ liệu:\n{frame.dtypes.to_string()}")
    return frame


def describe_dataframe(frame: pd.DataFrame) -> None:
    """In tổng quan thiếu giá trị và số giá trị khác nhau của từng cột."""
    print("\n" + "=" * 78)
    print("TỔNG QUAN DỮ LIỆU THÔ (số ô trống theo từng cột)")
    print("=" * 78)
    missing = frame.isna().sum()
    summary = pd.DataFrame(
        {
            "so_o_trong": missing,
            "so_gia_tri_khac_nhau": frame.nunique(dropna=True),
            "kieu_du_lieu": frame.dtypes.astype(str),
        }
    )
    print(summary.to_string())
    print(f"\nTổng số ô trống/NaN trong toàn bộ bảng: {int(frame.isna().sum().sum())}")


def describe_numeric_column(values: pd.Series) -> dict:
    """
    Tính Mod (mode), Mean, Var (phương sai mẫu) và Quartile cho một cột số.

    Trả về dictionary để tiện in ra và dùng lại cho phần xử lý ngoại lệ.
    """
    observed = pd.to_numeric(values, errors="coerce").dropna()

    # Mode: có thể có nhiều giá trị cùng tần suất cao nhất -> lấy tất cả.
    frequencies = observed.value_counts()
    max_frequency = int(frequencies.max())
    modes = sorted(frequencies[frequencies == max_frequency].index.tolist())

    return {
        "column": str(values.name),
        "count": int(observed.size),
        "mode": modes,
        "mode_frequency": max_frequency,
        "mean": float(observed.mean()),
        "variance": float(observed.var(ddof=1)),      # phương sai mẫu
        "std": float(observed.std(ddof=1)),
        "min": float(observed.min()),
        "q1": float(observed.quantile(0.25)),
        "median": float(observed.quantile(0.50)),     # Q2
        "q3": float(observed.quantile(0.75)),
        "max": float(observed.max()),
        "iqr": float(observed.quantile(0.75) - observed.quantile(0.25)),
    }


def print_descriptive_stats(stats: dict) -> None:
    """In bảng thống kê mô tả của một cột số."""
    print("\n" + "-" * 78)
    print(f"THỐNG KÊ MÔ TẢ CỘT SỐ: {stats['column']}")
    print("-" * 78)
    print(f"  Số quan sát (count) : {stats['count']}")
    print(f"  Mod (mode)          : {stats['mode']}  (tần suất {stats['mode_frequency']})")
    print(f"  Mean (trung bình)   : {stats['mean']:.4f}")
    print(f"  Var  (phương sai)   : {stats['variance']:.4f}   (mẫu, ddof=1)")
    print(f"  Std  (độ lệch chuẩn): {stats['std']:.4f}")
    print(f"  Min / Max           : {stats['min']:.2f} / {stats['max']:.2f}")
    print(
        f"  Quartile            : Q1={stats['q1']:.2f} | Q2(median)={stats['median']:.2f}"
        f" | Q3={stats['q3']:.2f} | IQR={stats['iqr']:.2f}"
    )


def find_iqr_outliers(values: pd.Series) -> tuple[float, float, pd.Series]:
    """
    Tìm ngoại lệ bằng thuật toán IQR (Interquartile Range).

    Ngưỡng:  lower = Q1 - 1.5*IQR ,  upper = Q3 + 1.5*IQR
    Giá trị nằm ngoài [lower, upper] được coi là ngoại lệ.
    """
    observed = pd.to_numeric(values, errors="coerce").dropna()
    q1 = float(observed.quantile(0.25))
    q3 = float(observed.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (observed < lower) | (observed > upper)
    return lower, upper, observed[mask]


def rolling_mean_impute(
    values: pd.Series,
    window: int = ROLLING_WINDOW,
    fallback: float | None = None,
) -> tuple[pd.Series, int, int]:
    """
    Thay thế giá trị trống/NaN bằng giá trị khác dựa trên kỹ thuật ROLLING MEAN.

    Nguyên tắc quan trọng:
      - Chỉ dùng các giá trị ĐÃ QUAN SÁT trước ô đang trống (cửa sổ lùi tối đa
        `window` dòng), đồng thời loại bỏ chính các ô đang trống.
      - KHÔNG dùng giá trị tương lai.
      - KHÔNG dùng giá trị đã điền trước đó làm "lịch sử" (nhờ vậy đọc lại từ
        `values` gốc chứ không đọc từ `result`).
      - Nếu cửa sổ chưa có dữ liệu quan sát nào -> dùng `fallback`
        (thống kê fit trên tập dữ liệu).

    Trả về: (series đã điền, số ô điền bằng rolling window, số ô dùng fallback).
    """
    result = pd.to_numeric(values, errors="coerce").astype("float64")
    source = values.tolist()
    filled_by_window = 0
    filled_by_fallback = 0

    for position, value in enumerate(source):
        # Ô này đã có giá trị -> giữ nguyên, sang ô tiếp theo.
        if not pd.isna(value):
            continue

        # Lấy tối đa `window` giá trị quan sát ngay trước ô đang trống.
        previous = [
            item
            for item in source[max(0, position - window):position]
            if not pd.isna(item)
        ]

        if previous:
            result.iloc[position] = float(np.mean(previous))
            filled_by_window += 1
        else:
            # Chưa có lịch sử quan sát -> dùng thống kê fallback đã fit sẵn.
            result.iloc[position] = fallback
            filled_by_fallback += 1

    return result, filled_by_window, filled_by_fallback


def rolling_mean_replace_outliers(
    values: pd.Series,
    lower: float,
    upper: float,
    window: int = ROLLING_WINDOW,
) -> tuple[pd.Series, list[int]]:
    """
    Thay thế giá trị ngoại lệ (nằm ngoài ngưỡng IQR) bằng giá trị khác theo
    kỹ thuật ROLLING MEAN.

    - Với ô ngoại lệ: giá trị mới = trung bình của tối đa `window` giá trị
      quan sát hợp lệ ngay trước đó.
    - Nếu không còn giá trị quan sát hợp lệ phía trước -> dùng trung bình
      của các giá trị hợp lệ còn lại trong cột (fallback).
    - Giá trị đã thay thế KHÔNG được dùng lại làm lịch sử cho ô tiếp theo.

    Trả về: (series mới, danh sách index của các ô bị thay thế).
    """
    source = pd.to_numeric(values, errors="coerce").tolist()
    valid_pool = [
        item
        for item in source
        if pd.notna(item) and lower <= item <= upper
    ]
    fallback = float(np.mean(valid_pool)) if valid_pool else float(np.mean(pd.to_numeric(values, errors="coerce").dropna()))

    result = pd.to_numeric(values, errors="coerce").astype("float64").copy(deep=True)
    replaced_positions: list[int] = []

    for position, value in enumerate(source):
        if pd.isna(value):
            continue
        # Không phải ngoại lệ -> giữ nguyên.
        if lower <= value <= upper:
            continue

        previous_valid = [
            item
            for item in source[max(0, position - window):position]
            if pd.notna(item) and lower <= item <= upper
        ]
        new_value = float(np.mean(previous_valid)) if previous_valid else fallback
        result.iloc[position] = new_value
        replaced_positions.append(position)

    return result, replaced_positions


def lower_case_text_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Chuyển tối thiểu 1 cột kiểu Text sang chữ thường (lower case).

    Đồng thời bỏ khoảng trắng thừa ở hai đầu chuỗi cho sạch dữ liệu.
    """
    result = frame.copy(deep=True)
    if column not in result.columns:
        raise KeyError(f"Không tìm thấy cột chữ '{column}' trong dữ liệu.")
    before = result[column].astype(str).unique().tolist()
    result[column] = (
        result[column]
        .astype("string")
        .str.strip()      # bỏ khoảng trắng thừa
        .str.lower()      # chuyển sang chữ thường
    )
    after = result[column].astype(str).unique().tolist()
    print(f"\n[Chuyển chữ thường] Cột '{column}':")
    print(f"   Trước: {before}")
    print(f"   Sau  : {after}")
    if before == after:
        print(
            "   → Lưu ý: dữ liệu gốc vốn đã viết thường, nên phép chuyển đổi "
            "không làm thay đổi giá trị nào (thao tác này là idempotent). "
            "Hàm vẫn áp dụng `.str.strip()` + `.str.lower()` cho mọi cột kiểu Text."
        )
    return result


# ---------------------------------------------------------------------------
# 2. CÁC HÀM TRỰC QUAN
# ---------------------------------------------------------------------------


def plot_histograms(frame: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    """
    Vẽ histogram cho tối thiểu 2 cột dữ liệu (thực tế vẽ cho nhiều cột số).

    Mỗi cột có 2 trục: histogram + đường KDE để dễ hình dung hình dạng phân phối.
    """
    n_cols = len(columns)
    n_rows = int(np.ceil(n_cols / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for index, column in enumerate(columns):
        axis = axes[index]
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        sns.histplot(
            series,
            bins=20,
            kde=True,
            color="#2E6FBA",
            edgecolor="white",
            line_kws={"color": "#C0392B", "linewidth": 1.8},
            ax=axis,
        )
        axis.set_title(f"Histogram cột '{column}'", fontsize=12, fontweight="bold")
        axis.set_xlabel("Giá trị")
        axis.set_ylabel("Tần số")
        axis.grid(axis="y", alpha=0.3)
    # Ẩn các subplot dư nếu số cột không chia hết cho 3 cột.
    for position in range(n_cols, len(axes)):
        axes[position].set_visible(False)

    fig.suptitle(
        "TRỰC QUAN HISTOGRAM CÁC CỘT DỮ LIỆU SỐ",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Histogram] Đã lưu: {output_path.name}")


def plot_correlation_matrix(frame: pd.DataFrame, output_path: Path) -> None:
    """
    Vẽ ma trận tương quan (Pearson) của TẤT CẢ các cột có kiểu numeric.
    """
    numeric_frame = frame.select_dtypes(include=[np.number])
    correlation = numeric_frame.corr(method="pearson")

    fig, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan Pearson"},
        ax=axis,
    )
    axis.set_title(
        "MA TRẬN TƯƠNG QUAN CỦA TẤT CẢ CỘT KIỂU NUMERIC",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Ma trận tương quan] Đã lưu: {output_path.name}")
    print("  Ma trận tương quan Pearson:")
    print(correlation.round(3).to_string())


def plot_before_after_outliers(
    frame: pd.DataFrame,
    column: str,
    cleaned_values: pd.Series,
    lower: float,
    upper: float,
    output_path: Path,
) -> None:
    """So sánh phân phối cột số trước và sau khi xử lý ngoại lệ."""
    before = pd.to_numeric(frame[column], errors="coerce").dropna()
    after = pd.to_numeric(cleaned_values, errors="coerce").dropna()
    outliers = before[(before < lower) | (before > upper)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(before, bins=20, kde=True, color="#C0392B", ax=axes[0])
    # Đánh dấu vị trí các giá trị ngoại lệ để thấy rõ.
    for value in outliers:
        axes[0].axvline(value, color="#7B241C", linestyle="--", linewidth=1.8, alpha=0.9)
    axes[0].annotate(
        f"{len(outliers)} giá trị ngoại lệ\n(ngoài [{lower:.0f}, {upper:.0f}])",
        xy=(float(outliers.iloc[0]) if len(outliers) else lower, axes[0].get_ylim()[1] * 0.85),
        xytext=(18, 0),
        textcoords="offset points",
        fontsize=9,
        color="#7B241C",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#7B241C"},
    )
    axes[0].set_title(f"TRƯỚC khi xử lý ngoại lệ — '{column}'", fontweight="bold")
    axes[0].set_xlabel("Giá trị")
    axes[0].set_ylabel("Tần số")

    sns.histplot(after, bins=20, kde=True, color="#1E8449", ax=axes[1])
    axes[1].axvspan(lower, upper, color="#1E8449", alpha=0.10)
    axes[1].set_title(f"SẢU khi xử lý ngoại lệ — '{column}'", fontweight="bold")
    axes[1].set_xlabel("Giá trị")
    axes[1].set_ylabel("Tần số")

    for axis in axes:
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"XỬ LÝ GIÁ TRỊ NGOẠI LỆ (IQR) BẰNG KỸ THUẬT ROLLING MEAN — CỘT '{column}'",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Trước/Sau ngoại lệ] Đã lưu: {output_path.name}")


# ---------------------------------------------------------------------------
# 3. CHƯƠNG TRÌNH CHÍNH
# ---------------------------------------------------------------------------


def main() -> int:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 78)
    print("CÂU 1 — TRỰC QUAN VÀ LÀM SẠCH DỮ LIỆU")
    print("Maternal Health Risk Data Set (Modi)")
    print("=" * 78)

    if not CSV_PATH.exists():
        print(f"LỖI: không tìm thấy file CSV tại {CSV_PATH}")
        return 1

    # ------------------------------------------------------------------
    # BƯỚC 1: Nạp dữ liệu và thống kê tổng quan
    # ------------------------------------------------------------------
    raw = load_dataset(CSV_PATH)
    describe_dataframe(raw)

    numeric_columns = raw.select_dtypes(include=[np.number]).columns.tolist()
    text_columns = raw.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    print(f"\nCột kiểu numeric ({len(numeric_columns)}): {numeric_columns}")
    print(f"Cột kiểu text     ({len(text_columns)}): {text_columns}")

    # ------------------------------------------------------------------
    # BƯỚC 2: TRỰC QUAN — HISTOGRAM CHO TỐI THIỂU 2 CỘT
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHẦN 1 — TRỰC QUAN HISTOGRAM")
    print("=" * 78)
    plot_histograms(
        raw,
        numeric_columns,
        OUTPUT_DIR / "cau1_histogram_6_cot_so.png",
    )

    # ------------------------------------------------------------------
    # PHẦN 2: TRỰC QUAN — MA TRẬN TƯƠNG QUAN CỦA CÁC CỘT NUMERIC
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHẦN 2 — MA TRẬN TƯƠNG QUAN (TẤT CẢ CỘT NUMERIC)")
    print("=" * 78)
    plot_correlation_matrix(
        raw,
        OUTPUT_DIR / "cau1_ma_tran_tuong_quan.png",
    )

    # ------------------------------------------------------------------
    # PHẦN 3: LÀM SẠCH DỮ LIỆU
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHẦN 3 — LÀM SẠCH DỮ LIỆU")
    print("=" * 78)

    # 3.1 Thay thế giá trị trống/NaN bằng kỹ thuật Rolling Mean -----------
    print("\n[3.1] Thay thế giá trị trống/NaN bằng ROLLING MEAN")
    cleaned = raw.copy(deep=True)
    imputation_log: list[dict] = []

    for column in numeric_columns:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue
        # Fallback khi cửa sổ rolling chưa có dữ liệu: trung bình toàn cột
        # (thống kê fit trên toàn bộ tập dữ liệu).
        fallback = float(pd.to_numeric(cleaned[column], errors="coerce").mean())
        cleaned[column], by_window, by_fallback = rolling_mean_impute(
            cleaned[column],
            window=ROLLING_WINDOW,
            fallback=fallback,
        )
        imputation_log.append(
            {
                "column": column,
                "missing_before": missing_count,
                "filled_by_rolling_window": by_window,
                "filled_by_fallback": by_fallback,
            }
        )

    for column in text_columns:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue
        # Cột phân loại: rolling mode (mode của tối đa 3 dòng quan sát trước).
        source = cleaned[column].tolist()
        for position, value in enumerate(source):
            if not pd.isna(value):
                continue
            previous = [
                item
                for item in source[max(0, position - ROLLING_WINDOW):position]
                if not pd.isna(item)
            ]
            if previous:
                modes = pd.Series(previous).mode(dropna=True)
                cleaned.iat[position, cleaned.columns.get_loc(column)] = (
                    modes.iloc[0] if not modes.empty else "unknown"
                )
            else:
                cleaned.iat[position, cleaned.columns.get_loc(column)] = "unknown"

    if imputation_log:
        print(pd.DataFrame(imputation_log).to_string(index=False))
    else:
        print(
            "   → Bảng dữ liệu gốc KHÔNG có ô trống/NaN nào "
            f"(tổng ô trống = {int(raw.isna().sum().sum())})."
        )
        print(
            "   → Vì vậy bước Rolling Mean được triển khai đầy đủ nhưng không "
            "ghi đè bất kỳ giá trị nào của dữ liệu thật."
        )

    # Minh hoạ kỹ thuật Rolling Mean trên một bản sao có chủ đích chừa trống
    # vài ô (KHÔNG thay đổi dữ liệu thật) để chứng minh hàm hoạt động đúng.
    if DEMONSTRATE_ROLLING:
        print("\n   [MINH HOẠ] Áp dụng Rolling Mean lên bản sao cố ý chừa trống:")
        demo = cleaned.copy(deep=True)
        demo_rows = [0, 1, 2, 5, 10, 50, 100, 500, 900]
        for position in demo_rows:
            demo.iloc[position, demo.columns.get_loc(STAT_COLUMN)] = np.nan
        demo_filled, window_count, fallback_count = rolling_mean_impute(
            demo[STAT_COLUMN],
            window=ROLLING_WINDOW,
            fallback=float(demo[STAT_COLUMN].mean()),
        )
        demo_table = pd.DataFrame(
            {
                "index": demo_rows,
                "gia_tri_goc_bi_chua_trong": [raw.iloc[p][STAT_COLUMN] for p in demo_rows],
                "gia_tri_sau_khi_dien": [round(float(demo_filled.iloc[p]), 3) for p in demo_rows],
            }
        )
        print(demo_table.to_string(index=False))
        print(
            f"   → {window_count} ô được điền bằng rolling mean, "
            f"{fallback_count} ô dùng fallback (không có lịch sử quan sát)."
        )

    # 3.2 Chuyển tối thiểu 1 cột kiểu Text sang chữ thường ----------------
    print("\n[3.2] Chuyển cột kiểu Text sang chữ thường (lower case)")
    cleaned = lower_case_text_column(cleaned, TEXT_COLUMN)

    # ------------------------------------------------------------------
    # PHẦN 4: MOD / MEAN / VAR / QUARTILE + XỬ LÝ NGOẠI LỘE
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"PHẦN 4 — THỐNG KÊ MÔ TẢ VÀ XỬ LÝ NGOẠI LỘE (IQR) TRÊN CỘT '{STAT_COLUMN}'")
    print("=" * 78)

    stats = describe_numeric_column(cleaned[STAT_COLUMN])
    print_descriptive_stats(stats)

    # Thống kê mô tả cho tất cả các cột số (bổ sung cho trực quan).
    print("\n[4.1] Bảng thống kê mô tả của TOÀN BỘ cột số:")
    all_stats = pd.DataFrame(
        [
            describe_numeric_column(cleaned[column]) for column in numeric_columns
        ]
    ).set_index("column")
    display_stats = all_stats.copy()
    display_stats["mode"] = display_stats["mode"].apply(lambda item: ", ".join(str(v) for v in item))
    print(
        display_stats[
            ["count", "mode", "mode_frequency", "mean", "variance", "std", "min", "q1", "median", "q3", "max", "iqr"]
        ].round(4).to_string()
    )

    # 4.2 Phát hiện ngoại lệ bằng IQR trên cột đã chọn.
    lower, upper, outliers = find_iqr_outliers(cleaned[STAT_COLUMN])
    print(f"\n[4.2] Phát hiện ngoại lệ bằng IQR trên cột '{STAT_COLUMN}':")
    print(f"   Ngưỡng dưới = Q1 - 1.5*IQR = {lower:.4f}")
    print(f"   Ngưỡng trên  = Q3 + 1.5*IQR = {upper:.4f}")
    print(f"   Số giá trị ngoại lệ: {len(outliers)}")
    if len(outliers):
        print("   Các dòng chứa ngoại lệ (kèm index):")
        outlier_rows = cleaned.loc[outliers.index]
        print(outlier_rows.to_string())

    # 4.3 Xử lý ngoại lệ bằng Rolling Mean (phương án 2 trong đề bài).
    replaced_values, replaced_positions = rolling_mean_replace_outliers(
        cleaned[STAT_COLUMN],
        lower=lower,
        upper=upper,
        window=ROLLING_WINDOW,
    )
    print(f"\n[4.3] Thay thế {len(replaced_positions)} giá trị ngoại lệ bằng ROLLING MEAN:")
    if replaced_positions:
        replacement_table = pd.DataFrame(
            {
                "index": replaced_positions,
                "gia_tri_cu": [cleaned[STAT_COLUMN].iloc[p] for p in replaced_positions],
                "gia_tri_moi": [round(float(replaced_values.iloc[p]), 4) for p in replaced_positions],
            }
        )
        print(replacement_table.to_string(index=False))
    cleaned[STAT_COLUMN] = replaced_values

    # Kiểm tra lại: còn ngoại lệ nào không?
    _, _, remaining = find_iqr_outliers(cleaned[STAT_COLUMN])
    print(f"   → Số ngoại lệ còn lại sau khi xử lý: {len(remaining)}")

    plot_before_after_outliers(
        raw,
        STAT_COLUMN,
        cleaned[STAT_COLUMN],
        lower,
        upper,
        OUTPUT_DIR / "cau1_truoc_sau_xu_ly_ngoai_le.png",
    )

    # ------------------------------------------------------------------
    # PHẦN 5: LƯU KẾT QUẢ VÀ BÁO CÁO
    # ------------------------------------------------------------------
    cleaned_output_path = OUTPUT_DIR / "Maternal_Health_Risk_cleaned.csv"
    cleaned.to_csv(cleaned_output_path, index=False, encoding="utf-8-sig")
    print(f"\n[5.1] Đã lưu bảng dữ liệu đã làm sạch: {cleaned_output_path.name}")

    report_lines = [
        "CÂU 1 — BÁO CÁO LÀM SẠCH DỮ LIỆU (Maternal Health Risk Data Set - Modi)",
        "=" * 78,
        f"Số dòng trước : {len(raw)}",
        f"Số dòng sau   : {len(cleaned)}",
        f"Ô trống trước : {int(raw.isna().sum().sum())}",
        f"Ô trống sau   : {int(cleaned.isna().sum().sum())}",
        "",
        "1) Trực quan histogram: 6 cột số (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate).",
        "2) Trực quan ma trận tương quan Pearson của tất cả 6 cột numeric.",
        f"3) Điền khuyết bằng Rolling Mean (cửa sổ lùi {ROLLING_WINDOW} dòng): "
        + (
            "không có ô trống nào cần điền."
            if not imputation_log
            else "; ".join(
                f"{item['column']}: {item['missing_before']} ô"
                for item in imputation_log
            )
        ),
        f"4) Chuyển cột chữ '{TEXT_COLUMN}' sang chữ thường.",
        f"5) Cột '{STAT_COLUMN}': Mod={stats['mode']}, Mean={stats['mean']:.4f}, "
        f"Var={stats['variance']:.4f}, Q1={stats['q1']:.2f}, Q2={stats['median']:.2f}, Q3={stats['q3']:.2f}.",
        f"6) Ngưỡng IQR = [{lower:.4f}, {upper:.4f}]; số ngoại lệ = {len(outliers)}; "
        f"đã thay thế bằng Rolling Mean: {len(replaced_positions)}; còn lại: {len(remaining)}.",
        "",
        "Ghi chú: bảng dữ liệu có 562 dòng trùng hoàn toàn (55% số dòng) do thu thập lặp; "
        "theo yêu cầu đề bài, các dòng này được giữ nguyên để không thay đổi dữ liệu gốc.",
    ]
    report_path = OUTPUT_DIR / "cau1_bao_cao_lam_sach.txt"
    # utf-8-sig để Notepad/PowerShell trên Windows hiển thị đúng tiếng Việt.
    report_path.write_text("\n".join(report_lines), encoding="utf-8-sig")
    print(f"[5.2] Đã lưu báo cáo: {report_path.name}")

    print("\n" + "=" * 78)
    print("HOÀN THÀNH CÂU 1")
    print("=" * 78)
    for item in sorted(OUTPUT_DIR.glob("cau1_*")):
        print(f"  - {item.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
