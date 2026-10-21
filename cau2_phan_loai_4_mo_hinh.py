# -*- coding: utf-8 -*-
r"""
CÂU 2 (2.0 điểm) — PHÂN LOẠI VỚI 4 MÔ HÌNH VÀ ĐO HIỆU SUẤT
=========================================================
Bộ dữ liệu: "Maternal Health Risk Data Set - Modi.csv" (đã làm sạch ở Câu 1).

Nội dung thực hiện:
  1. Chọn cột phù hợp: nhãn = `RiskLevel` (3 lớp: high risk / low risk / mid risk);
     đặc trưng = 6 cột số (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate).
  2. Chuẩn hoá dữ liệu TRƯỚC khi đưa vào mô hình (StandardScaler cho các mô hình
     dựa trên khoảng cách/hệ số; MinMaxScaler cho Naive Bayes vì MultinomialNB
     yêu cầu đặc trưng không âm).
  3. Chia train/test với `train_test_split(stratify=...)`.
  4. Huấn luyện và đánh giá 4 mô hình:
        (1) K-Nearest Neighbors (KNN)
        (2) Naive Bayes (MultinomialNB)
        (3) Decision Tree
        (4) AdaBoost Classifier
     Với mỗi mô hình:
        - Confusion matrix
        - Accuracy, Precision, Recall/Sensitivity, F1, AUC
  5. Trực quan ma trận nhầm lẫn và đường ROC của cả 4 mô hình.

Cách chạy:
    .venv\Scripts\python.exe bai-kiem-tra-cuoi-ky\cau2_phan_loai_4_mo_hinh.py
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

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# 0. CẤU HÌNH
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
CSV_PATH = (
    WORKSPACE_DIR
    / "data-cleaning-ml"
    / "test"
    / "Maternal Health Risk Data Set - Modi.csv"
)
# Nếu đã chạy Câu 1 thì dùng bản đã làm sạch; nếu chưa có thì dùng bản gốc.
CLEANED_CSV_PATH = BASE_DIR / "output" / "Maternal_Health_Risk_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "RiskLevel"
FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]
TEST_SIZE = 0.25
RANDOM_STATE = 42
KNN_NEIGHBORS = 5


# ---------------------------------------------------------------------------
# 1. NẠP VÀ CHUẨN BỊ DỮ LIỆU
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Nạp bản dữ liệu đã làm sạch (nếu có), ngược lại dùng bản gốc."""
    path = CLEANED_CSV_PATH if CLEANED_CSV_PATH.exists() else CSV_PATH
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    print(f"[Nạp dữ liệu] {path.name} | {len(frame)} dòng x {len(frame.columns)} cột")
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Tách đặc trưng / nhãn và mã hoá nhãn chữ thành số bằng LabelEncoder.

    - Nhãn: `RiskLevel` (text) -> 0/1/2 theo thứ tự alphabet của LabelEncoder.
    - Đặc trưng: 6 cột numeric đã được chuẩn hoá trong Pipeline (không rò rỉ
      dữ liệu test vì scaler được fit trên tập train).
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Thiếu cột đặc trưng: {missing}")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Thiếu cột nhãn '{TARGET_COLUMN}'.")

    x = frame[FEATURE_COLUMNS].copy()
    # Đảm bảo kiểu số cho toàn bộ đặc trưng.
    for column in FEATURE_COLUMNS:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.dropna().reset_index(drop=True)

    y_raw = frame.loc[x.index, TARGET_COLUMN].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print("\n[Chọn cột phù hợp]")
    print(f"   Nhãn (target)  : {TARGET_COLUMN} -> {list(label_encoder.classes_)}")
    print(f"   Đặc trưng      : {FEATURE_COLUMNS}")
    print(f"   Phân bố lớp (trước khi chia):")
    print(pd.Series(y_raw).value_counts().to_string())

    return x, pd.Series(y, index=x.index), label_encoder


def split_data(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chia dữ liệu train/test với `train_test_split`.
    Dùng `stratify=y` để giữ đúng tỉ lệ các lớp rủi ro ở cả hai tập.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[Chia dữ liệu] train = {len(x_train)} dòng | test = {len(x_test)} dòng")
    print(f"   Phân bố lớp trong tập test: {pd.Series(y_test).value_counts().sort_index().to_dict()}")
    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA 4 MÔ HÌNH
# ---------------------------------------------------------------------------


def build_models() -> list[tuple[str, Pipeline, str]]:
    """
    Xây dựng 4 mô hình phân loại, mỗi mô hình kèm sẵn bước chuẩn hoá.

    Vì sao chuẩn hoá?
      - KNN đo khoảng cách giữa các điểm: nếu không chuẩn hoá, cột có giá trị
        lớn (ví dụ Age ~ 70) sẽ lấn át các cột khác.
      - MultinomialNB yêu cầu đặc trưng KHÔNG ÂM -> dùng MinMaxScaler.
      - Decision Tree và AdaBoost (cây gốc) không nhạy với thang đo, nhưng
        dùng chung StandardScaler vẫn vô hại và giữ 4 mô hình đồng nhất.
    """
    models: list[tuple[str, Pipeline, str]] = [
        (
            "KNN (K-Nearest Neighbors)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
                ]
            ),
            "k = 5",
        ),
        (
            "Naive Bayes (MultinomialNB)",
            Pipeline(
                [
                    ("scaler", MinMaxScaler(clip=True)),
                    ("classifier", MultinomialNB(alpha=0.1)),
                ]
            ),
            "alpha = 0.1 (Laplacian smoothing)",
        ),
        (
            "Decision Tree",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        DecisionTreeClassifier(
                            criterion="gini",
                            max_depth=6,
                            min_samples_leaf=5,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Gini, max_depth = 6, min_samples_leaf = 5",
        ),
        (
            "AdaBoost",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
                            n_estimators=200,
                            learning_rate=0.5,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "200 bước, learning_rate = 0.5, cây gốc depth = 1",
        ),
    ]
    return models


# ---------------------------------------------------------------------------
# 3. ĐÁNH GIÁ MÔ HÌNH
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
    parameter_note: str,
) -> dict:
    """
    Huấn luyện một mô hình, in confusion matrix + các độ đo hiệu suất
    và trả về kết quả dạng dictionary.
    """
    class_names = [str(name_) for name_ in label_encoder.classes_]

    # Huấn luyện trên tập train.
    pipeline.fit(x_train, y_train)

    # Dự đoán trên tập test.
    y_pred = pipeline.predict(x_test)

    # Dự đoán xác suất để tính AUC.
    y_proba = pipeline.predict_proba(x_test)

    # ---- Confusion matrix --------------------------------------------
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    print("\n" + "=" * 78)
    print(f"MÔ HÌNH: {name}")
    print(f"Tham số: {parameter_note}")
    print("=" * 78)
    print("Confusion matrix (hàng = nhãn thật, cột = nhãn dự đoán):")
    print(pd.DataFrame(matrix, index=class_names, columns=class_names).to_string())

    # ---- Các độ đo hiệu suất -----------------------------------------
    # classification_report: in tổng hợp precision/recall/F1 cho từng lớp.
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    # accuracy: tỉ lệ dự đoán đúng trên toàn bộ tập test.
    accuracy = float(accuracy_score(y_test, y_pred))
    # precision: trong số dự đoán là lớp X, bao nhiêu đúng thật sự là lớp X.
    precision = float(
        precision_score(y_test, y_pred, average="macro", zero_division=0)
    )
    # recall / sensitivity: trong số thực sự thuộc lớp X, bao nhiêu bị tìm ra.
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    # f1: trung bình điều hoà giữa precision và recall.
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # AUC: với bài đa lớp dùng macro trung bình theo "one-vs-rest" (OVR).
    auc_value = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    # AUC của từng lớp (one-vs-rest) để in chi tiết.
    per_class_auc = {}
    for position, class_name in enumerate(class_names):
        binary_true = (y_test == position).astype(int)
        binary_proba = y_proba[:, position]
        if len(np.unique(binary_true)) < 2:
            per_class_auc[class_name] = float("nan")
        else:
            per_class_auc[class_name] = float(roc_auc_score(binary_true, binary_proba))

    print(f"\nĐộ đo hiệu suất (tập test, {len(y_test)} mẫu):")
    print(f"   Accuracy                       : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"   Precision (macro)              : {precision:.4f}")
    print(f"   Recall / Sensitivity (macro)   : {recall:.4f}")
    print(f"   F1 (macro)                     : {f1:.4f}")
    print(f"   AUC (macro OVR)                : {auc_value:.4f}")
    for class_name, value in per_class_auc.items():
        print(f"      - AUC lớp '{class_name}'{'':>2}: {value:.4f}")

    # Kiểm tra cụ thể: số mẫu dự đoán sai và các cặp nhãn hay nhầm.
    errors = int(matrix.sum() - np.trace(matrix))
    print(f"\n   Số mẫu dự đoán sai: {errors}/{len(y_test)}")
    for row_index, class_name in enumerate(class_names):
        for column_index, predicted_name in enumerate(class_names):
            if row_index != column_index and matrix[row_index, column_index] > 0:
                print(
                    f"      Nhầm '{class_name}' -> '{predicted_name}': "
                    f"{matrix[row_index, column_index]} mẫu"
                )

    return {
        "name": name,
        "parameters": parameter_note,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_value,
        "per_class_auc": per_class_auc,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "y_test": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "y_proba": y_proba,
    }


# ---------------------------------------------------------------------------
# 4. TRỰC QUAN
# ---------------------------------------------------------------------------


def plot_confusion_matrices(results: list[dict], output_path: Path) -> None:
    """Vẽ confusion matrix của cả 4 mô hình trên một lưới 2x2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    for axis, result in zip(axes, results):
        class_names = result["class_names"]
        matrix = np.asarray(result["confusion_matrix"])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_title(
            f"{result['name']}\nAccuracy = {result['accuracy']:.4f} | F1 = {result['f1']:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel("Nhãn dự đoán")
        axis.set_ylabel("Nhãn thật")

    fig.suptitle(
        "MA TRẬN NHẦM LẪN (CONFUSION MATRIX) CỦA 4 MÔ HÌNH",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Confusion matrix] Đã lưu: {output_path.name}")


def plot_roc_curves(results: list[dict], output_path: Path) -> None:
    """Vẽ đường ROC (one-vs-rest) của cả 4 mô hình."""
    fig, axis = plt.subplots(figsize=(9, 7))
    colors = ["#2E6FBA", "#C0392B", "#1E8449", "#8E44AD"]

    for color, result in zip(colors, results):
        y_test = result["y_test"]
        y_proba = result["y_proba"]
        for position, class_name in enumerate(result["class_names"]):
            binary_true = (y_test == position).astype(int)
            false_positive_rate, true_positive_rate, _ = roc_curve(binary_true, y_proba[:, position])
            single_auc = float(auc(false_positive_rate, true_positive_rate))
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                linewidth=1.7,
                label=f"{result['name']} – lớp {class_name} (AUC = {single_auc:.3f})",
            )

    axis.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Ngẫu nhiên")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ĐƯỜNG ROC CỦA 4 MÔ HÌNH (ONE-VS-REST)", fontsize=14, fontweight="bold")
    axis.legend(loc="lower right", fontsize=8.5)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ROC] Đã lưu: {output_path.name}")


def plot_metric_comparison(results: list[dict], output_path: Path) -> None:
    """Vẽ biểu đồ cột so sánh 5 độ đo hiệu suất giữa 4 mô hình."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels = ["Accuracy", "Precision", "Recall/Sens.", "F1", "AUC"]
    frame = pd.DataFrame(
        {
            result["name"]: {metric: result[metric] for metric in metrics}
            for result in results
        }
    )

    fig, axis = plt.subplots(figsize=(13, 6))
    frame.plot(kind="bar", ax=axis, width=0.8, color=["#2E6FBA", "#C0392B", "#1E8449", "#8E44AD"])
    axis.set_ylabel("Giá trị")
    axis.set_title(
        "SO SÁNH HIỆU SUẤT 4 MÔ HÌNH PHÂN LOẠI RỦI RO MẸ",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right", fontsize=9)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=7.5, rotation=90, padding=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[So sánh độ đo] Đã lưu: {output_path.name}")


# ---------------------------------------------------------------------------
# 5. TỔNG KẾT VÀ KẾT LUẬN
# ---------------------------------------------------------------------------


def build_conclusion(results: list[dict]) -> str:
    """
    Tự động sinh kết luận bằng tiếng Việt dựa trên kết quả của 4 mô hình.

    Nội dung nêu bật: mô hình tốt nhất, mô hình tệ nhất, phân tích lỗi nhầm lẫn
    và kiến nghị chọn mô hình cuối cùng.
    """
    ranked = sorted(results, key=lambda result: result["f1"], reverse=True)

    # Mô hình tốt nhất: ưu tiên theo Macro F1, nếu bằng thì theo Accuracy.
    best = max(results, key=lambda result: (result["f1"], result["accuracy"]))
    worst = min(results, key=lambda result: (result["f1"], result["accuracy"]))

    def confusion_pairs(result: dict) -> list[tuple[str, str, int]]:
        matrix = np.asarray(result["confusion_matrix"])
        class_names = result["class_names"]
        pairs: list[tuple[str, str, int]] = []
        for row_index, actual in enumerate(class_names):
            for column_index, predicted in enumerate(class_names):
                if row_index != column_index and matrix[row_index, column_index] > 0:
                    pairs.append((actual, predicted, int(matrix[row_index, column_index])))
        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs

    best_pairs = confusion_pairs(best)
    worst_pairs = confusion_pairs(worst)

    best_errors = int(np.asarray(best["confusion_matrix"]).sum() - np.trace(np.asarray(best["confusion_matrix"])))
    worst_errors = int(np.asarray(worst["confusion_matrix"]).sum() - np.trace(np.asarray(worst["confusion_matrix"])))

    conclusion = (
        f"Dựa vào kết quả thực nghiệm của 4 mô hình phân loại trên tập dữ liệu "
        f"'{TARGET_COLUMN}' đã được làm sạch, ta nhận thấy mô hình "
        f"**{best['name']}** hoạt động hiệu quả nhất với Macro F1 = {best['f1']:.4f}, "
        f"Accuracy = {best['accuracy']:.4f} và AUC = {best['auc']:.4f}; "
        f"trong khi mô hình **{worst['name']}** đứng cuối bảng với Macro F1 = {worst['f1']:.4f} "
        f"và AUC = {worst['auc']:.4f}. "
    )

    if best_pairs:
        top_pair = best_pairs[0]
        conclusion += (
            f"Nhìn vào ma trận nhầm lẫn của {best['name']}, lỗi phổ biến nhất là "
            f"nhầm lớp '{top_pair[0]}' thành '{top_pair[1]}' với {top_pair[2]} mẫu; "
            f"đây là mối quan hệ dễ xảy ra về mặt y tế, bởi các mức nguy cơ "
            f"'low risk' và 'mid risk' nằm sát nhau trên thang đo huyết áp và đường huyết, "
            f"nên chỉ cần lệch nhẹ cũng đủ để mô hình chọn nhầm lớp. "
        )
    if worst_pairs:
        top_pair = worst_pairs[0]
        conclusion += (
            f"Còn {worst['name']} dự đoán sai {worst_errors} mẫu, tập trung ở cặp "
            f"nhầm lẫn '{top_pair[0]}' -> '{top_pair[1]}' ({top_pair[2]} mẫu), "
            f"cho thấy ranh giới quyết định của mô hình này kém sắc bén hơn. "
        )

    conclusion += (
        f"Điểm đáng chú ý là độ chênh lệch hiệu suất giữa 4 mô hình không lớn "
        f"(Macro F1 nằm trong khoảng {min(result['f1'] for result in results):.4f} - "
        f"{max(result['f1'] for result in results):.4f}), nên về mặt kỹ thuật chưa có "
        f"mô hình nào vượt trội hẳn. Với mô hình được chọn làm đại diện, "
        f"{best['name']} vừa có điểm F1 cao nhất vừa cho phép giải thích được "
        f"(dùng các biến lâm sàng quen thuộc như huyết áp tâm thu, huyết áp tâm trương, "
        f"đường huyết và nhịp tim), nên kiến nghị chọn {best['name']} để nộp bài. "
        f"Tuy nhiên, cần lưu ý rằng khoảng 55% số dòng trong tập dữ liệu là dòng trùng "
        f"hoàn toàn, do đó điểm số thu được có thể bị đánh giá cao hơn thực tế; "
        f"kiến nghị kiểm định thêm bằng cross-validation hoặc trên tập dữ liệu mới trước "
        f"khi áp dụng vào bài toán thực tế."
    )
    return conclusion


# ---------------------------------------------------------------------------
# 6. CHƯƠNG TRÌNH CHÍNH
# ---------------------------------------------------------------------------


def main() -> int:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 78)
    print("CÂU 2 — PHÂN LOẠI VỚI 4 MÔ HÌNH: KNN, NAIVE BAYES, DECISION TREE, ADABOOST")
    print("=" * 78)

    if not CSV_PATH.exists():
        print(f"LỖI: không tìm thấy file CSV tại {CSV_PATH}")
        return 1

    frame = load_data()
    x, y, label_encoder = prepare_features(frame)
    x_train, x_test, y_train, y_test = split_data(x, y)

    models = build_models()
    results: list[dict] = []

    for name, pipeline, parameter_note in models:
        result = evaluate_model(
            name,
            pipeline,
            x_train,
            x_test,
            y_train,
            y_test,
            label_encoder,
            parameter_note,
        )
        results.append(result)

    # ---- Trực quan ----------------------------------------------------
    print("\n" + "=" * 78)
    print("TRỰC QUAN KẾT QUẢ")
    print("=" * 78)
    plot_confusion_matrices(results, OUTPUT_DIR / "cau2_confusion_matrix_4_mo_hinh.png")
    plot_roc_curves(results, OUTPUT_DIR / "cau2_duong_roc_4_mo_hinh.png")
    plot_metric_comparison(results, OUTPUT_DIR / "cau2_so_sanh_do_do_4_mo_hinh.png")

    # ---- Bảng tổng hợp ----------------------------------------------
    summary = pd.DataFrame(
        [
            {
                "Mô hình": result["name"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall/Sensitivity": result["recall"],
                "F1 (macro)": result["f1"],
                "AUC (macro OVR)": result["auc"],
            }
            for result in results
        ]
    ).set_index("Mô hình")
    print("\n" + "=" * 78)
    print("BẢNG TỔNG HỢP HIỆU SUẤT 4 MÔ HÌNH")
    print("=" * 78)
    print(summary.round(4).to_string())

    # ---- Kết luận tự động --------------------------------------------
    conclusion = build_conclusion(results)
    print("\n" + "=" * 78)
    print("KẾT LUẬN")
    print("=" * 78)
    print(conclusion)

    # ---- Lưu kết quả --------------------------------------------------
    summary.to_csv(OUTPUT_DIR / "cau2_bang_tong_hop_hieu_suat.csv", encoding="utf-8-sig")
    conclusion_path = OUTPUT_DIR / "cau2_ket_luan.txt"
    # utf-8-sig để Notepad/PowerShell trên Windows hiển thị đúng tiếng Việt.
    conclusion_path.write_text(conclusion, encoding="utf-8-sig")
    print(f"\n[Đã lưu] Bảng tổng hợp: cau2_bang_tong_hop_hieu_suat.csv")
    print(f"[Đã lưu] Kết luận: {conclusion_path.name}")

    print("\n" + "=" * 78)
    print("HOÀN THÀNH CÂU 2")
    print("=" * 78)
    for item in sorted(OUTPUT_DIR.glob("cau2_*")):
        print(f"  - {item.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
