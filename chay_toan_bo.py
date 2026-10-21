# -*- coding: utf-8 -*-
r"""
CHẠY TOÀN BỘ BÀI KIỂM TRA (CÂU 1 + CÂU 2)
=========================================
Script này gọi lần lượt Câu 1 (làm sạch + trực quan) và Câu 2 (phân loại
4 mô hình), rồi tự động mở thư mục `output/` để bạn xem kết quả.

Cách chạy:
    .venv\Scripts\python.exe bai-kiem-tra-cuoi-ky\chay_toan_bo.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Bảo đảm in được tiếng Việt ra console Windows (mặc định là cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent

# Danh sách script cần chạy theo thứ tự.
SCRIPTS = [
    "cau1_lam_sach_du_lieu.py",
    "cau2_phan_loai_4_mo_hinh.py",
]


def main() -> int:
    for script_name in SCRIPTS:
        script_path = BASE_DIR / script_name
        print("\n" + "#" * 78)
        print(f"# CHẠY: {script_name}")
        print("#" * 78)
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR.parent),
            # Truyền môi trường UTF-8 cho tiến trình con.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if completed.returncode != 0:
            print(f"LỖI: {script_name} thất bại với mã thoát {completed.returncode}.")
            return completed.returncode

    output_dir = BASE_DIR / "output"
    print("\n" + "=" * 78)
    print("TẤT CẢ KẾT QUẢ ĐÃ ĐƯỢC TẠO TRONG:", output_dir)
    print("=" * 78)
    for item in sorted(output_dir.iterdir()):
        print(f"  - {item.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
