from __future__ import annotations

import os
from pathlib import Path

import pathlib
import platform
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from services.plate_service import PlateRecognitionService

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent.parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"

# Tự động tạo thư mục kết quả nếu chưa có khi khởi chạy server
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(os.getenv("PLATE_MODEL_PATH", str(REPO_DIR / "runs" / "train" / "ket_qua_3202" / "weights" / "best.pt"))).resolve()
CHAR_MODEL_PATH = Path(
    os.getenv("PLATE_CHAR_MODEL_PATH", str(REPO_DIR / "runs" / "train" / "char36" / "weights" / "best.pt"))
).resolve()

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(STATIC_DIR))

# Khởi tạo service xử lý biển số
service = PlateRecognitionService(
    repo_dir=REPO_DIR,
    model_path=MODEL_PATH,
    results_dir=RESULTS_DIR,
    char_model_path=CHAR_MODEL_PATH,
    conf_threshold=float(os.getenv("PLATE_CONF", "0.25")),
    blur_threshold=float(os.getenv("PLATE_BLUR_THRESHOLD", "80")),
)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/recognize")
def recognize():
    # Kiểm tra file gửi lên
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Không tìm thấy file ảnh trong yêu cầu"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"ok": False, "error": "Bạn chưa chọn file ảnh"}), 400

    # Đọc dữ liệu ảnh từ buffer
    try:
        data = file.read()
        npbuf = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"ok": False, "error": "Định dạng ảnh không hợp lệ"}), 400

        # Gọi Service xử lý (Đây là nơi ảnh được lưu và URL được tạo ra)
        result = service.recognize(image)
        
        # Trả về kết quả kèm trạng thái ok
        return jsonify({"ok": True, **result})

    except Exception as exc:
        # Trả về lỗi chi tiết nếu có sự cố trong pipeline xử lý
        return jsonify({"ok": False, "error": f"Lỗi hệ thống: {str(exc)}"}), 500

if __name__ == "__main__":
    # Chạy Flask Server
    app.run(host="127.0.0.1", port=5000, debug=False)