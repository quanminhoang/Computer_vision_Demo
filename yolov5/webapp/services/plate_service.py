from __future__ import annotations

import os
import re
import uuid
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import torch

try:
    PaddleOCR = getattr(importlib.import_module("paddleocr"), "PaddleOCR", None)
except Exception:  # noqa: BLE001
    PaddleOCR = None

try:
    RapidOCR = getattr(importlib.import_module("rapidocr_onnxruntime"), "RapidOCR", None)
except Exception:  # noqa: BLE001
    RapidOCR = None


class PlateRecognitionService:
    def __init__(
        self,
        repo_dir: str | Path, 
        model_path: str | Path,
        results_dir: str | Path,
        char_model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        blur_threshold: float = 80.0,
    ) -> None:
        self.repo_dir = str(Path(repo_dir).resolve())
        self.model_path = str(model_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = conf_threshold
        self.blur_threshold = blur_threshold
        self.char_model_path = str(char_model_path) if char_model_path else ""

        # Load YOLOv5 model
        self.model = torch.hub.load(self.repo_dir, "custom", path=self.model_path, source="local")
        self.char_model = None
        self.char_model_names: dict[int, str] = {}
        self.char_model_loaded = False
        if self.char_model_path and Path(self.char_model_path).exists():
            self.char_model = torch.hub.load(self.repo_dir, "custom", path=self.char_model_path, source="local")
            names = getattr(self.char_model, "names", {})
            if isinstance(names, list):
                self.char_model_names = {idx: name for idx, name in enumerate(names)}
            elif isinstance(names, dict):
                self.char_model_names = {int(k): str(v) for k, v in names.items()}
            self.char_model_loaded = True
        else:
            print(
                f"[WARN] Character model not loaded. Path not found or not set: {self.char_model_path}. "
                "Using EasyOCR fallback only."
            )

        # Khởi tạo EasyOCR (Sử dụng tiếng Anh và số)
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.ocr_backend = "easyocr"
        self.paddle_reader = None
        self.rapid_reader = None

        preferred_backend = os.getenv("PLATE_OCR_BACKEND", "rapidocr").strip().lower()

        if preferred_backend in {"auto", "paddle"} and PaddleOCR is not None:
            try:
                self.paddle_reader = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
                self.ocr_backend = "paddleocr"
                print("[INFO] OCR backend: paddleocr")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] PaddleOCR init failed: {exc}")

        if self.ocr_backend == "easyocr" and preferred_backend in {"auto", "rapidocr"} and RapidOCR is not None:
            try:
                self.rapid_reader = RapidOCR()
                self.ocr_backend = "rapidocr"
                print("[INFO] OCR backend: rapidocr")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] RapidOCR init failed: {exc}")

        if self.ocr_backend == "easyocr":
            print("[INFO] OCR backend: easyocr")

    @staticmethod
    def blur_score(image: np.ndarray) -> float:
        """Tính toán độ mờ của ảnh bằng biến phân Laplacian."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def homomorphic_filter(image: np.ndarray, d0=30, gamma_l=0.5, gamma_h=1.5, c=1.0) -> np.ndarray:
        """Khử chói và cân bằng chiếu sáng bằng bộ lọc Homomorphic."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_log = np.log1p(np.array(gray, dtype="float32"))

        fft = np.fft.fft2(gray_log)
        fft_shift = np.fft.fftshift(fft)

        rows, cols = gray.shape
        u = np.arange(0, rows) - rows / 2
        v = np.arange(0, cols) - cols / 2
        U, V = np.meshgrid(v, u)
        D_sq = U**2 + V**2

        H = (gamma_h - gamma_l) * (1 - np.exp(-c * D_sq / (d0**2))) + gamma_l
        filtered_fft_shift = fft_shift * H

        ifft_shift = np.fft.ifftshift(filtered_fft_shift)
        ifft = np.fft.ifft2(ifft_shift)

        filtered_img = np.expm1(np.real(ifft))
        filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(filtered_img)

    @staticmethod
    def enhance_plate_for_ocr(plate: np.ndarray) -> np.ndarray:
        """Pipeline xử lý ảnh nâng cao để tối ưu cho việc đọc OCR."""
        # Giảm mức can thiệp: chỉ tăng tương phản nhẹ và sharpen nhẹ,
        # tránh biến ảnh rõ thành ảnh bệt trắng/đen.
        denoised = cv2.bilateralFilter(plate, d=5, sigmaColor=45, sigmaSpace=45)

        # Cân bằng tương phản nhẹ
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        if len(denoised.shape) == 3:
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        else:
            gray = denoised
        contrast_enhanced = clahe.apply(gray)

        # Sharpen nhẹ
        gaussian = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(contrast_enhanced, 1.25, gaussian, -0.25, 0)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(ch for ch in text.upper() if ch.isalnum() or ch == "-")

    @staticmethod
    def _best_digit_window(digits: str, length: int) -> str:
        if len(digits) <= length:
            return digits

        # OCR dư ký tự thường nằm ở cuối, ưu tiên giữ cụm đầu để tránh lệch chuỗi.
        return digits[:length]

    @staticmethod
    def _format_vn_plate(text: str) -> str:
        cleaned = "".join(ch for ch in text.upper() if ch.isalnum())

        # ❌ nếu không có 2 số đầu → reject luôn
        if not re.match(r"^\d{2}", cleaned):
            return ""

        # Tìm cấu trúc phổ biến: 2 số tỉnh + 1 chữ + 1 số series + 5 số.
        prefix_match = re.match(r"^(\d{2})", cleaned)
        if not prefix_match:
            return text
        province = prefix_match.group(1)

        rest = cleaned[2:]
        letter_idx = next((i for i, ch in enumerate(rest) if ch.isalpha()), -1)
        if letter_idx == -1:
            return text

        series = rest[letter_idx]
        after_series = rest[letter_idx + 1:]
        series_digit = next((ch for ch in after_series[:3] if ch.isdigit()), "")

        # Bỏ series digit khỏi phần đuôi nếu đã bắt được.
        if series_digit:
            cut = after_series.find(series_digit)
            tail_source = after_series[cut + 1:]
        else:
            tail_source = after_series

        tail_digits = "".join(ch for ch in tail_source if ch.isdigit())
        if not tail_digits:
            return f"{province}-{series}{series_digit}"

        # Phần đuôi thường là 5 số, nhưng vẫn chấp nhận 4-6 nếu OCR thiếu/dư.
        target_len = 5 if len(tail_digits) >= 5 else len(tail_digits)
        normalized_digits = PlateRecognitionService._best_digit_window(tail_digits, target_len)
        return f"{province}-{series}{series_digit}{normalized_digits}"

    @staticmethod
    def _plate_shape_score(text: str) -> float:
        cleaned = PlateRecognitionService._normalize_text(text)
        score = 0.0
        # Dạng phổ biến biển xe máy: 75-H135792 (2 số - 1 chữ + 1 số + 5 số)
        if re.match(r"^\d{2}-[A-Z]\d{6}$", cleaned):
            score += 0.5
        elif re.match(r"^\d{2}-[A-Z]\d{5}$", cleaned):
            score += 0.35
        elif re.match(r"^\d{2}[A-Z]\d{6,7}$", cleaned.replace("-", "")):
            score += 0.25

        alnum_len = len([ch for ch in cleaned if ch.isalnum()])
        score -= abs(alnum_len - 9) * 0.05
        return score
    
    @staticmethod
    def _classify_plate_type(plate: np.ndarray) -> str:
        h, w = plate.shape[:2]
        if h == 0 or w == 0:
            return "unknown"

        ratio = w / h

        # biển vuông → xe máy
        if ratio < 1.8:
            return "xe máy"

        # biển dài → ô tô
        if ratio > 2.5:
            return "xe hơi"

        return "unknown"

    @staticmethod
    def _is_valid_plate_char(ch: str) -> bool:
        return len(ch) == 1 and (ch.isdigit() or ("A" <= ch <= "Z"))

    @staticmethod
    def _merge_rows(symbols: list[dict[str, float | str]]) -> str:
        if not symbols:
            return ""

        ys = [float(s["cy"]) for s in symbols]
        hs = [float(s["h"]) for s in symbols]
        y_spread = max(ys) - min(ys)
        h_med = float(np.median(hs)) if hs else 0.0

        # Nếu độ lệch theo trục Y lớn, chia 2 hàng ký tự.
        if h_med > 0 and y_spread > 0.55 * h_med:
            y_mid = float(np.median(ys))
            top = [s for s in symbols if float(s["cy"]) <= y_mid]
            bottom = [s for s in symbols if float(s["cy"]) > y_mid]
            top.sort(key=lambda s: float(s["x1"]))
            bottom.sort(key=lambda s: float(s["x1"]))
            return "".join(str(s["ch"]) for s in top + bottom)

        one_row = sorted(symbols, key=lambda s: float(s["x1"]))
        return "".join(str(s["ch"]) for s in one_row)

    def _char_model_plate(self, plate: np.ndarray) -> tuple[str, float]:
        if self.char_model is None:
            return "", 0.0

        try:
            pred = self.char_model(plate, size=320)
        except Exception:
            return "", 0.0

        detections = pred.xyxy[0].cpu().numpy() if hasattr(pred, "xyxy") else np.empty((0, 6))
        if detections.size == 0:
            return "", 0.0

        symbols: list[dict[str, float | str]] = []
        for det in detections:
            conf = float(det[4])
            if conf < 0.25:
                continue

            cls_idx = int(det[5])
            ch = str(self.char_model_names.get(cls_idx, "")).upper()
            if not self._is_valid_plate_char(ch):
                continue

            x1, y1, x2, y2 = [float(v) for v in det[:4]]
            symbols.append(
                {
                    "ch": ch,
                    "conf": conf,
                    "x1": x1,
                    "cy": (y1 + y2) / 2.0,
                    "h": max(1.0, y2 - y1),
                }
            )

        if not symbols:
            return "", 0.0

        raw = self._merge_rows(symbols)
        text = self._format_vn_plate(raw)
        avg_conf = float(sum(float(s["conf"]) for s in symbols) / len(symbols))
        return text, avg_conf

    @staticmethod
    def _clean_letters_digits(text: str) -> str:
        return "".join(ch for ch in text.upper() if ch.isalnum())

    @staticmethod
    def _filter_allowlist(text: str, allowlist: str) -> str:
        if not allowlist:
            return text
        allowed = set(allowlist)
        return "".join(ch for ch in text if ch in allowed)

    def _readtext_with_backend(self, image: np.ndarray, allowlist: str) -> list[tuple[str, float]]:
        if self.ocr_backend == "paddleocr" and self.paddle_reader is not None:
            try:
                raw = self.paddle_reader.ocr(image, cls=False)
                lines = raw[0] if raw and len(raw) > 0 else []
                parsed: list[tuple[str, float]] = []
                for item in lines:
                    if not item or len(item) < 2:
                        continue
                    text = str(item[1][0]) if item[1] else ""
                    conf = float(item[1][1]) if item[1] and len(item[1]) > 1 else 0.0
                    text = self._filter_allowlist(text.upper(), allowlist)
                    if text:
                        parsed.append((text, conf))
                return parsed
            except Exception:
                pass

        if self.ocr_backend == "rapidocr" and self.rapid_reader is not None:
            try:
                out, _ = self.rapid_reader(image)
                parsed = []
                for item in out or []:
                    if len(item) < 3:
                        continue
                    text = self._filter_allowlist(str(item[1]).upper(), allowlist)
                    conf = float(item[2]) if item[2] is not None else 0.0
                    if text:
                        parsed.append((text, conf))
                return parsed
            except Exception:
                pass

        # EasyOCR fallback
        results = self.reader.readtext(
            image,
            detail=1,
            allowlist=allowlist,
            decoder="beamsearch",
            beamWidth=5,
        )
        parsed = []
        for _, text, conf in results:
            cleaned = self._filter_allowlist(str(text).upper(), allowlist)
            if cleaned:
                parsed.append((cleaned, float(conf)))
        return parsed

    @staticmethod
    def _split_plate_lines(plate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if len(plate.shape) == 3 else plate
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        proj = np.sum(binary > 0, axis=1).astype(np.float32)
        h = gray.shape[0]
        start = int(h * 0.25)
        end = int(h * 0.75)
        if end <= start:
            mid = h // 2
            return plate[:mid, :], plate[mid:, :]

        valley = int(np.argmin(proj[start:end]) + start)
        valley = max(int(h * 0.35), min(int(h * 0.65), valley))

        pad = max(1, int(h * 0.02))
        top = plate[: max(1, valley - pad), :]
        bottom = plate[min(h - 1, valley + pad) :, :]
        if top.size == 0 or bottom.size == 0:
            mid = h // 2
            return plate[:mid, :], plate[mid:, :]
        return top, bottom

    def _ocr_text_line(self, image: np.ndarray, allowlist: str) -> tuple[str, float]:
        results = self._readtext_with_backend(image, allowlist)
        if not results:
            return "", 0.0

        parts: list[str] = []
        confs: list[float] = []
        for text, conf in results:
            cleaned = self._clean_letters_digits(text)
            if cleaned:
                parts.append(cleaned)
                confs.append(conf)

        if not parts:
            return "", 0.0

        joined = "".join(parts)
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return joined, avg_conf

    def _ocr_plate_two_line(self, plate: np.ndarray) -> tuple[str, float]:
        top, bottom = self._split_plate_lines(plate)

        top_text, top_conf = self._ocr_text_line(top, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
        bottom_text, bottom_conf = self._ocr_text_line(bottom, allowlist="0123456789")

        if not top_text and not bottom_text:
            return "", 0.0

        top_norm = self._clean_letters_digits(top_text)
        bottom_norm = "".join(ch for ch in bottom_text if ch.isdigit())

        province = ""
        series = ""
        if len(top_norm) >= 2:
            province = "".join(ch for ch in top_norm[:2] if ch.isdigit())
            series = "".join(ch for ch in top_norm[2:] if ch.isalnum())

        # Nếu dòng trên thiếu thông tin, fallback về OCR toàn biển để lấy prefix.
        if len(province) < 2 or len(series) < 2:
            full_text, full_conf = self._ocr_plate(plate)
            full_clean = self._clean_letters_digits(full_text)
            if len(province) < 2 and len(full_clean) >= 2:
                province = "".join(ch for ch in full_clean[:2] if ch.isdigit())
            if len(series) < 2 and len(full_clean) >= 4:
                series = "".join(ch for ch in full_clean[2:4] if ch.isalnum())
            top_conf = max(top_conf, full_conf)

        tail = self._best_digit_window(bottom_norm, 5) if len(bottom_norm) >= 5 else bottom_norm

        if len(province) >= 2 and len(series) >= 2 and len(tail) >= 4:
            text = f"{province[:2]}-{series[:2]}{tail}"
            avg_conf = (top_conf + bottom_conf) / 2.0
            return text, avg_conf

        # Không đủ dữ kiện để ghép 2 dòng ổn định.
        return "", 0.0

    def _ocr_plate(self, plate: np.ndarray) -> tuple[str, float]:
        results = self._readtext_with_backend(plate, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
        if not results:
            return "", 0.0

        parts: list[str] = []
        confidences: list[float] = []
        for text, conf in results:
            cleaned = self._normalize_text(text)
            if cleaned:
                parts.append(cleaned)
                confidences.append(conf)

        if not parts:
            return "", 0.0

        joined = "".join(parts)
        joined = self._format_vn_plate(joined)
        average_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return joined, average_conf

    @staticmethod
    def _ocr_score(text: str, confidence: float) -> float:
        if not text:
            return 0.0

        score = confidence

        cleaned = PlateRecognitionService._normalize_text(text)

        # ✅ thưởng nếu đúng format biển VN
        if re.match(r"^\d{2}-[A-Z]\d{5,6}$", cleaned):
            score += 0.6

        # ✅ thưởng nếu có đủ 2 số đầu (rất quan trọng)
        if re.match(r"^\d{2}", cleaned):
            score += 0.25
        else:
            score -= 0.35   # ❌ phạt nặng nếu mất 2 số đầu

        # ✅ thưởng có chữ + số
        if any(ch.isdigit() for ch in cleaned):
            score += 0.05
        if any(ch.isalpha() for ch in cleaned):
            score += 0.05

        # ❌ phạt nếu quá ngắn
        if len(cleaned) < 7:
            score -= 0.2

        return score

    @staticmethod
    def _to_int_box(det: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        return x1, y1, x2, y2

    def _new_run_dir(self) -> Path:
        """Tạo thư mục mới cho mỗi lần nhấn 'Nhận diện'."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "crops").mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _url_from_path(path: Path) -> str:
        """Chuyển Path hệ thống thành URL web (đường dẫn tương đối từ static)."""
        try:
            parts = path.parts
            if "static" in parts:
                idx = parts.index("static")
                return "/" + "/".join(parts[idx:])
            return str(path)
        except Exception:
            return str(path)

    def recognize(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("Invalid image data")

        run_dir = self._new_run_dir()
        original_path = run_dir / "original.jpg"
        processed_path = run_dir / "processed.jpg"

        cv2.imwrite(str(original_path), image)

        # Chạy model YOLOv5
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()

        drawn = image.copy()
        h, w = image.shape[:2]
        crops: list[dict[str, Any]] = []

        for idx, det in enumerate(detections, start=1):
            conf = float(det[4])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = self._to_int_box(det, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            plate = image[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            plate_type = self._classify_plate_type(plate)

            blur = self.blur_score(plate)
            original_text, original_conf = self._ocr_plate(plate)

            enhanced_plate = self.enhance_plate_for_ocr(plate)
            enhanced_text, enhanced_conf = self._ocr_plate(enhanced_plate)

            upscaled_plate = cv2.resize(plate, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            upscaled_text, upscaled_conf = self._ocr_plate(upscaled_plate)

            if plate_type == "xe máy":
                two_line_text, two_line_conf = self._ocr_plate_two_line(plate)
            else:
                two_line_text, two_line_conf = "", 0.0

            if two_line_text:
                bonus = 0.25 if plate_type == "xe máy" else 0.05
                two_line_score = self._ocr_score(two_line_text, two_line_conf) + bonus

            char_model_text, char_model_conf = self._char_model_plate(plate)

            candidates: list[tuple[str, str, float, float]] = [
                ("original", original_text, original_conf, self._ocr_score(original_text, original_conf) + 0.03),
                ("enhanced", enhanced_text, enhanced_conf, self._ocr_score(enhanced_text, enhanced_conf)),
                ("upscaled", upscaled_text, upscaled_conf, self._ocr_score(upscaled_text, upscaled_conf) + 0.02),
            ]

            # ❌ loại candidate thiếu 2 số đầu (nếu có cái tốt hơn)
            valid_candidates = [
                c for c in candidates
                if re.match(r"^\d{2}", PlateRecognitionService._normalize_text(c[1]))
            ]

            if valid_candidates:
                candidates = valid_candidates

            if two_line_text:
                bonus = 0.25 if plate_type == "xe máy" else 0.05
                two_line_score = self._ocr_score(two_line_text, two_line_conf) + bonus
                candidates.append(("two_line", two_line_text, two_line_conf, two_line_score))

            if char_model_text:
                # Ưu tiên nhẹ cho model chuyên ký tự để giảm lộn chữ/số.
                char_score = self._ocr_score(char_model_text, char_model_conf) + 0.12
                candidates.append(("char_model", char_model_text, char_model_conf, char_score))

            # Nếu enhanced có độ tin cậy OCR thấp, tránh để nó lấn át ảnh gốc.
            if enhanced_conf < 0.55:
                candidates = [
                    (src, txt, conf, score - 0.08 if src == "enhanced" else score)
                    for src, txt, conf, score in candidates
                ]

            # 👉 Ưu tiên theo loại biển
            if plate_type == "xe máy":
                candidates = [
                    (src, txt, conf, score + (0.15 if src == "two_line" else 0))
                    for src, txt, conf, score in candidates
                ]
            else:  # car
                candidates = [
                    (src, txt, conf, score + (0.1 if src == "original" else 0))
                    for src, txt, conf, score in candidates
                ]

            source, plate_text, plate_conf, _ = max(candidates, key=lambda x: x[3])

            candidates_list = [
                {
                    "type": src, 
                 "text": txt, 
                 "confidence": round(conf, 4), 
                 "score": round(score, 4)
                 }
                for (src, txt, conf, score) in candidates
            ]

            if not plate_text:
                source = "enhanced"
                plate_text = enhanced_text
                plate_conf = enhanced_conf

            plate_text = plate_text or "(Không đọc được)"

            # Lưu ảnh Crop Gốc
            crop_filename = f"plate_{idx:02d}.jpg"
            crop_path = run_dir / "crops" / crop_filename
            cv2.imwrite(str(crop_path), plate)
            
            # Lưu ảnh Đã Xử Lý (Enhanced)
            enhanced_filename = f"plate_{idx:02d}_enhanced.jpg"
            enhanced_path = run_dir / "crops" / enhanced_filename
            cv2.imwrite(str(enhanced_path), enhanced_plate)

            # Vẽ bounding box và text lên ảnh kết quả
            cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{plate_text} ({conf:.2f})"
            cv2.putText(drawn, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            crops.append({
                "text": plate_text,
                "confidence": round(conf, 4),
                "blur_score": round(blur, 2),
                "enhanced": source == "enhanced",
                "ocr_confidence": round(plate_conf, 4),
                "ocr_source": source,
                "candidates": candidates_list,
                "plate_type": plate_type,
                "crop_url": self._url_from_path(crop_path),
                "enhanced_url": self._url_from_path(enhanced_path),
                "bbox": [x1, y1, x2, y2],
            })

        cv2.imwrite(str(processed_path), drawn)

        return {
            "original_url": self._url_from_path(original_path),
            "processed_url": self._url_from_path(processed_path),
            "count": len(crops),
            "crops": crops,
            "char_model_loaded": self.char_model_loaded,
            "ocr_backend": self.ocr_backend,
            "ok": True
        }