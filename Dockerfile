# 1. Sử dụng Python bản slim để nhẹ và ổn định
FROM python:3.10-slim

# 2. Cài đặt các thư viện hệ thống (Bắt buộc cho OpenCV và EasyOCR)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Tạo thư mục làm việc
WORKDIR /app

# 4. Copy và cài đặt requirements trước (để build nhanh hơn lần sau)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code vào container
COPY . .

# 6. Thiết lập biến môi trường để Python không bị nghẽn log
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# 7. Lệnh chạy app
# LƯU Ý: Nếu file app.py nằm trong thư mục webapp, dùng lệnh dưới:
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "webapp.app:app"]

