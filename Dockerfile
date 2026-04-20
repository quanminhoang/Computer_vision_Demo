# Sử dụng Python bản nhẹ
FROM python:3.10-slim

# Cài đặt thư viện hệ thống cho OpenCV và build python-bidi
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt trước để tận dụng cache
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Render sẽ cấp Port qua biến môi trường $PORT
ENV PORT=10000
EXPOSE 10000

# Chạy app bằng gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT webapp.app:app:
