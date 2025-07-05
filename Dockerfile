# Stage 1: Build stage
FROM python:3.11-slim-bookworm as builder

# تحديث النظام وتثبيت حزم البناء الأساسية
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد التثبيت
RUN mkdir /install
WORKDIR /install

# نسخ ملف المتطلبات وتثبيت الاعتمادات
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim-bookworm

# تثبيت حزم النظام الضرورية للتشغيل
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# نسخ الاعتمادات المثبتة من مرحلة البناء
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# نسخ ملفات المشروع
WORKDIR /app
COPY . .

# تعيين متغيرات البيئة
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
ENV PYTHONUNBUFFERED=1

# فتح المنفذ
EXPOSE 80

# تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers","1"]