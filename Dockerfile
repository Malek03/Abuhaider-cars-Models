# استخدم صورة Python الرسمية
FROM python:3.9-slim

# لتثبيت system dependencies الأساسية المطلوبة لـ opencv و easyocr و faiss
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد للعمل
WORKDIR /app

# نسخ ملفات المشروع إلى الحاوية
COPY . .

# ترقية pip وتثبيت المتطلبات
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# تعيين المتغير لتشغيل Uvicorn على كل العناوين
ENV HOST=0.0.0.0
ENV PORT=8000

# تشغيل التطبيق باستخدام Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
