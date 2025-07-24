import os
import io
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import faiss
from google.cloud import storage

# إعداد المتغيرات
GCS_BUCKET_NAME = "image-search-bucket"
GCS_INDEX_FILE = "car_index.index"
GCS_PATHS_FILE = "image_paths.npy"

# تحميل بيانات الاعتماد من متغير بيئة
GCS_CREDENTIALS = os.getenv("GCS_CREDENTIALS")
if not GCS_CREDENTIALS:
    raise Exception("GCS_CREDENTIALS env var not found")

credentials_dict = json.loads(GCS_CREDENTIALS)
storage_client = storage.Client.from_service_account_info(credentials_dict)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# تحميل النموذج من TensorFlow Hub
model_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
feature_extractor = hub.KerasLayer(model_url, trainable=False)

# ---------- دوال GCS ----------

def download_blob_as_bytes(blob_name):
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def upload_bytes_to_blob(blob_name, data):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)

# ---------- تحميل البيانات ----------

# تحميل الفهرس من GCS
def load_faiss_index():
    index_bytes = download_blob_as_bytes(GCS_INDEX_FILE)
    return faiss.read_index(io.BytesIO(index_bytes))

# تحميل قائمة المسارات من GCS
def load_image_paths():
    paths_bytes = download_blob_as_bytes(GCS_PATHS_FILE)
    return np.load(io.BytesIO(paths_bytes), allow_pickle=True).tolist()

# ---------- دوال المعالجة ----------

def preprocess_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_image_embedding(img_tensor):
    emb = feature_extractor(img_tensor)
    emb = tf.nn.l2_normalize(emb, axis=1)
    return emb.numpy()

# ---------- حفظ محدث ----------

def save_faiss_index(index):
    buffer = io.BytesIO()
    faiss.write_index(index, buffer)
    upload_bytes_to_blob(GCS_INDEX_FILE, buffer.getvalue())

def save_image_paths(paths):
    buffer = io.BytesIO()
    np.save(buffer, np.array(paths, dtype=object))
    upload_bytes_to_blob(GCS_PATHS_FILE, buffer.getvalue())
