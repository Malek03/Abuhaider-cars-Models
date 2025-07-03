import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import faiss
from PIL import Image
import io

# تحميل الموديل
model_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
feature_extractor = hub.KerasLayer(model_url, trainable=False)

# تحميل الفهرس والـ image_paths
index = faiss.read_index("./data/car_index.index")
image_paths = np.load("./data/image_paths.npy").tolist()


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