import os
import json
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import storage
from model import preprocess_image_bytes, get_image_embedding, index, image_paths

app = FastAPI()

# إعداد Google Cloud Storage
GCS_BUCKET_NAME = "image-search-bucket"

# تحميل بيانات الاعتماد من متغير بيئة (مناسب لـ Render)
GCS_CREDENTIALS = os.getenv("GCS_CREDENTIALS")
if not GCS_CREDENTIALS:
    raise Exception("لم يتم العثور على بيانات اعتماد Google Cloud Storage في متغير البيئة GCS_CREDENTIALS.")

credentials_dict = json.loads(GCS_CREDENTIALS)
storage_client = storage.Client.from_service_account_info(credentials_dict)
bucket = storage_client.bucket(GCS_BUCKET_NAME)


def upload_to_gcs(file_bytes: bytes, destination_blob_name: str) -> str:
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    return blob.public_url


@app.get("/")
def hello():
    return {"message": "Hello, this is the image search API!"}


@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="يرجى رفع صورة فقط.")

        image_bytes = await file.read()
        img_tensor = preprocess_image_bytes(image_bytes)
        query_emb = get_image_embedding(img_tensor)

        if index.ntotal == 0 or not image_paths:
            raise HTTPException(status_code=404, detail="قاعدة بيانات الصور فارغة.")

        k = min(5, index.ntotal)
        distances, indices = index.search(query_emb.astype('float32'), k)

        results = []
        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0 or idx >= len(image_paths):
                continue

            results.append({
                "rank": rank,
                "image_url": image_paths[idx],  # الصورة أصبحت URL
                "distance": float(distances[0][rank - 1])
            })

        if not results:
            return JSONResponse(
                status_code=404,
                content={"message": "لم يتم العثور على صور مشابهة."}
            )

        return {"results": results}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء البحث: {str(e)}")


@app.post("/add_image")
async def add_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="يرجى رفع صورة فقط.")

        image_bytes = await file.read()
        img_tensor = preprocess_image_bytes(image_bytes)
        emb = get_image_embedding(img_tensor)
        emb_float32 = emb.astype('float32')

        index.add(emb_float32)

        # إنشاء اسم فريد
        unique_filename =file.filename
        public_url = upload_to_gcs(image_bytes, unique_filename)

        image_paths.append(public_url)

        # حفظ الفهرس وقائمة الصور
        os.makedirs("data", exist_ok=True)
        faiss.write_index(index, "./data/car_index.index")
        np.save("./data/image_paths.npy", np.array(image_paths))

        return JSONResponse(
            status_code=200,
            content={
                "message": "تمت إضافة الصورة بنجاح.",
                "image_url": public_url
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء إضافة الصورة: {str(e)}")
