import os
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import (
    preprocess_image_bytes,
    get_image_embedding,
    index,
    image_paths,
    save_faiss_index,
    save_image_paths,
)

app = FastAPI()


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
        distances, indices = index.search(query_emb.astype("float32"), k)

        results = []
        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0 or idx >= len(image_paths):
                continue

            image_name = image_paths[idx]
            image_url = f"/images/{image_name}"  # أو رابط كامل حسب الخادم

            results.append(
                {
                    "rank": rank,
                    "image_name": image_name,  # الاسم فقط
                    "image_url": image_url,    # مسار افتراضي على الخادم
                    "distance": float(distances[0][rank - 1]),
                }
            )

        if not results:
            return JSONResponse(
                status_code=404, content={"message": "لم يتم العثور على صور مشابهة."}
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
        emb_float32 = emb.astype("float32")

        index.add(emb_float32)

        # فقط نأخذ الاسم ولا نرفع الصورة إلى GCS
        unique_filename = file.filename  # يمكنك توليد UUID هنا لتجنب التكرار
        image_paths.append(unique_filename)

        save_faiss_index(index)
        save_image_paths(image_paths)

        return JSONResponse(
            status_code=200,
            content={"message": "تمت إضافة الصورة بنجاح.", "image_name": unique_filename},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء إضافة الصورة: {str(e)}")
