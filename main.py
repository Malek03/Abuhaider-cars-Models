import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI,UploadFile,File,HTTPException
import uuid
from fastapi.responses import JSONResponse
from model import preprocess_image_bytes, get_image_embedding, index, image_paths
import tempfile
import faiss
import numpy as np


app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, this is the image search API!"}



@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    try:
        # التحقق من نوع الملف (صورة فقط)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="يرجى رفع صورة فقط.")

        # قراءة البيانات
        image_bytes = await file.read()

        # معالجة الصورة وتحويلها إلى embedding
        img_tensor = preprocess_image_bytes(image_bytes)
        query_emb = get_image_embedding(img_tensor)

        # التأكد من أن الفهرس وقائمة المسارات غير فارغة
        if index.ntotal == 0 or not image_paths:
            raise HTTPException(status_code=404, detail="قاعدة بيانات الصور فارغة.")

        # عدد النتائج المراد استرجاعها (k)
        k = 5
        # إذا كان عدد الصور في الفهرس أقل من k نعدل k
        k = min(k, index.ntotal)

        # بحث التشابه
        distances, indices = index.search(query_emb.astype('float32'), k)

        results = []
        for rank, idx in enumerate(indices[0], start=1):
            # تأكد أن الفهرس ضمن الحدود
            if idx < 0 or idx >= len(image_paths):
                continue

            filename_with_ext = os.path.basename(image_paths[idx])
            filename = os.path.splitext(filename_with_ext)[0]

            results.append({
                "rank": rank,
                "image_name": filename,
                "distance": float(distances[0][rank-1])  # يمكن إظهار المسافة كمعيار
            })

        # إذا لم تُرجع نتائج، نبلغ المستخدم
        if not results:
            return JSONResponse(
                status_code=404,
                content={"message": "لم يتم العثور على صور مشابهة."}
            )

        return {"results": results}

    except HTTPException as he:
        # تمرير استثناءات HTTP مباشرة
        raise he

    except Exception as e:
        # أي خطأ غير متوقع
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء البحث: {str(e)}")




@app.post("/add_image")
async def add_image(file: UploadFile = File(...)):
    try:
        # تحقق من نوع الملف
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="يرجى رفع صورة فقط.")

        # قراءة الصورة من المستخدم
        image_bytes = await file.read()

        # المعالجة الأولية وتحويل الصورة إلى embedding
        img_tensor = preprocess_image_bytes(image_bytes)
        emb = get_image_embedding(img_tensor)
        emb_float32 = emb.astype('float32')

        # تحديث الفهرس
        index.add(emb_float32)

        # إنشاء اسم فريد للصورة
        unique_filename = file.filename

        # حفظ الصورة في مجلد images/
        save_dir = "images"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, unique_filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        # تحديث قائمة المسارات
        image_paths.append(unique_filename)

        # حفظ الفهرس المحدث والبيانات في ملفات دائمة
        os.makedirs("data", exist_ok=True)
        faiss.write_index(index, "./data/car_index.index")
        np.save("./data/image_paths.npy", np.array(image_paths))

        return JSONResponse(
            status_code=200,
            content={
                "message": "تمت إضافة الصورة بنجاح.",
                "image_name": unique_filename,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء إضافة الصورة: {str(e)}")
