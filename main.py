import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI,UploadFile,File
from model import preprocess_image_bytes, get_image_embedding, index, image_paths
import tempfile




app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, this is the image search API!"}


@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_tensor = preprocess_image_bytes(image_bytes)
    query_emb = get_image_embedding(img_tensor)
    k = 5
    _, indices = index.search(query_emb.astype('float32'), k)
    results = []
    for i in range(k):
        filename_with_ext = os.path.basename(image_paths[indices[0][i]])
        filename = os.path.splitext(filename_with_ext)[0]
        result = {
            "rank": int(i+1),
            "image_name":filename,
        }
        results.append(result)
    
    return {"results": results}

@app.post("/add_image")
async def add_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_tensor = preprocess_image_bytes(image_bytes)
    emb = get_image_embedding(img_tensor)
    emb_float32 = emb.astype('float32')
    index.add(emb_float32)
    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    image_paths.append(file.filename)
    os.remove(save_path)
    return {
        "message": "Image successfully added.",
        "image_name": file.filename,
    }

@app.post("/recognize_plate")
async def recognize_plate(file: UploadFile = File(...)):
    try:
        # حفظ الصورة مؤقتاً
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        # الخطوات
        predictions = detect_license_plate(tmp_path)
        plate_img = extract_plate_from_image(tmp_path, predictions)
        if plate_img is None:
            os.remove(tmp_path)
            return JSONResponse(status_code=404, content={"error": "لم يتم اكتشاف أي لوحة"})
        enhanced_img = enhance_image_for_ocr(plate_img)
        numbers = extract_license_number(enhanced_img)
        os.remove(tmp_path)
        if numbers:
            longest = max(numbers, key=len)
            return {"plate_number": longest}
        else:
            return {"plate_number": None}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})