import cv2
import numpy as np
import easyocr
from inference_sdk import InferenceHTTPClient

# تعريف العميل
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="4m9tZRxBfEK8C4no7zsZ"
)
reader = easyocr.Reader(['en'])
def detect_license_plate(image_path, model_id="license-plate-recognition-rxg4e/11"):
    """إجراء التنبؤ باستخدام نموذج Roboflow"""
    result = CLIENT.infer(image_path, model_id=model_id)
    return result

def extract_plate_from_image(image_path, predictions):
    """استخلاص صورة اللوحة من الصورة الكاملة بناءً على التنبؤات"""
    image = cv2.imread(image_path)
    max_confidence = 0
    best_coords = None

    for prediction in predictions['predictions']:
        confidence = prediction['confidence']
        if confidence > max_confidence:
            max_confidence = confidence
            x = int(prediction['x'])
            y = int(prediction['y'])
            width = int(prediction['width'])
            height = int(prediction['height'])
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
            best_coords = (x1, y1, x2, y2)

    if not best_coords:
        return None  # لم يتم الكشف عن أي لوحة

    x1, y1, x2, y2 = best_coords
    polygon_points = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(polygon_points)
    cropped_plate = masked_image[y:y+h, x:x+w]

    return cropped_plate

def enhance_image_for_ocr(image):
    """تحسين الصورة لاستخلاص النصوص باستخدام EasyOCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def extract_license_number(image):
    """استخراج الأرقام من صورة اللوحة"""
    result = reader.readtext(image)
    plate_numbers = []

    for detection in result:
        text = detection[1]
        numbers_only = ''.join(c for c in text if c.isdigit())
        if numbers_only:
            plate_numbers.append(numbers_only)
    return plate_numbers
