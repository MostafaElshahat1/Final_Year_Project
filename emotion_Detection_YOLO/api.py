import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import base64

app = FastAPI()

# Load Model
try:
    model = YOLO("model/best_yolov8_model.pt")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading models: {e}")

@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    # NEW LOGIC: Run YOLO on the WHOLE image instead of using Haarcascades
    # This is much more robust for tilted faces or different angles
    results = model.predict(source=frame, conf=0.25, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = r.names[int(box.cls)]
            conf = float(box.conf[0])

            detections.append({
                "emotion": label,
                "confidence": round(conf, 2),
                "bbox": {"x": int(x1), "y": int(y1), "w": int(x2-x1), "h": int(y2-y1)}
            })

    return {
        "student_count": len(detections),
        "results": detections
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)