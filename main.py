import cv2
from ultralytics import YOLO

# Load your model
model = YOLO("model/best.pt")

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Convert to Gray for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Find faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 3. Crop the face area
        face_img = frame[y:y+h, x:x+w]
        
        # 4. Run YOLO ONLY on the face crop
        # We use a very small imgsz because the crop is small
        results = model.predict(source=face_img, conf=0.2, imgsz=128, verbose=False)

        # 5. Draw results on the original frame
        for r in results:
            if len(r.boxes) > 0:
                label = r.names[int(r.boxes[0].cls)]
                conf = r.boxes[0].conf[0]
                
                # Draw rectangle around face and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Smart Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()