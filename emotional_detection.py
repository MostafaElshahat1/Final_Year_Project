import cv2
import torch
import timm
import numpy as np
from torchvision import transforms
from huggingface_hub import hf_hub_download

# ======================
# GLOBALS & MODEL SETUP
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    model = timm.create_model("resnet18", pretrained=False, num_classes=7)
    try:
        path = hf_hub_download(repo_id="ElenaRyumina/face_emotion_recognition", filename="pytorch_model.bin")
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        print("✅ AI Model Loaded")
    except Exception as e:
        print(f"❌ Load Error: {e}")
    return model

model = load_model()

def get_emotion(face_img):
    """Core inference function"""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        idx = torch.argmax(output, dim=1).item()
    return emotions[idx]

# ======================
# FUNCTION 1: IMAGE ONLY
# ======================
def process_static_image(path):
    frame = cv2.imread(path)
    if frame is None: return print("Image not found")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        emotion = get_emotion(frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.imshow("Static Image Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ======================
# FUNCTION 2 & 3: VIDEO & WEBCAM
# ======================
def process_stream(source_path=0, is_webcam=True):
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30
    if fps <= 0: fps = 30
    
    # Calculate: 10 frames per 60 seconds
    # If 30fps, we see 1800 frames/min. We want to analyze every 180th frame.
    analyze_every = int((fps * 60) / 10)
    
    frame_count = 0
    current_emotion = "Waiting..."

    print(f"Running stream. Analyzing 1 frame every {analyze_every} frames.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ONLY perform AI detection every 'analyze_every' frames
        if frame_count % analyze_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0] # Analyze the first face found
                current_emotion = get_emotion(frame[y:y+h, x:x+w])
            else:
                current_emotion = "No Face"

        # DRAW on every frame so the video looks smooth
        cv2.putText(frame, f"Analysis (10/min): {current_emotion}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Emotion Stream", frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

# ======================
# EXECUTION CONTROL
# ======================
# Choose one to run:
# process_static_image("test.jpg")
process_stream(0, is_webcam=True) # For Webcam
# process_stream("test.mp4", is_webcam=False) # For Video