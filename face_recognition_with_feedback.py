
import os
import cv2
import torch
import pickle
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from ultralytics import YOLO
from faceembeddingmodel import FaceEmbeddingModel
from model import load_model, MobileFaceNet
from collections import deque, Counter
import RPi.GPIO as GPIO
import time
import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from gtts import gTTS
from playsound import playsound

# ========= Email Function ========= #
def send_mail():
    sender = '22521341@gm.uit.edu.vn'
    receiver = 'phamtrungthang9.6@gmail.com'
    password = 'qvfe nguo limx jrcx'

    message = f"Unknown Person Detected at {datetime.datetime.now()}"
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = 'Alert: Unknown Face Detected at Door'

    with open('Unknown.jpg', 'rb') as f:
        img_data = f.read()
    msg.attach(MIMEImage(img_data, name='Unknown.jpg'))
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        feedback("Email alert sent.")
    except Exception as e:
        feedback("Failed to send email.")

# ========= GPIO Setup ========= #
RELAY = 17
LEDR = 27
LEDW = 22
BUZZER = 23
PIR = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.setup(LEDR, GPIO.OUT)
GPIO.setup(LEDW, GPIO.OUT)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(PIR, GPIO.IN)

GPIO.output(RELAY, GPIO.LOW)
GPIO.output(LEDR, GPIO.LOW)
GPIO.output(LEDW, GPIO.LOW)
GPIO.output(BUZZER, GPIO.LOW)

# ========= Feedback Voice ========= #

def speak(text, filename="feedback.mp3"):
    try:
        tts = gTTS(text=text, lang='vi')
        tts.save(filename)
        os.system(f'mpv --no-video --speed=1.5 {filename} > /dev/null 2>&1')
    except Exception as e:
        print("Speech error:", e)
def feedback(text):
    print(text)
    speak(text)


# ========= Model Loading ========= #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_model_path = "/home/acer/Detect_Recogniton_Face/(yolo)best.pt"
assert os.path.exists(yolo_model_path), f"Missing YOLO model at {yolo_model_path}"
face_detector = YOLO(yolo_model_path)

model_path = "/home/acer/Detect_Recogniton_Face/model_only.pth"
assert os.path.exists(model_path), f"Missing Face model at {model_path}"
model = load_model(model_path, device=device, model_type='mobilefacenet').to(device)
model.eval()

with open("/home/acer/Detect_Recogniton_Face/all_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

gallery_embeddings = torch.tensor(data['embeddings']).to(device)
gallery_embeddings = F.normalize(gallery_embeddings, dim=1)
gallery_labels = data['labels']

# ========= Preprocessing ========= #
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def cosine_similarity(emb1, emb2):
    emb1 = F.normalize(emb1, dim=0)
    emb2 = F.normalize(emb2, dim=1)
    return torch.matmul(emb2, emb1)

# ========= Face Recognition ========= #
feedback("System is armed. Waiting for motion detection...")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    feedback("Camera cannot be opened.")
    exit()


buffer_size = 5
recent_labels = deque(maxlen=buffer_size)
recent_scores = deque(maxlen=buffer_size)
threshold = 0.5

GPIO.output(LEDR, GPIO.HIGH)
GPIO.output(LEDW, GPIO.HIGH)
GPIO.output(BUZZER, GPIO.LOW)

def led_on(pin):
    GPIO.output(pin, GPIO.LOW)

def led_off(pin):
    GPIO.output(pin, GPIO.HIGH)

def buzzer_on(pin):
    GPIO.output(pin, GPIO.HIGH)

def buzzer_off(pin):
    GPIO.output(pin, GPIO.LOW)

def relay_on(pin):
    GPIO.output(pin, GPIO.HIGH)

def relay_off(pin):
    GPIO.output(pin, GPIO.LOW)

try:
    while True:
        if GPIO.input(PIR):
            GPIO.output(BUZZER, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(BUZZER, GPIO.LOW)
            feedback("Motion detected. Warming up camera...")

            for _ in range(20):
                cap.read()
                time.sleep(0.1)

            start_time = time.time()
            duration = 3
            recognized = False

            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                results = face_detector(frame)

                if len(results) == 0 or all(len(r.boxes) == 0 for r in results):
                    recent_labels.append("Unknown")
                    recent_scores.append(0)
                else:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf.cpu().numpy()[0]

                            if confidence > 0.45:
                                h, w = frame.shape[:2]
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w, x2)
                                y2 = min(h, y2)
                                if x2 <= x1 or y2 <= y1:
                                    continue

                                face_img = frame[y1:y2, x1:x2]
                                try:
                                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                    face_tensor = transform(face_pil).unsqueeze(0).to(device)

                                    with torch.no_grad():
                                        embedding = model(face_tensor).squeeze(0)

                                    sims = cosine_similarity(embedding, gallery_embeddings)
                                    best_idx = torch.argmax(sims).item()
                                    best_score = sims[best_idx].item()
                                    current_label = gallery_labels[best_idx] if best_score > threshold else "Unknown"

                                    recent_labels.append(current_label)
                                    recent_scores.append(best_score)

                                    most_common_label = Counter(recent_labels).most_common(1)[0][0]
                                    avg_score = sum(recent_scores) / len(recent_scores)
                                    final_label = most_common_label if avg_score > threshold else "Unknown"

                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"{final_label} ({avg_score:.2f})", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                    if final_label != "Unknown":
                                        feedback(f"Hello {final_label}, access granted.")
                                        led_on(LEDW)
                                        relay_on(RELAY)
                                        feedback("Door opened.")
                                        time.sleep(3)
                                        led_off(LEDW)
                                        relay_off(RELAY)
                                        feedback("Door closed.")
                                        recognized = True
                                        led_off(LEDR)
                                        break
                                    else:
                                        blink_duration = 2
                                        blink_interval = 0.2
                                        end_time = time.time() + blink_duration
                                        while time.time() < end_time:
                                            led_on(LEDR)
                                            buzzer_on(BUZZER)
                                            time.sleep(blink_interval)

                                            led_off(LEDR)
                                            buzzer_off(BUZZER)
                                            time.sleep(blink_interval)
                                        feedback("Unknown person detected. Sending email alert.")
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                        cv2.imwrite("Unknown.jpg", frame)
                                        send_mail()
                                        break
                                except Exception as e:
                                    feedback("Error during face processing.")

                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            if not recognized:
                feedback("Movement detected, but access was not granted.")

        else:
            time.sleep(0.2)

except KeyboardInterrupt:
    feedback("Interrupted by user.")
finally:
    GPIO.output(LEDW, GPIO.HIGH)
    GPIO.output(LEDR, GPIO.HIGH)
    GPIO.output(BUZZER, GPIO.LOW)
    GPIO.output(RELAY, GPIO.LOW)

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
