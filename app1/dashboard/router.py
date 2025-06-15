from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from app1 import db
from app1.models import AttendanceRecord
from datetime import datetime
import os
import base64
import numpy as np
import cv2
import torch
import pickle
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from torch.nn import functional as F
from collections import deque, Counter
from .model import load_model
from app1.models import Student


dashboard_bp = Blueprint('dashboard', __name__)

# Constants
FACE_DATA_ROOT = "/home/acer/Detect_Recogniton_Face/face_data"
YOLO_MODEL_PATH = "/home/acer/Detect_Recogniton_Face/(yolo)best.pt"
MODEL_PATH = "/home/acer/Detect_Recogniton_Face/model_only.pth"
EMBEDDING_PATH = "/home/acer/Detect_Recogniton_Face/all_embeddings.pkl"
CACHE_PATH = "/home/acer/Detect_Recogniton_Face/face_cache.json"
THRESHOLD = 0.5

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = YOLO(YOLO_MODEL_PATH)
model = load_model(MODEL_PATH, device=device, model_type='mobilefacenet').to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= DASHBOARD =================
@dashboard_bp.route('/dashboard')
@login_required
def index():
    return render_template("dashboard.html")


@dashboard_bp.route('/api/students')
@login_required
def get_students():
    records = AttendanceRecord.query.order_by(AttendanceRecord.timestamp.desc()).all()
    data = [
        {
            "id": r.student_id,
            "name": r.student.ho_ten,
            "faculty": r.student.khoa,
            "class": r.student.lop,
            "avatar_url": r.student.avatar_url or "",
            "time": r.timestamp.strftime('%H:%M:%S'),
            "status": "Present"
        }
        for r in records
    ]
    return jsonify(data)


@dashboard_bp.route('/api/students', methods=['POST'])
@login_required
def add_student_record():
    try:
        data = request.get_json()

        student = Student.query.filter_by(mssv=data.get('mssv')).first()
        if not student:
            return jsonify({"error": "Student not found"}), 404

        new_record = AttendanceRecord(
            student_id=student.id,
            timestamp=datetime.strptime(data.get('timestamp'), '%Y-%m-%d %H:%M:%S')
        )

        db.session.add(new_record)
        db.session.commit()
        return jsonify({'message': 'Record saved successfully'}), 201
    except Exception as e:
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@dashboard_bp.route('/upload_face', methods=['POST'])
@login_required
def upload_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Invalid request'}), 400

        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        username = current_user.full_name if hasattr(current_user, 'full_name') else current_user.username
        save_dir = os.path.join(FACE_DATA_ROOT, username)
        os.makedirs(save_dir, exist_ok=True)

        filename = datetime.now().strftime('%Y%m%d_%H%M%S.jpg')
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, image)

        return jsonify({"message": f"Image saved successfully: {filename}"})
    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500



@dashboard_bp.route('/update_face', methods=['POST'])
@login_required
def update_face_embeddings():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            face_cache = json.load(f)
    else:
        face_cache = {}

    all_embeddings = []
    all_labels = []
    all_image_paths = []

    def detect_and_crop_face(img_path):
        results = face_detector(img_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None
        x1, y1, x2, y2 = map(int, boxes[0])
        image = Image.open(img_path).convert("RGB")
        face_crop = image.crop((x1, y1, x2, y2)).resize((112, 112))
        return np.array(face_crop)

    for person in os.listdir(FACE_DATA_ROOT):
        person_path = os.path.join(FACE_DATA_ROOT, person)
        if not os.path.isdir(person_path):
            continue

        person_cache = face_cache.get(person, {})

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            mod_time = os.path.getmtime(img_path)
            mod_time_str = datetime.fromtimestamp(mod_time).isoformat()

            if person_cache.get(img_name) == mod_time_str:
                continue

            face_np = detect_and_crop_face(img_path)
            if face_np is None:
                continue

            for i in range(3):
                if i == 0:
                    pil_img = Image.fromarray(face_np)
                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                else:
                    augmented_img = augmented(image=face_np)['image']
                    input_tensor = torch.tensor(augmented_img).permute(2, 0, 1).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = embedding_model(input_tensor)

                all_embeddings.append(embedding.cpu().numpy().flatten())
                all_labels.append(person)
                all_image_paths.append(img_path)

            person_cache[img_name] = mod_time_str
        face_cache[person] = person_cache

    # Save cache and embeddings
    with open(CACHE_PATH, "w") as f:
        json.dump(face_cache, f, indent=2)

    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump({
            "embeddings": all_embeddings,
            "labels": all_labels,
            "image_paths": all_image_paths
        }, f)

    return jsonify({"message": "Augmented embeddings updated and saved."})

    
@dashboard_bp.route('/api/attendance-summary')
@login_required
def attendance_summary():
    try:
        student_folders = [
            d for d in os.listdir(FACE_DATA_ROOT)
            if os.path.isdir(os.path.join(FACE_DATA_ROOT, d))
        ]
        total_students = len(student_folders)

        recent_records = AttendanceRecord.query.order_by(AttendanceRecord.timestamp.desc()).all()
        rec_count = len(recent_records)

        students = []
        for record in recent_records:
            folder_name = record.student.ho_ten
            folder_path = os.path.join(FACE_DATA_ROOT, folder_name)
            avatar_file = ""

            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    avatar_file = f"/face_data/{folder_name}/{images[0]}"

            students.append({
                "id": record.student_id,
                "name": record.student.ho_ten,
                "faculty": record.student.khoa,
                "class": record.student.lop,
                "time": record.timestamp.strftime("%H:%M:%S"),
                "status": "Present",
                "folder_name": folder_name,
                "avatar_url": record.student.avatar_url or ""
            })

        return jsonify({
            "total_students": total_students,
            "rec_count": rec_count,
            "students": students
        })
    except Exception as e:
        return jsonify({"error": "Failed to get attendance summary", "details": str(e)}), 500


@dashboard_bp.route('/api/recognize-face', methods=['POST'])
@login_required
def recognize_face():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image"}), 400

        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        with open(EMBEDDING_PATH, "rb") as f:
            embedding_data = pickle.load(f)

        gallery_embeddings = torch.tensor(embedding_data['embeddings']).to(device)
        gallery_embeddings = F.normalize(gallery_embeddings, dim=1)
        gallery_labels = embedding_data['labels']

        results = face_detector(frame)
        response_data = []
        buffer_size = 5
        recent_labels = deque(maxlen=buffer_size)
        recent_scores = deque(maxlen=buffer_size)

        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf.cpu().numpy()[0]

                if confidence < 0.5:
                    continue

                h, w = frame.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                face_img = frame[y1:y2, x1:x2]
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model(face_tensor).squeeze(0)
                    embedding = F.normalize(embedding, dim=0)

                sims = torch.matmul(gallery_embeddings, embedding)
                best_idx = torch.argmax(sims).item()
                best_score = sims[best_idx].item()
                label = gallery_labels[best_idx] if best_score > THRESHOLD else "Unknown"

                recent_labels.append(label)
                recent_scores.append(best_score)
                
        if recent_labels:
            most_common_label = Counter(recent_labels).most_common(1)[0][0]
            avg_score = sum(recent_scores) / len(recent_scores)
            final_label = most_common_label if avg_score > THRESHOLD else "Unknown"
        else:
            final_label = "Unknown"
            avg_score = 0.0


        if final_label != "Unknown":
            try:
                folder_name = final_label
                folder_path = os.path.join(FACE_DATA_ROOT, folder_name)
                avatar_file = ""

                if os.path.exists(folder_path):
                    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        avatar_file = f"/face_data/{folder_name}/{images[0]}"

  

                student = Student.query.filter_by(ho_ten=final_label).first()

                if student:
                    new_record = AttendanceRecord(
                        student_id=student.id,
                        timestamp=datetime.utcnow()
                    )
                    db.session.add(new_record)
                    db.session.commit()

            except Exception as db_err:
                print("? Failed to save attendance:", db_err)

        _, jpeg = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

        return jsonify({
            "faces": [{
                "label": final_label,
                "similarity": round(avg_score, 4)
            }],
            "image": f"data:image/jpeg;base64,{frame_base64}"
        })

    except Exception as e:
        return jsonify({"error": "Recognition failed", "details": str(e)}), 500
