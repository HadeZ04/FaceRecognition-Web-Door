from app1 import db
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'student' or 'teacher'
    email = db.Column(db.String(120), unique=True, nullable=True)
    full_name = db.Column(db.String(100), nullable=True)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mssv = db.Column(db.String(10), unique=True, nullable=False, index=True)
    ho_ten = db.Column(db.String(100), nullable=False)
    khoa = db.Column(db.String(50), nullable=False)
    lop = db.Column(db.String(10), nullable=False)
    face_encoding = db.Column(db.Text)
    avatar_url = db.Column(db.String(255))  # optional, for UI
    attendances = db.relationship('AttendanceRecord', backref='student', lazy=True)


class AttendanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    status = db.Column(db.String(20), default='present', nullable=False)
