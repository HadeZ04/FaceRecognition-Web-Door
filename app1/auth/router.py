from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from flask_login import login_user, logout_user, current_user, login_required
from app1.models import User
from app1 import db

auth_bp = Blueprint('auth', __name__)



@auth_bp.route('/')
def root_redirect():
    return redirect(url_for('auth.login_page'))



@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    if current_user.is_authenticated:
        return jsonify({"success": True, "redirect": url_for('dashboard.index')})

    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "Missing username or password"}), 400

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({"success": True, "redirect": url_for('dashboard.index')})

    return jsonify({"success": False, "message": "Invalid username or password"}), 401


from flask import request, jsonify
from app1.models import User, Student
from app1 import db
from werkzeug.security import generate_password_hash

@auth_bp.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()

    full_name = data.get("full_name")
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    confirm_password = data.get("confirm_password")
    instructor_code = data.get("instructor_code")

    # Student-specific fields
    student_id = data.get("mssv")  # Student ID
    department = data.get("khoa")
    class_name = data.get("lop")
    avatar_url = data.get("avatar_url", "")

    # Basic validation
    if not all([username, password, full_name, instructor_code]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    if password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"success": False, "message": "Username already exists"}), 400

    # Determine role based on instructor code
    if instructor_code == "SECRET123":
        role = "student"
        if not all([student_id, department, class_name]):
            return jsonify({"success": False, "message": "Missing student information"}), 400
    else:
        return jsonify({"success": False, "message": "Invalid instructor code"}), 403

    try:
        # Create new user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role
        )
        user.set_password(password)
        db.session.add(user)
        db.session.flush()  # Ensure user.id is available

        # If role is student, create a corresponding Student entry
        if role == "student":
            student = Student(
                mssv=student_id,
                ho_ten=full_name,
                khoa=department,
                lop=class_name,
                avatar_url=avatar_url
            )
            db.session.add(student)

        db.session.commit()
        return jsonify({"success": True, "message": "User registered successfully"}), 201

    except Exception as e:
        db.session.rollback()
        print(f"Registration error: {e}")
        return jsonify({"success": False, "message": "Internal server error"}), 500




@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login_page'))


# ? Render login page
@auth_bp.route('/login')
def login_page():
    return render_template("login.html")


@auth_bp.route('/register')
def register_page():
    return render_template("login.html")
