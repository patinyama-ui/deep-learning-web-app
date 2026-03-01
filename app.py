# app.py
from flask import Flask, render_template, request, redirect, url_for, abort
import os
import cv2
from ultralytics import YOLO
import json
from datetime import datetime, date, timezone, timedelta
import hashlib

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

UPLOAD_FOLDER = os.path.join("static", "uploads", "original")
RESULT_FOLDER = os.path.join("static", "results")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(app.root_path, "static", "profile"), exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
model = YOLO("best.pt")

# กำหนด TH_TZ ระดับ module เพื่อให้ทุก function ใช้ร่วมกันได้
TH_TZ = timezone(timedelta(hours=7))

COIN_VALUES = {
    0: 0.25, 1: 0.50, 2: 1, 3: 10, 4: 2, 5: 5
}

CLASS_COLORS = {
    0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0),
    3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0),
}


# ---------------------------
# DB Models
# ---------------------------
class User(db.Model, UserMixin):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


def utcnow_aware():
    return datetime.now(timezone.utc)


class Job(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    original_url = db.Column(db.String(255), nullable=False)
    result_url   = db.Column(db.String(255), nullable=False)
    summary_json = db.Column(db.Text, nullable=False)
    total        = db.Column(db.Float, nullable=False)
    created_at   = db.Column(db.DateTime, default=utcnow_aware, nullable=False)

    user         = db.relationship("User", backref=db.backref("jobs", lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ตารางกลาง many-to-many
box_job = db.Table(
    "box_job",
    db.Column("box_id", db.Integer, db.ForeignKey("box.id")),
    db.Column("job_id", db.Integer, db.ForeignKey("job.id"))
)

class Box(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    name    = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    user = db.relationship("User", backref=db.backref("boxes", lazy=True))
    jobs = db.relationship("Job", secondary=box_job, backref="boxes")


with app.app_context():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", password_hash=generate_password_hash("1234"))
        db.session.add(admin)
        db.session.commit()


# ---------------------------
# Utils
# ---------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_aware(dt):
    """แปลง naive datetime เป็น UTC-aware เพื่อความปลอดภัยในการเปรียบเทียบ"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def detect_coins(image_path: str):
    results = model(image_path)
    result  = results[0]
    img     = result.orig_img
    coin_counts = {}
    total = 0.0

    for box in result.boxes:
        class_id = int(box.cls)
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        coin_counts[class_id] = coin_counts.get(class_id, 0) + 1
        total += COIN_VALUES.get(class_id, 0)

        coin_value = COIN_VALUES.get(class_id, 0)
        color      = CLASS_COLORS.get(class_id, (0, 255, 0))
        label      = f"{coin_value:g}B ({conf:.2f})"

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r  = int(min(x2 - x1, y2 - y1) / 2) + 4

        cv2.circle(img, (cx, cy), r, color, 3)
        cv2.putText(img, label, (max(0, cx - r), max(25, cy - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    summary = {}
    for class_id, count in coin_counts.items():
        name  = model.names[class_id]
        value = COIN_VALUES[class_id]
        summary[name] = {
            "count":    int(count),
            "value":    float(value),
            "subtotal": float(count * value)
        }
    summary = dict(sorted(summary.items(), key=lambda item: item[1]["value"], reverse=True))

    filename    = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    output_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(output_path, img)

    return summary, float(total), output_path


# ---------------------------
# Template Filters
# ---------------------------
TH_MONTHS = {
    1: "ม.ค.", 2: "ก.พ.", 3: "มี.ค.",  4: "เม.ย.",
    5: "พ.ค.", 6: "มิ.ย.", 7: "ก.ค.", 8: "ส.ค.",
    9: "ก.ย.", 10: "ต.ค.", 11: "พ.ย.", 12: "ธ.ค."
}

@app.template_filter('th_time')
def th_time_filter(dt):
    if dt is None:
        return "-"
    dt_th = to_aware(dt).astimezone(TH_TZ)
    day   = dt_th.strftime('%d')
    month = TH_MONTHS[dt_th.month]
    year  = dt_th.year + 543
    time  = dt_th.strftime('%H:%M')
    return f"{day} {month} {year}  {time} น."


# ---------------------------
# Auth Routes
# ---------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return "กรอก username/password ให้ครบ", 400
        if User.query.filter_by(username=username).first():
            return "username นี้ถูกใช้แล้ว", 400
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return "username หรือ password ไม่ถูกต้อง", 401
        login_user(user)
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------
# App Routes
# ---------------------------
@app.route("/home", methods=["GET", "POST"])
@login_required
def home():

    if request.method == "POST":
        box_name = request.form.get("box_name")
        if box_name:
            new_box = Box(name=box_name, user_id=current_user.id)
            db.session.add(new_box)
            db.session.commit()
        return redirect(url_for("home"))

    boxes = Box.query.filter_by(user_id=current_user.id).all()

    return render_template(
        "home.html",
        active="home",
        boxes=boxes
    )


@app.route("/history")
@login_required
def history():
    jobs = Job.query.filter_by(user_id=current_user.id)\
                    .order_by(Job.created_at.desc()).all()
    return render_template("history.html", jobs=jobs, active="history")

@app.route("/add_to_box", methods=["POST"])
@login_required
def add_to_box():
    box_id  = request.form.get("box_id")
    job_ids = request.form.getlist("job_ids")

    box = Box.query.get_or_404(box_id)
    if box.user_id != current_user.id:
        abort(403)

    for jid in job_ids:
        job = Job.query.get(int(jid))
        if job and job.user_id == current_user.id:
            # ตรวจก่อนว่า job นี้อยู่ในกล่องนี้แล้วหรือยัง
            # ถ้ายังไม่มีค่อยเพิ่ม เพื่อป้องกันยอดพองซ้ำ
            if job not in box.jobs:
                box.jobs.append(job)

    db.session.commit()
    return redirect(url_for("history"))

@app.route("/guide")
@login_required
def guide():
    return render_template("guide.html", active="guide")


@app.route("/profile")
@login_required
def profile():
    profile_images_dir_path = os.path.join(app.root_path, 'static', 'profile')
    profile_image_filenames_only = []

    if os.path.exists(profile_images_dir_path):
        for f in os.listdir(profile_images_dir_path):
            if f.lower().startswith('profile') and f.lower().endswith('.jpg'):
                try:
                    num_str = f[7:-4]
                    if num_str.isdigit():
                        profile_image_filenames_only.append((int(num_str), f))
                except ValueError:
                    continue
        profile_image_filenames_only.sort(key=lambda x: x[0])
        profile_image_filenames_only = [f[1] for f in profile_image_filenames_only]

    num_available_images = len(profile_image_filenames_only)
    ultimate_fallback_url = "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=300&q=60"
    profile_image_source  = ultimate_fallback_url

    if num_available_images > 0:
        if current_user.is_authenticated and current_user.username:
            hash_val     = hashlib.md5(current_user.username.encode('utf-8')).hexdigest()
            chosen_index = int(hash_val[:8], 16) % num_available_images
            profile_image_source = url_for('static', filename=f'profile/{profile_image_filenames_only[chosen_index]}')
        else:
            profile_image_source = url_for('static', filename=f'profile/{profile_image_filenames_only[0]}')
    else:
        default_path = os.path.join(profile_images_dir_path, 'default.jpg')
        if os.path.exists(default_path):
            profile_image_source = url_for('static', filename='profile/default.jpg')

    return render_template("profile.html", active="profile", profile_image_source=profile_image_source)


# ---------------------------
# Main Routes
# ---------------------------
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("upload.html", active="camera")

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        return "File type not allowed", 400

    mode     = request.form.get("mode", "normal")
    filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    summary, total, result_image_path = detect_coins(filepath)

    original_url    = url_for("static", filename=f"uploads/original/{filename}")
    result_filename = os.path.basename(result_image_path)
    result_url      = url_for("static", filename=f"results/{result_filename}")

    job = Job(
        user_id=current_user.id,
        original_url=original_url,
        result_url=result_url,
        summary_json=json.dumps(summary, ensure_ascii=False),
        total=float(total)
    )
    db.session.add(job)
    db.session.commit()

    template = "classroom.html" if mode == "classroom" else "result.html"
    return render_template(template,
                           original_image=original_url,
                           result_image=result_url,
                           summary=summary,
                           total=total,
                           active="camera")

@app.route("/history/<int:job_id>", methods=["GET"])
@login_required
def history_detail(job_id):
    job = Job.query.get_or_404(job_id)
    if job.user_id != current_user.id:
        abort(403)

    ids_desc = [x.id for x in
                Job.query.filter_by(user_id=current_user.id)
                         .order_by(Job.created_at.desc())
                         .with_entities(Job.id).all()]

    display_no = len(ids_desc) - ids_desc.index(job.id) if job.id in ids_desc else None
    summary    = json.loads(job.summary_json) if job.summary_json else {}

    return render_template("history_detail.html",
                           j=job,
                           summary=summary,
                           display_no=display_no,
                           active="history")

@app.route("/box/<int:box_id>")
@login_required
def box_detail(box_id):
    box = Box.query.get_or_404(box_id)
    if box.user_id != current_user.id:
        abort(403)
    return render_template("box_detail.html", box=box, active="home")


@app.route("/box/<int:box_id>/delete", methods=["POST"])
@login_required
def delete_box(box_id):
    box = Box.query.get_or_404(box_id)
    if box.user_id != current_user.id:
        abort(403)
    db.session.delete(box)
    db.session.commit()
    return redirect(url_for("home"))


@app.route("/box/<int:box_id>/remove/<int:job_id>", methods=["POST"])
@login_required
def remove_from_box(box_id, job_id):
    box = Box.query.get_or_404(box_id)
    if box.user_id != current_user.id:
        abort(403)
    job = Job.query.get_or_404(job_id)
    if job in box.jobs:
        box.jobs.remove(job)
        db.session.commit()
    return redirect(url_for("box_detail", box_id=box_id))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)