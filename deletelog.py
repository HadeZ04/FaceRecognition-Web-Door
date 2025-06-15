from app1 import db, create_app  
from app1.models import AttendanceRecord

app = create_app()

with app.app_context():
    AttendanceRecord.query.delete()
    db.session.commit()
    print("? All attendance records deleted.")
