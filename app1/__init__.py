from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login_page'  # fallback khi chua login

def create_app():
    app = Flask(__name__, template_folder='../templates',static_folder='../static')

    app.config.from_object(Config)


    db.init_app(app)
    login_manager.init_app(app)

    from app1.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from app1.auth.router import auth_bp
    from app1.dashboard.router import dashboard_bp  

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)

    return app
