from flask import Flask
from app.config import Config

from app.routes.main_routes import main_bp
from app.routes.image_routes import image_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(image_bp)

    return app
