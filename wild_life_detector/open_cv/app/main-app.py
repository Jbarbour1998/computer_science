"""
Main application file for the wildlife detection system.
Initializes services and starts the Flask web server.
"""
from flask import Flask
import threading
import os



from app.config import APP_SECRET_KEY
from app.services.camera_service import CameraService
from app.services.timelapse_service import TimelapseManager
from app.services.email_service import EmailService
from app.services.storage_service import StorageService
from app.services.logging_service import LoggingService
from app.models.animal_detector import AnimalDetector

# Initialize Flask app
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates')

app.secret_key = APP_SECRET_KEY

# Initialize services
camera_service = CameraService()
timelapse_manager = TimelapseManager()
email_service = EmailService()
storage_service = StorageService()
logging_service = LoggingService()

# Initialize animal detector with timelapse manager
animal_detector = AnimalDetector(timelapse_manager=timelapse_manager)

# Current filter for detections
selected_filter = "all"

# Import and register routes
from app.routes.main_routes import main_bp
from app.routes.api_routes import api_bp
from app.routes.image_routes import image_bp
from app.routes.video_routes import video_bp
from app.routes.settings_routes import settings_bp

app.register_blueprint(main_bp)
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(image_bp, url_prefix='/images')
app.register_blueprint(video_bp, url_prefix='/videos')
app.register_blueprint(settings_bp, url_prefix='/settings')

# Register global context processors
@app.context_processor
def inject_globals():
    """Inject global variables into all templates."""
    return {
        'current_filter': selected_filter,
        'timelapse_enabled': timelapse_manager.enabled,
        'email_notifications': email_service.receiver_emails != ''
    }

# Cleanup function to run when app shuts down
@app.teardown_appcontext
def shutdown_app(exception=None):
    """Clean up resources when the app shuts down."""
    camera_service.stop_camera()

if __name__ == "__main__":
    # Start the app
    app.run(debug=True, host='0.0.0.0', port=5000)
