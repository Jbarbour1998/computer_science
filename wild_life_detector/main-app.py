"""
Configuration settings for the wildlife detection application.
"""
import os

# App configuration
APP_SECRET_KEY = 'wildlife_detection_secret_key'
CONFIG_DIR = 'static/config'
TIMELAPSE_CONFIG_FILE = f'{CONFIG_DIR}/timelapse_settings.json'

# Directory paths
SHOTS_DIR = 'static/shots'
VIDEOS_DIR = 'static/videos'
TIMELAPSE_DIR = 'static/timelapse'
TEMP_DIR = 'static/temp'

# Create necessary folders
os.makedirs(SHOTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(TIMELAPSE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# YOLO model configuration
YOLO_MODEL_PATH = r"C:\Users\jorda\OneDrive\Computer Science 3rd Year\Final Year Project\Open CV code\object_detection\redz_greyz_martenz\weights\best.pt"

# Email configuration
DEFAULT_SENDER_EMAIL = 'jordanbarbour65@gmail.com'
DEFAULT_RECEIVER_EMAILS = 'jordanbarbour65@gmail.com'
EMAIL_COOLDOWN = 10  # seconds between emails

# Timelapse default settings
DEFAULT_TIMELAPSE_CONFIG = {
    "enabled": False,
    "interval_minutes": 30,
    "start_time": "00:00",
    "end_time": "23:59"
}

# Google Cloud Storage configuration
GCS_BUCKET_NAME = "wildcapturestorage"

# Path to Google Cloud credentials file
CREDENTIALS_PATHS = [
    "wildcapture-key.json",  # Same directory
    r"C:\Users\jorda\OneDrive\Computer Science 3rd Year\Final Year Project\Open CV code\frontEndFireBase\wildcapture-key.json"  # Full path
]

# Find the first valid credentials path
CREDENTIALS_FILE = None
for path in CREDENTIALS_PATHS:
    if os.path.exists(path):
        CREDENTIALS_FILE = path
        break

# Set environment variable for Google credentials
if CREDENTIALS_FILE:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE
else:
    print("WARNING: Could not find Google Cloud credentials file!")

# Detection settings
DETECTION_COOLDOWN = 12.0  # seconds to wait before counting the same animal again
DISAPPEAR_TIMEOUT = 10  # seconds until we clear a "stale" detection
