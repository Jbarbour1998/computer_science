WildCapture 🦔

Wildlife Video Capture Using Raspberry Pi and OpenCV
BSc Computer Science Final Year Project — Jordan Barbour, Queen's University Belfast (2025)


Overview

WildCapture is an automated wildlife monitoring system built on the Raspberry Pi 4 that uses computer vision and edge AI to detect and classify Red Squirrels, Grey Squirrels, and Pine Martens in real time. The system combines a PIR motion sensor, a camera module, and the YOLOv8n object detection model to provide continuous, non-invasive wildlife monitoring with minimal human intervention.


Key Features


Real-time species detection using a custom-trained YOLOv8n model (mAP@0.5: 0.89)
Motion-triggered capture via PIR sensor to conserve processing power
Live web dashboard (Flask + Bootstrap) with video streaming, detection analytics and timelapse
Email notifications via Gmail API with attached detection images
Google Cloud Storage integration for remote media access
Species filtering — users can select which animals trigger notifications
Multithreaded architecture to maintain responsive performance on constrained hardware



System Architecture

Wildlife Movement
       ↓
PIR Motion Sensor
       ↓
Raspberry Pi Camera
       ↓
YOLOv8n Detector
    ↙        ↘
Detection Manager    Flask Web UI
    ↙        ↘
Email Notification   CSV Logger / Local Storage
(SMTP)


Tech Stack

ComponentTechnologyHardwareRaspberry Pi 4 (4GB RAM), Camera Module v2, PIR SensorDetection ModelYOLOv8n (Ultralytics)Computer VisionOpenCVBackendPython 3, FlaskFrontendHTML, Bootstrap CSS, JavaScript, Chart.jsCloud StorageGoogle Cloud StorageEmail AlertsGmail API (OAuth2)Data LoggingCSV (Pandas, Matplotlib)ConcurrencyPython Threading


Model Performance

The custom YOLOv8n model was trained on 823 images across 3 classes using Google Colab Pro (A100 GPU).

MetricScoremAP@0.50.89Precision0.87Recall0.85F1 Score0.86

Detection speed on Raspberry Pi 4: 7.7 – 8.4 FPS


Installation

Requirements

bashpip install -r requirements.txt

Key dependencies:

opencv-python==4.6.0.66
ultralytics==8.0.20
flask==2.2.3
gpiozero==1.6.2
picamera2==0.3.9
google-cloud-storage==2.7.0
pandas==1.5.3

Setup


Clone the repository:


bashgit clone https://github.com/Jbarbour1998/computer_science.git
cd computer_science


Add your Gmail API credentials (credentials.json) to the project root.
Add your Google Cloud Storage service account key.
Run the application:


bashpython main_motion.py


Access the web interface at http://<raspberry-pi-ip>:5000



Web Interface

The dashboard provides:


Live video feed with detection bounding boxes and confidence scores
Species filter — select which animals to monitor
Animal Detection Analytics — time series chart of detections by species
Images tab — browse and download captured detection images
Videos tab — review and download recorded wildlife footage
Timelapse tab — configure and generate timelapse videos
Email Settings — configure notification preferences



Testing

35 test cases were executed covering all major components:

ResultCountPassed31Failed2Error2Skipped1

Tests cover camera initialisation, YOLOv8 inference, Flask API routes, email notifications, Google Cloud Storage upload, and CSV logging.

bashpython -m unittest discover test_cases


Known Limitations


Detection accuracy reduces in very low light (night vision not yet implemented)
Requires internet connectivity for cloud upload and email alerts
Inference speed limited to ~8 FPS on Raspberry Pi 4 CPU



Future Work


Night vision / infrared camera support
On-device model optimisation (TensorRT / OpenVINO)
Solar power and battery management for remote deployment
Extended species detection (foxes, hedgehogs, owls etc.)
Mobile app companion for push notifications
Offline-first mode with local sync



Project Structure

wildCapture/
- main_motion.py          # Application entry point
- app_controller.py       # WildlifeController — coordinates all components
- animal_detector.py      # AnimalDetector — YOLOv8 inference
- detection_worker.py     # Background detection thread
- integrated_camera.py    # Camera + PIR motion sensor integration
- email_notifier.py       # Gmail API notification system
- storage_manager.py      # Google Cloud Storage integration
- timelapse_manager.py    # Timelapse capture and generation
- detection_logger.py     # CSV logging and analytics
- templates/              # Flask HTML templates
- static/                 # CSS, JS assets
- test_cases/             # Unit and integration tests
- models/                 # YOLOv8 model weights


Acknowledgements


Supervisor: Giuseppe Trombino
Queen's University Belfast School of EEECS for hardware funding
Dataset annotation via Roboflow
YOLOv8 by Ultralytics



© 2025 WildCapture. Jordan Barbour, Queen's University Belfast.
