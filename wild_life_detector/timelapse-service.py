"""
Camera service for handling video capture and streaming operations.
"""
import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from threading import Thread, Lock

from app.config import VIDEOS_DIR, SHOTS_DIR


class CameraService:
    """
    Service for managing camera operations, including video streaming and recording.
    """
    def __init__(self, camera_index=0):
        """Initialize camera service with configuration."""
        self.camera_index = camera_index
        self.camera = None
        self.is_running = False
        self.recorder = None
        
        # Frame processing
        self.current_frame = None
        self.frame_lock = Lock()
        
        # Recording state
        self.is_recording = False
        self.record_frame = None
        self.record_lock = Lock()
        self.out = None
        self.current_video_path = None
        
        # Capture state
        self.capture_requested = False
        self.last_capture_path = None
        
        # Initialize camera
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize the camera with the configured settings."""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera.isOpened():
                self.is_running = True
                print(f"Camera initialized successfully on index {self.camera_index}")
                
                # Start frame capture thread
                self.capture_thread = Thread(target=self._capture_frames, daemon=True)
                self.capture_thread.start()
                
                return True
            else:
                print(f"Failed to open camera on index {self.camera_index}")
                return False
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def _capture_frames(self):
        """Thread function to continuously capture frames."""
        while self.is_running and self.camera and self.camera.isOpened():
            success, frame = self.camera.read()
            if success:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # If recording, update the recording frame
                if self.is_recording:
                    with self.record_lock:
                        self.record_frame = frame.copy()
            else:
                print("Failed to read frame from camera")
                time.sleep(0.1)  # Short pause to avoid tight loop
    
    def get_frame(self):
        """Get the current frame from the camera."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def start_camera(self):
        """Start the camera if it's not already running."""
        if not self.is_running:
            success = self.initialize_camera()
            return success
        return True
    
    def stop_camera(self):
        """Stop the camera and release resources."""
        self.is_running = False
        
        # Wait for frame capture thread to end
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Release camera
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
            print("Camera stopped and resources released")
            return True
        return False
    
    def capture_image(self, filename=None):
        """Capture a still image from the camera."""
        frame = self.get_frame()
        
        if frame is None:
            print("No frame available for capture")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shot_{timestamp}.png"
        
        # Save the image
        filepath = os.path.join(SHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        self.last_capture_path = filepath
        print(f"Image captured: {filepath}")
        return filepath
    
    def start_recording(self, filename=None):
        """Start recording video from the camera."""
        if self.is_recording:
            print("Already recording")
            return False
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
        
        # Create video writer
        filepath = os.path.join(VIDEOS_DIR, filename)
        self.current_video_path = filepath
        
        # Get frame dimensions
        frame = self.get_frame()
        if frame is None:
            print("No frame available to start recording")
            return False
        
        height, width = frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
        
        # Start recording thread
        self.is_recording = True
        self.recorder = Thread(target=self._record_video, daemon=True)
        self.recorder.start()
        
        print(f"Recording started: {filepath}")
        return True
    
    def _record_video(self):
        """Thread function to continuously write frames to video file."""
        while self.is_recording and self.out is not None:
            with self.record_lock:
                if self.record_frame is not None:
                    self.out.write(self.record_frame)
            time.sleep(0.05)  # Small delay to reduce CPU usage
    
    def stop_recording(self):
        """Stop the current recording."""
        if not self.is_recording:
            print("Not currently recording")
            return False
        
        # Set flag to stop recording thread
        self.is_recording = False
        
        # Wait for recorder thread to finish
        if self.recorder and self.recorder.is_alive():
            self.recorder.join(timeout=1.0)
        
        # Release video writer
        if self.out:
            self.out.release()
            completed_path = self.current_video_path
            self.out = None
            self.current_video_path = None
            print(f"Recording stopped: {completed_path}")
            return completed_path
        
        return False
    
    def generate_frames(self, detector=None, email_service=None):
        """
        Generate frames for streaming, with optional object detection.
        
        This is a generator function that yields processed frames for streaming.
        If a detector is provided, it will be used to detect objects in each frame.
        """
        while self.is_running:
            frame = self.get_frame()
            
            if frame is None:
                # If no frame is available, yield a blank frame or placeholder
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Add text saying camera not available
                cv2.putText(blank_frame, "Camera not available", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)  # Avoid tight loop
                continue
            
            # Process frame with detector if available
            if detector:
                try:
                    frame, _ = detector.getObjects(frame, email_service)
                except Exception as e:
                    print(f"Error in object detection: {e}")
                    # Add error text to frame
                    cv2.putText(frame, f"Detection error: {str(e)[:30]}...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if a capture was requested
            if self.capture_requested:
                self.capture_requested = False
                self.capture_image()
            
            # Add recording indicator if recording
            if self.is_recording:
                height, width = frame.shape[:2]
                cv2.putText(frame, "● REC", (width - 100, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    def request_capture(self):
        """Request a frame capture on the next frame."""
        self.capture_requested = True
        return True
