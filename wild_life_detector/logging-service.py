"""
Main routes for the wildlife detection application.
"""
from flask import Blueprint, render_template, Response, request, session, redirect, url_for
import time

from app.main import camera_service, animal_detector, selected_filter

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Render the main page."""
    global selected_filter
    
    # Check if there's a filter in the session and use it
    if 'selected_filter' in session:
        selected_filter = session['selected_filter']
        print(f"Retrieved filter from session: {selected_filter}")
    
    # Get list of animals for dropdown
    animal_list = animal_detector.get_animal_list()
    
    # Pass the current filter to the template
    return render_template('index.html', 
                          animals=animal_list, 
                          current_filter=selected_filter)

@main_bp.route('/video_feed')
def video_feed():
    """Stream video feed from the camera."""
    from app.main import email_service
    
    # Pass both the detector and email service to generate_frames
    return Response(
        camera_service.generate_frames(
            detector=animal_detector,
            email_service=email_service
        ),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@main_bp.route('/set_filter', methods=['POST'])
def set_filter():
    """Set the current animal detection filter."""
    global selected_filter
    
    if request.method == 'POST':
        try:
            # Check if it's a form submission or JSON data
            if request.form and 'animal_filter' in request.form:
                # Handle form data
                new_filter = request.form.get('animal_filter')
                selected_filter = new_filter
                session['selected_filter'] = selected_filter
                
                # Also update the detector's filter
                animal_detector.set_filter(selected_filter)
                
                print(f"Filter set from form to: {selected_filter}")
                # Return success JSON instead of redirect
                return {"status": "success", "filter": selected_filter}
            else:
                # Handle JSON data (for AJAX requests)
                data = request.get_json()
                if 'filter' in data:
                    selected_filter = data['filter']
                    session['selected_filter'] = selected_filter
                    
                    # Also update the detector's filter
                    animal_detector.set_filter(selected_filter)
                    
                    print(f"Filter set from JSON to: {selected_filter}")
                    return {"status": "success", "filter": selected_filter}
                else:
                    return {"status": "error", "message": "Filter key not found in request"}
        except Exception as e:
            print(f"Error in set_filter route: {e}")
            return {"status": "error", "message": str(e)}
    
    return {"status": "error", "message": "Invalid request method"}

@main_bp.route('/controls', methods=['POST'])
def camera_controls():
    """Handle camera control actions."""
    action = request.form.get('action')
    
    if action == 'capture':
        # Capture a still image
        camera_service.request_capture()
        return {"status": "success", "message": "Image capture requested"}
        
    elif action == 'start_stop_camera':
        # Toggle camera on/off
        if camera_service.is_running:
            camera_service.stop_camera()
            return {"status": "success", "message": "Camera stopped"}
        else:
            if camera_service.start_camera():
                return {"status": "success", "message": "Camera started"}
            else:
                return {"status": "error", "message": "Failed to start camera"}
                
    elif action == 'start_stop_recording':
        # Toggle recording
        if camera_service.is_recording:
            video_path = camera_service.stop_recording()
            if video_path:
                # Upload to cloud storage if configured
                from app.main import storage_service
                threading.Thread(
                    target=storage_service.upload_to_gcs,
                    args=(video_path,)
                ).start()
                return {"status": "success", "message": "Recording stopped", "path": video_path}
            else:
                return {"status": "error", "message": "Failed to stop recording"}
        else:
            if camera_service.start_recording():
                return {"status": "success", "message": "Recording started"}
            else:
                return {"status": "error", "message": "Failed to start recording"}
    
    return {"status": "error", "message": "Unknown action"}
