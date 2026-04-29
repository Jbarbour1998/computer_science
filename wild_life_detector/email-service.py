"""
Routes for image management in the wildlife detection application.
"""
import os
from flask import Blueprint, render_template, request, jsonify, send_from_directory, Response
from datetime import datetime
import zipfile
from io import BytesIO

from app.config import SHOTS_DIR
from app.main import storage_service

# Create blueprint
image_bp = Blueprint('images', __name__)

@image_bp.route('/')
def images():
    """Render the images gallery page."""
    return render_template('images.html')

@image_bp.route('/serve/<path:filename>')
def serve_image(filename):
    """Serve an image file with proper content type."""
    return send_from_directory(SHOTS_DIR, filename)

@image_bp.route('/capture', methods=['POST'])
def capture_image():
    """Manually capture an image from the video stream."""
    from app.main import camera_service
    
    try:
        # Request a capture from the camera service
        camera_service.request_capture()
        
        # Wait a moment for the capture to occur
        import time
        time.sleep(0.5)
        
        # Get the path of the last captured image
        image_path = camera_service.last_capture_path
        
        if image_path and os.path.exists(image_path):
            # Return the image filename
            return jsonify({
                "status": "success", 
                "message": "Image captured",
                "filename": os.path.basename(image_path),
                "path": image_path
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Capture requested but image not found"
            })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error capturing image: {str(e)}"
        })

@image_bp.route('/delete/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Delete an image from local storage and/or cloud."""
    try:
        deleted = False
        
        # Check if this is a local image
        local_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = os.path.join(SHOTS_DIR, f"{image_id}{ext}")
            if os.path.exists(test_path):
                local_path = test_path
                break
        
        if local_path:
            # Delete the file
            os.remove(local_path)
            print(f"Deleted local image: {local_path}")
            deleted = True
            
            # Also delete thumbnail if it exists
            thumb_path = os.path.join(SHOTS_DIR, f"{image_id}_thumb.jpg")
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
                print(f"Deleted thumbnail: {thumb_path}")
            
            # Delete metadata if it exists
            metadata_path = os.path.join(SHOTS_DIR, f"{image_id}_metadata.txt")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                print(f"Deleted metadata: {metadata_path}")
        
        # Check if it's a cloud image
        if storage_service.gcs_available:
            try:
                # Try different extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    cloud_filename = f"{image_id}{ext}"
                    
                    # Try to delete from cloud storage
                    if storage_service.delete_from_gcs(cloud_filename):
                        deleted = True
                        
                        # Also delete thumbnail if it exists
                        storage_service.delete_from_gcs(f"{image_id}_thumb.jpg")
                        break
            except Exception as e:
                print(f"Error deleting cloud image: {e}")
        
        if deleted:
            return jsonify({"status": "success", "message": "Image deleted successfully"})
        else:
            return jsonify({"status": "error", "message": "Image not found"}), 404
            
    except Exception as e:
        print(f"Error deleting image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@image_bp.route('/filter', methods=['POST'])
def filter_images():
    """Filter images by animal type and date range."""
    try:
        data = request.get_json()
        animal_type = data.get('animal_type', 'all')
        date_range = data.get('date_range', None)
        
        # Get all images from API
        from app.main import api_bp
        all_images_response = api_bp.get_images()
        all_images = all_images_response.json
        
        # Apply filters
        filtered_images = []
        for image in all_images:
            # Apply animal type filter
            if animal_type != 'all' and image.get('animal', '').lower() != animal_type.lower():
                continue
            
            # Apply date range filter if provided
            if date_range and 'start' in date_range and 'end' in date_range:
                try:
                    image_date = datetime.fromisoformat(image.get('timestamp', ''))
                    start_date = datetime.fromisoformat(date_range['start'])
                    end_date = datetime.fromisoformat(date_range['end'])
                    
                    if image_date < start_date or image_date > end_date:
                        continue
                except:
                    # Skip date filtering if dates are invalid
                    pass
            
            filtered_images.append(image)
        
        return jsonify(filtered_images)
    
    except Exception as e:
        print(f"Error filtering images: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@image_bp.route('/batch_download', methods=['POST'])
def batch_download():
    """Create a ZIP file with selected images for download."""
    try:
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        
        if not image_ids:
            return jsonify({"status": "error", "message": "No images selected"}), 400
        
        # Create a ZIP file in memory
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for image_id in image_ids:
                # Try local file first
                local_found = False
                for ext in ['.png', '.jpg', '.jpeg']:
                    filepath = os.path.join(SHOTS_DIR, f"{image_id}{ext}")
                    if os.path.exists(filepath):
                        zf.write(filepath, arcname=os.path.basename(filepath))
                        local_found = True
                        break
                
                # If not found locally and cloud storage is available, try to download from cloud
                if not local_found and storage_service.gcs_available:
                    try:
                        # Create temporary directory if it doesn't exist
                        from app.config import TEMP_DIR
                        os.makedirs(TEMP_DIR, exist_ok=True)
                        
                        # Try to find the image in cloud storage
                        cloud_files = storage_service.list_gcs_files()
                        
                        for cloud_file in cloud_files:
                            file_base = os.path.splitext(cloud_file['name'])[0]
                            if file_base == image_id:
                                # Download to temporary file
                                from urllib.request import urlretrieve
                                temp_file = os.path.join(TEMP_DIR, cloud_file['name'])
                                urlretrieve(cloud_file['url'], temp_file)
                                
                                # Add to ZIP
                                zf.write(temp_file, arcname=os.path.basename(temp_file))
                                
                                # Clean up
                                os.remove(temp_file)
                                break
                    except Exception as e:
                        print(f"Error downloading cloud image {image_id}: {e}")
        
        # Prepare the ZIP file for download
        memory_file.seek(0)
        
        response = Response(
            memory_file.getvalue(),
            mimetype='application/zip',
            headers={
                'Content-Disposition': 'attachment;filename=wildlife_images.zip'
            }
        )
        
        return response
        
    except Exception as e:
        print(f"Error creating batch download: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@image_bp.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image to the system."""
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image file provided"}), 400
            
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({"status": "error", "message": "No image selected"}), 400
            
        if image_file:
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uploaded_{timestamp}_{image_file.filename}"
            
            # Save locally first
            filepath = os.path.join(SHOTS_DIR, filename)
            image_file.save(filepath)
            
            # Create thumbnail
            thumb_path = storage_service.create_thumbnail(filepath)
            
            # Extract animal type from form data
            animal_type = request.form.get('animal_type', 'Unknown')
            
            # Save metadata
            metadata_file = os.path.join(SHOTS_DIR, f"{os.path.splitext(filename)[0]}_metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Animal: {animal_type}\n")
                f.write(f"Source: Uploaded\n")
            
            # Upload to cloud if available
            cloud_url = None
            if storage_service.gcs_available:
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'animal': animal_type,
                    'source': 'upload'
                }
                cloud_url = storage_service.upload_to_gcs(filepath, filename, metadata)
                
                # Upload thumbnail too if created
                if thumb_path:
                    thumb_filename = os.path.basename(thumb_path)
                    storage_service.upload_to_gcs(thumb_path, thumb_filename)
            
            return jsonify({
                "status": "success",
                "message": "Image uploaded successfully",
                "filename": filename,
                "path": filepath,
                "cloud_url": cloud_url
            })
    
    except Exception as e:
        print(f"Error uploading image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
