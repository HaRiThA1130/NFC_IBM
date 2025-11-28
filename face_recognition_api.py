from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for your frontend

# Load the trained YOLO model
MODEL_PATH = "best (15).pt"
model = YOLO(MODEL_PATH)

print(f"✅ Loaded face recognition model: {MODEL_PATH}")
print(f"📋 Model trained with classes: {model.names}")

@app.route('/verify', methods=['POST'])
def verify_face():
    """
    Endpoint to verify face from image
    Expects JSON: { "image": "base64_encoded_image", "nfc_id": "student_id" }
    Returns JSON: { "success": true/false, "detected_persons": [...] }
    """
    try:
        data = request.json
        
        # Get base64 image
        image_data = data.get('image', '')
        nfc_id = str(data.get('nfc_id', '')).strip()
        
        # Remove data URL prefix if present (data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV/YOLO
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # List of images to try (Original, Rotated 90, Rotated -90, Rotated 180)
        # Phone cameras often send rotated images
        images_to_try = [
            img_bgr,
            cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_180)
        ]
        
        best_results = []
        found_match = False
        
        print(f"\n🔍 Verifying NFC ID: '{nfc_id}'")
        
        for i, current_img in enumerate(images_to_try):
            print(f"  👉 Trying orientation {i+1}/4...")
            results = model(current_img, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = str(model.names[class_id]).strip()
                    
                    print(f"     👀 Detected: '{class_name}' ({confidence:.2f})")
                    
                    detection = {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence
                    }
                    best_results.append(detection)
                    
                    # Check for match (case insensitive)
                    if class_name.lower() == nfc_id.lower():
                        found_match = True
                        # Prioritize this result
                        best_results = [detection] + best_results
                        break
            
            if found_match:
                print("  ✅ Match found in this orientation!")
                break
        
        # Remove duplicates
        unique_results = []
        seen = set()
        for res in best_results:
            if res['class'] not in seen:
                unique_results.append(res)
                seen.add(res['class'])
        
        print(f"📊 Final Results: {unique_results}")
        
        return jsonify({
            "success": True,
            "detected_persons": unique_results,
            "nfc_id": nfc_id
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "detected_persons": []
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": MODEL_PATH,
        "classes": model.names
    })

if __name__ == '__main__':
    print("🚀 Starting Face Recognition API on http://localhost:5001")
    print("📡 Endpoint: POST http://localhost:5001/verify")
    print("🏥 Health check: GET http://localhost:5001/health")
    app.run(host='0.0.0.0', port=5001, debug=True)
