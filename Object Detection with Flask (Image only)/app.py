from flask import Flask, render_template, request, jsonify
import os
import cv2
import torch
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Set up file upload folder and allowed file types
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the YOLO model (replace 'best.pt' with the path to your trained model)
model = YOLO('best.pt')  # Initialize YOLO with your trained model

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Perform prediction using the loaded YOLO model
        try:
            result_file = run_inference(model, file_path, filename)
            result_path = os.path.join('static', 'uploads', result_file)  # Relative path for video or image
            return jsonify({'result': result_path})  # Return the relative URL path
        except Exception as e:
            return jsonify({'error': f'Inference failed: {e}'})
    
    return jsonify({'error': 'Invalid file format'})

def run_inference(model, file_path, filename):
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(file_path)
        results = model(image, save=True)
        result_image = "processed_" + filename
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_image)
        result_image_data = results[0].plot()
        cv2.imwrite(result_image_path, result_image_data)
        return result_image

    elif file_path.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        result_video = "processed_" + filename
        result_video_path = os.path.join(app.config['UPLOAD_FOLDER'], result_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            processed_frame = results[0].plot()
            out.write(processed_frame)

        cap.release()
        out.release()
        return result_video

    else:
        raise ValueError("Unsupported file format")

if __name__ == '__main__':
    app.run(debug=True)
