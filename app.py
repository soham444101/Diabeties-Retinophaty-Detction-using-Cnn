from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Make sure uploads are in the static folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = load_model('models/diabetic_retinopathy_cnn.keras')

# Class labels
class_labels = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))  # Adjust size if needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result_message = generate_message(predicted_class)

        # Pass filename, severity level, and confidence to the template
        return render_template('result.html', 
                               severity=class_labels[predicted_class],
                               confidence=round(confidence * 100, 2),
                               filename=filename,  # Pass just the filename here
                               message=result_message)

    return redirect(url_for('index'))

# Message generator for DR levels
def generate_message(predicted_class):
    messages = {
        0: "Your eyes show no signs of diabetic retinopathy. Continue regular check-ups.",
        1: "Mild diabetic retinopathy detected. Keep your blood sugar levels in control and consult your ophthalmologist.",
        2: "Moderate diabetic retinopathy detected. Medical evaluation is advised.",
        3: "Severe diabetic retinopathy detected. Please consult a retina specialist immediately.",
        4: "Proliferative diabetic retinopathy detected. This is the most advanced stage. Urgent treatment is needed."
    }
    return messages.get(predicted_class, "Unable to determine the retinopathy level.")

if __name__ == '__main__':
    app.run(debug=True)
