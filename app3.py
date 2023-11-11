from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained model (update the model path)
model = load_model('crop_disease_model.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize the pixel values
    img = img.reshape((1, 224, 224, 3))
    return img

@app.route('/')
def index():
    return render_template('indexes.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to a temporary file
        temp_image_path = 'temp_image.jpg'
        file.save(temp_image_path)

        # Preprocess the image
        img = preprocess_image(temp_image_path)

        # Make a prediction
        prediction = model.predict(img)

        # Process the prediction (modify as needed)
        class_names = ['Diseased', 'Not_Diseased']  # Replace with your class labels
        predicted_class = class_names[int(round(prediction[0][0]))]

        # Delete the temporary image file
        os.remove(temp_image_path)

        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
