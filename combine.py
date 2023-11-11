from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import os

app = Flask(__name__)

# Load the crop disease prediction model
model = load_model('crop_disease_model.h5')
print(model.summary())

# Load the crop recommendation model
csv_file_path = os.path.abspath(r'C:\Users\deeks\OneDrive\Desktop\jupyter test\project.csv')
crop_recommendation_model = DecisionTreeClassifier()
training_data = pd.read_csv(csv_file_path)
X = training_data[['Temperature', 'Humidity', 'Rainfall', 'Ph', 'N', 'P', 'K']]
y = training_data['label']
crop_recommendation_model.fit(X, y)
text_representation = tree.export_text(crop_recommendation_model)
print(text_representation)
# Define a function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = img.reshape((1, 224, 224, 3))
    return img

@app.route('/')
def index():
    return render_template('pro1.html')

# Use 'GET' method for the predict route to show the form
@app.route('/predict', methods=['GET'])
def show_predict_form():
    return render_template('indexes.html')

# Use 'POST' method for the predict route
# ... (other Flask code) ...

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Entering /predict route")

    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part")
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'})

        try:
            # Save the uploaded image to a temporary file
            temp_image_path = 'temp_image.jpg'
            file.save(temp_image_path)
            print("Image saved at:", temp_image_path)

            # Preprocess the image
            img = preprocess_image(temp_image_path)

            # Make a prediction
            prediction = model.predict(img)
            print("Prediction:", prediction)

            # Process the prediction (modify as needed)
            class_names = ['Diseased', 'Not_Diseased']
            predicted_class = class_names[int(round(prediction[0][0]))]

            # Delete the temporary image file
            os.remove(temp_image_path)
            print("Exiting /predict route")

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'prediction': predicted_class})
            else:
                return render_template('indexes.html', prediction=predicted_class)

        except Exception as e:
            print("Exception:", str(e))
            return jsonify({'error': 'Unexpected error'})

    return render_template('indexes.html')



from flask import jsonify

# ... (previous code)

@app.route('/recommendation', methods=['GET','POST'])
def recommendation():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Rainfall = float(request.form['Rainfall'])
        Ph = float(request.form['Ph'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        input_data = pd.DataFrame({
            'Temperature': [Temperature],
            'Humidity': [Humidity],
            'Rainfall': [Rainfall],
            'Ph': [Ph],
            'N': [N],
            'P': [P],
            'K': [K]
        })

        new_predictions = crop_recommendation_model.predict(input_data)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # If it's an AJAX request, respond with JSON
            return jsonify({'recommendation_result': new_predictions[0]})
        else:
            # If it's a regular form submission, render the template
            return render_template('recommendation.html', recommendation_result=new_predictions[0])

    return render_template('recommendation.html')


if __name__ == '__main__':
    app.run(debug=True)
