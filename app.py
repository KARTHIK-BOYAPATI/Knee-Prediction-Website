from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

model_path = 'C:\\Users\\boyap\\Videos\\project Knee Website\\knee.h5'
model = load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']

            if uploaded_file.filename != '':
                temp_dir = 'temp'
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                uploaded_file_path = os.path.join(temp_dir, uploaded_file.filename)
                uploaded_file.save(uploaded_file_path)

                img = Image.open(uploaded_file_path)

                if img.mode == 'L':
                    img = img.convert('RGB')

                input_shape = model.input_shape[1:3]
                img = img.resize(input_shape)

                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)

                class_labels = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
                predicted_label = class_labels[predicted_class]

                # Render the result page with the predicted label and uploaded image
                return render_template('result.html', predicted_label=predicted_label, uploaded_image=uploaded_file_path)

        except Exception as e:
            print(f"An error occurred: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
