from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import keras.utils as image
import numpy as np
import cv2
import os

from models import Model, load_depression_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained Alzheimer's prediction model
alzheimer_model = load_model('models/alzheimer_model.h5')
brain_tumour_model = load_model('models/brain_tumour_model.h5')


@app.route('/predict_alzheimer', methods=['POST'])
def predict_alzheimer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image to server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Preprocess the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (208, 176))  # Resize image to model input size
    img = img.astype(np.float32) / 255.0  # Normalize image

    # Perform prediction using the model
    prediction = alzheimer_model.predict(np.expand_dims(img, axis=0))
    print(prediction)
    predicted_class_index = np.argmax(prediction)
    class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia']
    class_info = [
        "In this stage, individuals experience noticeable cognitive decline, such as increased forgetfulness, difficulty finding words, and challenges with planning or organizing daily activities.",
        "Moderate dementia is characterized by significant cognitive impairment affecting "
        "memory, language, reasoning, and daily functioning, often requiring substantial "
        "assistance with activities of daily living.", "This category describes individuals "
                                                       "with normal cognitive function and no "
                                                       "signs of memory or cognitive decline, "
                                                       "maintaining independence in daily "
                                                       "activities and tasks.", "This stage "
                                                                                "involves "
                                                                                "subtle "
                                                                                "cognitive "
                                                                                "changes that "
                                                                                "may include "
                                                                                "mild "
                                                                                "forgetfulness and difficulty with complex tasks, often not immediately apparent to others."]

    predicted_class_name = class_names[predicted_class_index]
    predicted_class_info = class_info[predicted_class_index]
    print(f"Prediction : {predicted_class_name}")

    # Convert prediction to readable format (e.g., class label)
    # Example: Assuming model outputs probability of Alzheimer's (1) or Normal (0)
    result = {'prediction': predicted_class_name, "info": predicted_class_info}  # Thresholding at 0.5 probability

    return jsonify(result), 200


@app.route("/predict_bt", methods=["POST"])
def predict_bt():
    info = ""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image to server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    # Preprocess the image
    img = cv2.imread(filename)
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (150, 150))
    img = img.reshape((1, 150, 150, 3))
    prediction = brain_tumour_model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    if prediction == 0:
        prediction = 'Glioma Tumor'
        info = "A glioma is a tumor that forms when glial cells grow out of control. Normally, these cells support nerves and help your central nervous system work"
    elif prediction == 1:
        prediction = 'No tumor'
        info = "Patient doesn't have brain tumour"
    elif prediction == 2:
        prediction = 'Meningioma Tumor'
        info = "A meningioma is a tumor that grows from the membranes that surround the brain and spinal cord, called the meninges. A meningioma is not a brain tumor, but it may press on the nearby brain, nerves and vessels."
    else:
        prediction = 'Pituitary Tumor'
        info = "A pituitary tumor is an abnormal growth in the pituitary gland. The pituitary is a small gland in the brain. It is located behind the back of the nose. It makes hormones that affect many other glands and many functions in your body."

    result = {'prediction': prediction, "info": info}

    return jsonify(result)


@app.route('/predict_depression', methods=["POST"])
def predict_depression():
    result = ""
    info = ""
    data = request.get_json()
    patient_answers = data["patientAnswers"]

    # getting values from the form
    # answers of each questions
    q1 = int(patient_answers[0])
    q2 = int(patient_answers[1])
    q3 = int(patient_answers[2])
    q4 = int(patient_answers[3])
    q5 = int(patient_answers[4])
    q6 = int(patient_answers[5])
    q7 = int(patient_answers[6])
    q8 = int(patient_answers[7])
    q9 = int(patient_answers[8])
    q10 = int(patient_answers[9])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    # creating Model instance
    model = Model()
    # choosing algorithm
    classifier = model.svm_classifier()
    # predicting answer
    prediction = classifier.predict([values])
    # classification of our prediction
    if prediction[0] == 0:
        result = 'No Depression'
        info = "Congratulations! You don't have any symptoms of depression."
    if prediction[0] == 1:
        result = 'Mild Depression'
        info = "Mild depression, also known as dysthymia, involves persistent low mood and reduced interest in daily activities. Symptoms may include hopelessness, low energy, sleep issues, and difficulty concentrating, though less severe than major depression. Seeking help from a healthcare professional is important for effective management and support."
    if prediction[0] == 2:
        result = 'Moderate Depression'
        info = "Moderate depression is a mood disorder with symptoms more intense than mild depression but less severe than major depression. Symptoms include persistent sadness, changes in appetite or sleep, low energy, difficulty concentrating, and feelings of worthlessness. Seeking professional help is important for diagnosis and effective treatment."
    if prediction[0] == 3:
        result = 'Moderately severe Depression'
        info = "Moderately severe depression is marked by pronounced symptoms that are more intense than moderate depression but less severe than major depression. Symptoms include deep sadness, sleep and appetite disturbances, low energy, difficulty concentrating, and feelings of worthlessness. Seeking professional help is essential for diagnosis and effective treatment."
    if prediction[0] == 4:
        result = 'Severe Depression'
        info = "Severe depression is a serious mood disorder with intense symptoms like deep sadness, disrupted sleep and appetite, extreme fatigue, difficulty concentrating, and feelings of worthlessness. Getting professional help is crucial for diagnosis and treatment."

    print(result, info)
    return jsonify({"result": result, "info": info})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
