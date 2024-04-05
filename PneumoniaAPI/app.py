from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('C:/Users/José Adauto/PycharmProjects/pythonProject/models/Pneumonia.keras')

def preProcessaImage(image):
    image_to_process = Image.open(image).convert('RGB')
    image_to_process = image_to_process.resize((128,128))

    image_asarray = np.asarray(image_to_process)
    image_asarray = np.expand_dims(image_asarray, axis=0)
    image_asarray = np.sum(image_asarray / 3, axis=3, keepdims=True)
    image_asarray = (image_asarray - 128) / 128
    return image_asarray

@app.route("/", methods=['GET'])
def hello_world():
    return "Hello"


@app.route("/predict_pneumonia", methods=["POST"])
def process_image():
    dictPneumonia = {
        0: "Normal",
        1: "Pneumonia"
    }
    file = request.files['image']
    img_toarray = preProcessaImage(file)
    predict = np.argmax(model.predict(img_toarray), axis=1)[0]
    result = dictPneumonia[predict]
    return jsonify({"result": f"O resultado do raio X consta que o pulmão está: {result}"})


if __name__ == "__main__":
    app.run(debug=True)