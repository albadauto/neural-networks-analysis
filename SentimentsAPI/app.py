import pickle

from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import pickle as pck
import numpy as np
app = Flask(__name__)

model = tf.keras.models.load_model('../models/sentiments.keras')
with(open('../models/sentiments.pkl', 'rb')) as token_file:
    sentiments = pck.load(token_file)


sentiments_vocabulary = CountVectorizer(vocabulary=sentiments)
def pre_processing(text):
    text_tokenized = sentiments_vocabulary.transform(text)

    text_tokenized_array = text_tokenized.toarray()
    return text_tokenized_array


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
   text = request.json['text']
   text = pre_processing([text])
   prediction = np.round(model.predict(text))
   json_result = "Frase positiva" if prediction == 1 else "Frase negativa"
   return jsonify({'prediction': json_result})


if __name__ == '__main__':
    app.run(debug=True)