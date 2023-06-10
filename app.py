import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify

import random
import pickle

app = Flask(__name__)

# Load data from intents.json file
with open('intents.json') as file:
    data = json.load(file)

# Load trained model
model = keras.models.load_model('chat_model.h5')

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['input']
    
    if user_input.lower() == 'quit':
        response = "Terimakasih, sampai jumpa lagi!"
    else:
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                                         truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        found_tag = False

        for intent in data['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                found_tag = True

        if not found_tag:
            response = "Data tidak tersedia di dataset."

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
