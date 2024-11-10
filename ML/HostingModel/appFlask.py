from flask import Flask, request, jsonify
import joblib
import numpy as np
from load import predict

app = Flask(__name__)

# define a route for making predictions
@app.route('/predictions', methods=['POST'])
def predictions():
    data = request.get_json()  # Get input data from POST request
    input_data = np.array(data['input']).reshape(1, -1)  # Reshape input
    prediction = predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)