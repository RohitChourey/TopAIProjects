import joblib

# function to load the saved model
def load_model():
    model = joblib.load('rf_model.pkl')
    return model

# function to make predictions
def predict(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    return prediction