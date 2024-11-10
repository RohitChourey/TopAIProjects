from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from load import predict  # Import your predict function

# Define the request model with Pydantic
class PredictionRequest(BaseModel):
    input: list[float]  # List of floats for the input data

# Create the FastAPI app
app = FastAPI()

# Load the model (if needed, adjust if predict function already uses a model)
model = joblib.load('rf_model.pkl')

# Define a route for making predictions
@app.post('/predictions')
def predictions(data: PredictionRequest):
    try:
        # Convert input data to numpy array and reshape for prediction
        input_data = np.array(data.input).reshape(1, -1)
        
        # Make prediction
        prediction = predict(input_data)
        
        # Return the prediction as JSON response
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # Handle exceptions with an HTTP error response
        raise HTTPException(status_code=500, detail=str(e))

# Run the application using: `uvicorn filename:app --reload`
