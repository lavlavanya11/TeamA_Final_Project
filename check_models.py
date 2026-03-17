import os

model_path = "models/intent_model.pkl"

if os.path.exists(model_path):
    print("Model found")
else:
    print("Model not found. Train model first.")