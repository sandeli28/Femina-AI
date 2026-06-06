import sys
import os

# Add the project root to the path so we can import app modules
sys.path.append(os.getcwd())

try:
    from app.ml.model_manager import model_manager
    print("Attempting to load model...")
    model = model_manager.load_model()
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    
    # Check if the file exists on disk
    if os.path.exists("pcos_voting_ensemble.joblib"):
        print("Verified: pcos_voting_ensemble.joblib exists.")
    else:
        print("ERROR: pcos_voting_ensemble.joblib does not exist.")

except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
