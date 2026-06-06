
import joblib
import shap
import pandas as pd
import numpy as np
import os

MODEL_PATH = 'pcos_nonlinear_stack_calibrated.joblib'
EXPLAINER_PATH = 'shap_explainer.joblib'

FEATURE_COLUMNS = [
    'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism', 'Testosterone_ng_per_dL',
    'Menstrual_Cycle', 'LH_FSH_Ratio', 'Conception_Difficulty', 'Family_History_PCOS',
    'Family_History_Menstrual', 'Insulin_Resistance', 'Diabetes', 'Cardiovascular_Disease',
    'Mental_Health', 'Childhood_Trauma', 'Age', 'Weight_kg', 'Height_cm',
    'Diet_Type', 'Veg_or_NonVeg', 'Exercise_hrs_per_week', 'Sleep_hrs_per_night',
    'Stress_score_0_10', 'Smoking_Status'
]

def create_explainer():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model type: {type(model)}")
        
        # Unwrap CalibratedClassifierCV if present
        if hasattr(model, 'calibrated_classifiers_'):
            print("Detected CalibratedClassifierCV. Using the first base estimator for SHAP.")
            # Usually base estimators are trained on folds, or we might need the prefit one
            # If using 'prefit', 'base_estimator' is the one. If 'cv', there are many.
            # Let's check `base_estimator` first.
            if hasattr(model, 'base_estimator'):
                 # Deprecated in newer sklearn, but might exist
                 est = model.base_estimator
            elif hasattr(model, 'estimator'):
                 est = model.estimator
            elif len(model.calibrated_classifiers_) > 0:
                 est = model.calibrated_classifiers_[0].base_estimator
            else:
                 print("Could not retrieve base estimator.")
                 return
            
            print(f"Base estimator type: {type(est)}")
            model_to_explain = est
        elif hasattr(model, 'estimators_'):
             # StackingClassifier
             print("Detected StackingClassifier/VotingClassifier. This is harder for SHAP directly.")
             # We might need to explain the final estimator or one of the base ones?
             # For a global explanation, a KernelExplainer on the predict_proba might be safer 
             # but requires background data.
             model_to_explain = model
        else:
            model_to_explain = model

        # Create dummy background data (SHAP often needs this for model-agnostic or Kernel explainer)
        # We simulate 100 samples based on reasonable ranges or random
        print("Generating synthetic background data...")
        # Note: This is an approximation. Real training data is better.
        n_background = 50
        background_data = {
            'Hormonal_Imbalance': np.random.choice([0, 1], n_background),
            'Hyperandrogenism': np.random.choice([0, 1], n_background),
            'Hirsutism': np.random.choice([0, 1], n_background),
            'Testosterone_ng_per_dL': np.random.uniform(10, 100, n_background),
            'Menstrual_Cycle': np.random.choice([0, 1], n_background), # Assuming binary or categorical
            'LH_FSH_Ratio': np.random.uniform(0.5, 3.0, n_background),
            'Conception_Difficulty': np.random.choice([0, 1], n_background),
            'Family_History_PCOS': np.random.choice([0, 1], n_background),
            'Family_History_Menstrual': np.random.choice([0, 1], n_background),
            'Insulin_Resistance': np.random.choice([0, 1], n_background),
            'Diabetes': np.random.choice([0, 1], n_background),
            'Cardiovascular_Disease': np.random.choice([0, 1], n_background),
            'Mental_Health': np.random.choice([0, 1], n_background),
            'Childhood_Trauma': np.random.choice([0, 1], n_background),
            'Age': np.random.randint(18, 45, n_background),
            'Weight_kg': np.random.uniform(50, 100, n_background),
            'Height_cm': np.random.uniform(150, 180, n_background),
            'Diet_Type': np.random.choice(['Veg', 'Non-Veg'], n_background), # Needs encoding?
            'Veg_or_NonVeg': np.random.choice([0, 1], n_background), # Redundant?
            'Exercise_hrs_per_week': np.random.uniform(0, 10, n_background),
            'Sleep_hrs_per_night': np.random.uniform(4, 9, n_background),
            'Stress_score_0_10': np.random.randint(0, 11, n_background),
            'Smoking_Status': np.random.choice([0, 1], n_background)
        }
        

        # Check if model has encoding pipeline? assuming it accepts raw or we interpret mismatch
        # The check_model_features.py passed, so column names are correct.
        # But are they encoded?
        # Let's inspect the model pipeline steps if possible in a second pass.
        # For now, create DataFrame
        X_background = pd.DataFrame(background_data)
        

        print("Test prediction with background data sample...")
        use_numpy = False
        try:
            test_pred = model.predict_proba(X_background.head())
            print(f"Test prediction shape: {test_pred.shape}")
            print("Model accepted the background data format (DataFrame).")
        except Exception as e:
            print(f"Model rejected DataFrame: {e}")
            print("Trying numpy array...")
            try:
                test_pred = model.predict_proba(X_background.values)
                print(f"Test prediction shape: {test_pred.shape}")
                print("Model accepted the background data format (Numpy).")
                use_numpy = True
            except Exception as e2:
                print(f"\nCRITICAL ERROR: Model rejected both DataFrame and Numpy: {e2}")
                return

        print("Creating SHAP Explainer...")
        try:
            # Try a generic KernelExplainer using predict_proba
            if use_numpy:
                f = lambda x: model.predict_proba(x)[:, 1]
                data_for_shap = X_background.values
            else:
                f = lambda x: model.predict_proba(x)[:, 1]
                data_for_shap = X_background
                
            explainer = shap.KernelExplainer(f, data_for_shap)
            print("SHAP KernelExplainer created successfully.")
            
            print(f"Saving explainer to {EXPLAINER_PATH}...")
            joblib.dump(explainer, EXPLAINER_PATH)
            print("Success!")
            
        except Exception as e:
            print(f"Failed to create KernelExplainer: {e}")
            # Fallback or specific handling could go here

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    create_explainer()
