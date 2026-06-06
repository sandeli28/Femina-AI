import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# IMPORTS FIXED
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------
DATA_PATH = r"E:\pcos\pcos_synthetic_50000.csv"
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Define Features and Target
target = 'PCOS'
X = df.drop(columns=[target])
y = df[target]

# ------------------------------------------------------------------------------
# 2. FEATURE GROUPS (Based on Notebook Analysis)
# ------------------------------------------------------------------------------
# These groups match the notebook's preprocessing logic
numerical_cols = [
    'Age', 'Weight_kg', 'Height_cm', 'Testosterone_ng_per_dL',
    'LH_FSH_Ratio', 'Exercise_hrs_per_week', 'Sleep_hrs_per_night', 'Stress_score_0_10'
]

binary_cols = [
    'Menstrual_Cycle', 'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
    'Conception_Difficulty', 'Family_History_PCOS', 'Family_History_Menstrual',
    'Insulin_Resistance', 'Diabetes', 'Cardiovascular_Disease', 'Mental_Health',
    'Childhood_Trauma', 'Smoking_Status'
]

multi_cat_cols = ['Diet_Type', 'Veg_or_NonVeg']

# ------------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE (ColumnTransformer)
# ------------------------------------------------------------------------------
# Numerical: Impute Mean -> Scale
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Binary/Ordinal: Impute Mode -> Ordinal Encode
bin_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Multi-Categorical: Impute Mode -> OneHot Encode
multi_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numerical_cols),
    ('bin', bin_pipe, binary_cols),
    ('multi', multi_pipe, multi_cat_cols)
], verbose_feature_names_out=False)

# ------------------------------------------------------------------------------
# MANUAL VOTING CLASSIFIER (Bypasses VotingClassifier strict checks)
# ------------------------------------------------------------------------------
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class ManualVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        self.fitted_estimators_ = []

    def fit(self, X, y):
        # Fit each model manually
        self.fitted_estimators_ = []
        for name, model in self.estimators:
            print(f"  Training {name}...")
            # We don't verify tags here, just fit
            # Check if it needs wrapping for sklearn 1.6 tags issue? 
            # If we call model.fit directly, it should be fine as long as we don't pass it to a strict meta-learner.
            # However, if 'model' is CatBoost, it might fail inside CalibratedCV if we didn't wrap it.
            # But here we are INSIDE the loop. 
            fitted_model = clone(model) if hasattr(model, 'get_params') else model
            fitted_model.fit(X, y)
            self.fitted_estimators_.append(fitted_model)
        
        self.classes_ = np.unique(y)
        self.classes_ = np.unique(y)
        # self._estimator_type = "classifier" # Handled by property
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        # Average probabilities
        avg_proba = None
        for model in self.fitted_estimators_:
            proba = model.predict_proba(X)
            if avg_proba is None:
                avg_proba = proba
            else:
                avg_proba += proba
        avg_proba /= len(self.fitted_estimators_)
        return avg_proba
    
    @property
    def _estimator_type(self):
        return "classifier"

    def __sklearn_tags__(self):
        # Minimal tags for sklearn 1.6
        from sklearn.utils._tags import _DEFAULT_TAGS
        tags = _DEFAULT_TAGS.copy()
        tags["estimator_type"] = "classifier" 
        return tags

# Hyperparameters
rf = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=30, n_jobs=1)

# Usage with Manual Voting
# Note: We use the ORIGINAL classes because we are managing the fit manually.
# But CatBoost still needs to be importable.
from xgboost import XGBClassifier as OriginalXGBClassifier
from lightgbm import LGBMClassifier as OriginalLGBMClassifier
from catboost import CatBoostClassifier as OriginalCatBoostClassifier

xgb_clf = OriginalXGBClassifier(
    n_estimators=350, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=30, eval_metric='logloss', n_jobs=1
)

lgbm_clf = OriginalLGBMClassifier(
    n_estimators=350, learning_rate=0.05, num_leaves=31, random_state=30, verbose=-1, n_jobs=1
)

class SklearnCompatibleCatBoostClassifier(OriginalCatBoostClassifier):
    def __sklearn_tags__(self):
        from sklearn.utils._tags import _DEFAULT_TAGS
        tags = _DEFAULT_TAGS.copy()
        tags["estimator_type"] = "classifier"
        return tags

cat_clf = SklearnCompatibleCatBoostClassifier(
    iterations=350, learning_rate=0.05, depth=6, verbose=0, random_state=30
)

svm_clf = SVC(
    kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=30
)

estimators_list = [
    ('rf', rf),
    ('xgb', xgb_clf),
    ('lgbm', lgbm_clf),
    ('cat', cat_clf),
    ('svm', svm_clf)
]

manual_voting_clf = ManualVotingClassifier(estimators=estimators_list)

# ------------------------------------------------------------------------------
# 5. VOTING ENSEMBLE
# ------------------------------------------------------------------------------
# Full Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', manual_voting_clf) # Using Manual Voting Ensemble
])

# ------------------------------------------------------------------------------
# 6. TRAINING & EVALUATION
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ------------------------------------------------------------------------------
# 7. SAVE MODEL PARAM
# ------------------------------------------------------------------------------
OUTPUT_FILE = 'pcos_voting_ensemble.joblib'
print("\nTraining Voting Ensemble (this may take a few minutes)...")
# Wrap in CalibratedClassifierCV as per new.ipynb
calibrated_pipeline = CalibratedClassifierCV(model_pipeline, method='isotonic', cv=3)
calibrated_pipeline.fit(X_train, y_train)

print(f"\nSaving model to {OUTPUT_FILE}...")
joblib.dump(calibrated_pipeline, OUTPUT_FILE)

print("\nEvaluating Model...")
y_pred = calibrated_pipeline.predict(X_test)
y_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print("Done!")
