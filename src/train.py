# fl_dashboard/src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier # Using RandomForest
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving the model
import os
import numpy as np # Ensure numpy is imported

# Import utility functions from utils.py
# This assumes utils.py is in the same src/ directory
try:
    from utils import load_and_preprocess_data, RANDOM_STATE
except ImportError:
    print("Error: Could not import from utils.py. Make sure it's in the src/ directory and has no errors.")
    exit()


MODEL_DIR = "models" # Relative to the project root if script is run from root
GLOBAL_MODEL_FILENAME = "global_model.pkl"
DATA_DIR = "data"
DATASET_FILENAME = "framingham.csv" # <<< IMPORTANT: CHANGE THIS IF YOUR 100K DATASET HAS A DIFFERENT NAME

def train_baseline_model():
    """
    Trains a baseline global model on the entire preprocessed dataset (or a train split of it)
    and saves it. Prints its accuracy on a test split.
    """
    print("--- Phase 2: Baseline Model Training ---")

    # Construct paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of train.py (src/)
    project_root = os.path.dirname(script_dir) # Go up one level to fl_dashboard/
    
    models_path = os.path.join(project_root, MODEL_DIR)
    data_file_path = os.path.join(project_root, DATA_DIR, DATASET_FILENAME)

    print(f"Attempting to load data from: {data_file_path}")
    print(f"Models will be saved to: {models_path}")

    if not os.path.exists(models_path):
        try:
            os.makedirs(models_path)
            print(f"Created directory: {models_path}")
        except OSError as e:
            print(f"Error creating models directory {models_path}: {e}")
            return None


    # 1. Load and preprocess all data
    # The load_and_preprocess_data function now returns X, y, original_X_columns
    loaded_data = load_and_preprocess_data(file_path=data_file_path) 
    
    if loaded_data is None or loaded_data[0] is None or loaded_data[1] is None:
        print("Failed to load or preprocess data. Exiting baseline training.")
        return None
    
    X, y, original_X_cols = loaded_data

    if X.empty or y.empty:
        print("Error: Data is empty after preprocessing. Cannot train.")
        return None

    # Check class distribution
    print("\nTarget variable distribution (TenYearCHD):")
    if y.nunique() > 0: # Check if y is not empty and has values
        print(y.value_counts(normalize=True))
        is_balanced = y.value_counts(normalize=True).min() > 0.4 # Simple check for balance
    else:
        print("Warning: Target variable 'y' is empty or has no variance.")
        is_balanced = False # Assume not balanced if no data to check

    # 2. Split data into training and testing sets for this baseline model
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
        )
    except ValueError as e:
        print(f"Error during train_test_split: {e}. This might happen if the dataset is too small or imbalanced after preprocessing.")
        # Fallback to no stratification if that's the issue and y has samples
        if len(y) > 1:
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
        else:
            print("Cannot perform train_test_split with current data.")
            return None


    if X_train.empty or y_train.empty:
        print("Error: Training data is empty after split. Cannot train baseline model.")
        return None

    print(f"\nTraining baseline model on {X_train.shape[0]} samples...")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Use original_X_cols if X_train.columns is not available (e.g. if X_train is a numpy array from utils)
    # However, load_and_preprocess_data should return X as a DataFrame.
    current_feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else original_X_cols
    if current_feature_names:
        print(f"Feature names: {current_feature_names}")
    else:
        print("Warning: Could not determine feature names for display.")


    # 3. Fit a global classifier
    model = RandomForestClassifier(
        n_estimators=100,       
        random_state=RANDOM_STATE,
        max_depth=20,           
        min_samples_split=5,    
        min_samples_leaf=2,     
        class_weight='balanced' if not is_balanced and y_train.nunique() > 1 else None, 
        n_jobs=-1               
    )
    
    try:
        model.fit(X_train, y_train)
        print("Baseline model trained.")
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return None

    # 4. Evaluate the model
    accuracy = None
    if not X_test.empty:
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nBaseline Global Model Accuracy on test set: {accuracy:.4f} (approx {(accuracy*100):.1f}%)")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            if hasattr(model, 'feature_importances_') and current_feature_names:
                print("\nTop 10 Feature Importances:")
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({'feature': current_feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10)
                print(feature_importance_df)
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            accuracy = None # Ensure accuracy is None if evaluation fails
    else:
        print("Warning: Test set is empty. Cannot evaluate baseline model accuracy.")

    # 5. Save the trained model
    model_save_path = os.path.join(models_path, GLOBAL_MODEL_FILENAME)
    try:
        joblib.dump(model, model_save_path)
        print(f"\nBaseline global model saved to: {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        
    return accuracy


if __name__ == '__main__':
    # This structure assumes you run `python src/train.py` from the `fl_dashboard` root directory.
    # If you run `python train.py` directly from within `src/`, the relative paths for data/models need adjustment,
    # which is handled by using os.path.dirname(__file__) and os.path.join.
    train_baseline_model()