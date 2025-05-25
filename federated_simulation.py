# federated_simulation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Simpler model for speed
from sklearn.metrics import accuracy_score, mean_squared_error # precision_score, recall_score are in app.py
import os

NUM_CLIENTS = 5
POISONED_CLIENT_INDEX = 4 # Client 5 (0-indexed)
FIXED_TEST_SIZE = 100 
RANDOM_STATE = 42

def load_and_prep_data():
    csv_file_path = "Framingham.csv" # Assuming it's in the same directory as this script or app.py
    try:
        # More robust path finding for deployment vs local
        base_path = os.path.dirname(os.path.abspath(__file__))
        deployment_path = os.path.join(base_path, csv_file_path)

        if os.path.exists(deployment_path):
            final_path_to_try = deployment_path
        else: # Fallback to current working directory (common for Streamlit Cloud root)
            final_path_to_try = csv_file_path 
            print(f"DEPLOYMENT_DEBUG: Path from __file__ ({deployment_path}) not found. Trying direct path: {final_path_to_try}")

        print(f"DEPLOYMENT_DEBUG: Attempting to load Framingham.csv from final path: {os.path.abspath(final_path_to_try)}")
        print(f"DEPLOYMENT_DEBUG: Current working directory: {os.getcwd()}")
        try:
            print(f"DEPLOYMENT_DEBUG: Files in CWD: {os.listdir('.')}")
        except Exception as e_ls:
            print(f"DEPLOYMENT_DEBUG: Could not list files in CWD: {e_ls}")
        
        df = pd.read_csv(final_path_to_try)
        print(f"DEPLOYMENT_DEBUG: Framingham.csv loaded successfully. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR (FileNotFoundError) in federated_simulation.py: Framingham.csv not found. Tried path: {final_path_to_try}")
        return None, None, None, None, None
    except Exception as e:
        print(f"ERROR (Other Exception) in federated_simulation.py loading Framingham.csv: {e}")
        return None, None, None, None, None
    
    print(f"DEPLOYMENT_DEBUG: DataFrame columns before dropna: {df.columns.tolist()}")
    print(f"DEPLOYMENT_DEBUG: Number of rows before dropna: {len(df)}")
    df.dropna(inplace=True) 
    print(f"DEPLOYMENT_DEBUG: Dataset shape after dropna: {df.shape}")
    print(f"DEPLOYMENT_DEBUG: Number of rows AFTER dropna: {len(df)}")

    if "TenYearCHD" not in df.columns:
        print("ERROR: 'TenYearCHD' column not found in DataFrame after loading and dropna.")
        print(f"DEBUG: Available columns: {df.columns.tolist()}")
        return None, None, None, None, None
        
    feature_names = df.drop(columns=["TenYearCHD"]).columns.tolist()
    
    # Adjusted min_data_needed to be more realistic for train/test split AND client partitioning
    # Each client needs at least 2 samples for training if possible, plus the fixed test set.
    # Let's say each client gets at least `min_samples_per_client_train`
    min_samples_per_client_train = 2 # At least 2 for Logistic Regression to work with two classes if stratified
    min_data_for_pool = NUM_CLIENTS * min_samples_per_client_train
    min_data_needed = FIXED_TEST_SIZE + min_data_for_pool 
    
    if len(df) < min_data_needed:
        print("ERROR in federated_simulation.py: Dataset issues check triggered due to insufficient data after NA drop.")
        print(f"DEBUG: Columns in DataFrame after dropna: {df.columns.tolist()}")
        print(f"DEBUG: 'TenYearCHD' in columns: {'TenYearCHD' in df.columns}")
        print(f"DEBUG: Length of DataFrame after dropna: {len(df)}")
        print(f"DEBUG: Required total data (test_set + client_pool_minimum): {min_data_needed} (Test: {FIXED_TEST_SIZE}, Min Pool: {min_data_for_pool})")
        return None, None, None, None, None

    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    # Check for sufficient data in X_pool after test split before client partitioning
    if len(X) - FIXED_TEST_SIZE < min_data_for_pool:
        print(f"ERROR: Not enough data for client pool after reserving test set. Pool would have {len(X) - FIXED_TEST_SIZE} samples, need {min_data_for_pool}.")
        return None, None, None, None, None

    stratify_y = y if y.nunique() > 1 else None
    
    try:
        X_pool, X_test_fixed, y_pool, y_test_fixed = train_test_split(
            X, y, test_size=FIXED_TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_y
        )
    except ValueError as e_split:
        print(f"ERROR during train_test_split: {e_split}")
        print(f"DEBUG: X shape: {X.shape}, y shape: {y.shape}, test_size: {FIXED_TEST_SIZE}, stratify_y unique values: {y.nunique() if stratify_y is not None else 'None'}")
        return None, None, None, None, None

    print(f"DEPLOYMENT_DEBUG: Data split successful. X_pool shape: {X_pool.shape}, X_test_fixed shape: {X_test_fixed.shape}")
    return X_pool, y_pool, X_test_fixed, y_test_fixed, feature_names

def train_initial_client_models(X_pool, y_pool): # Expects X_pool and y_pool
    client_models = [] # Returns a LIST of models
    
    # Use actual column names from X_pool if available for dummy data
    columns_for_dummy = X_pool.columns if not X_pool.empty else [f'f{i}' for i in range(10)] # Default 10 features
    num_features_for_dummy = len(columns_for_dummy)

    dummy_X_fallback = pd.DataFrame(np.random.rand(2, num_features_for_dummy), columns=columns_for_dummy)
    dummy_y_fallback = pd.Series([0, 1])

    if X_pool.empty or len(X_pool) < NUM_CLIENTS: # Check if pool itself is too small
        print(f"WARNING (train_initial_client_models): Pool data (X_pool has {len(X_pool)} samples) is insufficient for {NUM_CLIENTS} clients. Using dummy models for all clients.")
        for i in range(NUM_CLIENTS):
            model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, max_iter=100)
            model.fit(dummy_X_fallback, dummy_y_fallback) 
            client_models.append(model)
        return client_models

    shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(len(X_pool))
    X_pool_shuffled = X_pool.iloc[shuffled_indices].reset_index(drop=True)
    y_pool_shuffled = y_pool.iloc[shuffled_indices].reset_index(drop=True)

    # Ensure at least 1 sample per client, even if it means overlap or very small datasets
    split_size = max(1, len(X_pool_shuffled) // NUM_CLIENTS) 
    
    print(f"DEBUG (train_initial_client_models): X_pool_shuffled shape: {X_pool_shuffled.shape}, split_size per client: {split_size}")

    for i in range(NUM_CLIENTS):
        model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, max_iter=100)
        start = i * split_size
        # Ensure the last client gets all remaining data
        end = (i + 1) * split_size if i < NUM_CLIENTS - 1 else len(X_pool_shuffled)
        
        client_X_train = X_pool_shuffled.iloc[start:end]
        client_y_train = y_pool_shuffled.iloc[start:end]

        print(f"DEBUG: Client {i+1} training data shape: X={client_X_train.shape}, y={client_y_train.shape}, y unique: {client_y_train.nunique() if not client_y_train.empty else 'N/A'}")

        if not client_X_train.empty and client_y_train.nunique() >= 2:
            model.fit(client_X_train, client_y_train)
        elif not client_X_train.empty and client_y_train.nunique() == 1:
            print(f"WARNING: Client {i+1} has only one class ({client_y_train.iloc[0]}). Augmenting with a dummy sample of the other class for LR training.")
            other_class = 0 if client_y_train.iloc[0] == 1 else 1
            # Create a dummy sample that matches the feature structure of client_X_train
            # Use one row from dummy_X_fallback, ensuring columns match client_X_train
            temp_dummy_X_sample_values = np.random.rand(1, len(client_X_train.columns))
            temp_dummy_X_sample = pd.DataFrame(temp_dummy_X_sample_values, columns=client_X_train.columns)
            
            temp_X = pd.concat([client_X_train, temp_dummy_X_sample], ignore_index=True)
            temp_y = pd.concat([client_y_train, pd.Series([other_class])], ignore_index=True)
            model.fit(temp_X, temp_y)
        else: 
            print(f"WARNING: Client {i+1} has insufficient/empty data (after split). Training on minimal dummy data (from fallback).")
            model.fit(dummy_X_fallback, dummy_y_fallback) # Fallback if client data is truly empty
            
        client_models.append(model)
    return client_models

def get_client_predictions_proba(model, X_data):
    if X_data.empty:
        print("WARN (get_client_predictions_proba): Received empty X_data. Returning default probability 0.5.")
        # Return an array with shape (0, ) or (1, ) depending on expectation.
        # If X_data was for multiple samples, returning a single 0.5 might be an issue.
        # Let's assume if X_data is empty, it means 0 predictions needed.
        return np.array([]) # Or handle as error / return array of 0.5s matching expected output len
    
    # Defensive check for feature mismatch
    try:
        # This assumes model object has a way to get expected feature names/count
        # For scikit-learn, model.feature_names_in_ or model.n_features_in_ (after fit)
        if hasattr(model, 'n_features_in_') and X_data.shape[1] != model.n_features_in_:
            print(f"ERROR (get_client_predictions_proba): Feature mismatch. Model expects {model.n_features_in_}, got {X_data.shape[1]} for X_data.")
            print(f"DEBUG: Model features: {getattr(model, 'feature_names_in_', 'Not available')}")
            print(f"DEBUG: X_data columns: {X_data.columns.tolist()}")
            return np.full(len(X_data), 0.5) # Return default for all samples in X_data
    except AttributeError:
        pass # Model might not have n_features_in_ (e.g., if not yet fit or different type)

    try:
        return model.predict_proba(X_data)[:, 1] 
    except ValueError as e: # Often due to feature mismatch or unexpected data
        print(f"ERROR (ValueError) during predict_proba: {e}.")
        print(f"DEBUG: X_data columns for this prediction: {X_data.columns.tolist()}")
        if hasattr(model, 'feature_names_in_'):
             print(f"DEBUG: Model expected features: {model.feature_names_in_}")
        return np.full(len(X_data), 0.5) # Return array of 0.5s
    except Exception as e:
        print(f"UNEXPECTED ERROR during predict_proba: {e}")
        return np.full(len(X_data), 0.5) # Return array of 0.5s

def poison_predictions_simple_flip(predictions_proba):
    # Ensure it's a numpy array for element-wise operations
    if not isinstance(predictions_proba, np.ndarray):
        predictions_proba = np.array(predictions_proba)
    return 1.0 - predictions_proba

if __name__ == '__main__':
    print("--- Running federated_simulation.py directly for testing ---")
    load_results = load_and_prep_data() # This now returns X_pool, y_pool, X_test_fixed, y_test_fixed, feature_names
    
    if load_results and load_results[0] is not None: # Check if X_pool (first element) is not None
        X_p, y_p, X_t_f, y_t_f, f_names = load_results # Unpack all 5
        print(f"Data loaded: X_pool shape {X_p.shape}, X_test_fixed shape {X_t_f.shape}, Features: {f_names}")
        
        models = train_initial_client_models(X_p, y_p) # Correctly pass X_pool, y_pool
        print(f"Trained {len(models)} client models.")
        
        if models and not X_t_f.empty:
            # Ensure sample_patient_data_df has columns in the same order as f_names
            # This is crucial if X_t_f columns got reordered or are different from training
            if set(X_t_f.columns) == set(f_names):
                sample_patient_data_df = X_t_f[f_names].iloc[[0]] # Use f_names to order/select columns
                print(f"Test patient data (1 sample) columns: {sample_patient_data_df.columns.tolist()}")
                
                for i, model_instance in enumerate(models):
                    try:
                        pred = get_client_predictions_proba(model_instance, sample_patient_data_df)
                        print(f"Client {i+1} prediction for sample: {pred}")
                    except Exception as e:
                        print(f"Error predicting with client {i+1} for sample patient: {e}")
            else:
                print("ERROR: Test data columns do not match expected feature names. Cannot run prediction test.")
                print(f"Test data columns: {X_t_f.columns.tolist()}")
                print(f"Expected features: {f_names}")

        elif X_t_f.empty:
            print("Test data (X_t_f) is empty, cannot run prediction test on patient sample.")
        elif not models:
            print("Client models were not trained successfully.")
            
    else:
        print("Data loading failed or returned None components during script test.")
    print("--- End of federated_simulation.py direct test ---")