# federated_simulation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import os

NUM_CLIENTS = 5
POISONED_CLIENT_INDEX = 4
FIXED_TEST_SIZE = 100
RANDOM_STATE = 42

def load_and_prep_data():
    print("--- load_and_prep_data START ---")
    csv_file_name = "Framingham.csv"
    final_path_to_try = csv_file_name # Default to CWD, common for Streamlit Cloud

    # Try to construct absolute path (good for local, might vary on cloud)
    try:
        abs_path_current_file_dir = os.path.dirname(os.path.abspath(__file__))
        path_option_1 = os.path.join(abs_path_current_file_dir, csv_file_name)
        if os.path.exists(path_option_1):
            final_path_to_try = path_option_1
            print(f"DEBUG: Using path based on __file__: {final_path_to_try}")
        else:
            print(f"DEBUG: Path from __file__ ({path_option_1}) not found. Using relative path: {final_path_to_try}")
    except Exception as e_path1:
        print(f"DEBUG: Error constructing path based on __file__: {e_path1}. Using relative path: {final_path_to_try}")

    print(f"DEBUG: Attempting to load Framingham.csv from final path: {os.path.abspath(final_path_to_try)}")
    
    df = None
    try:
        df = pd.read_csv(final_path_to_try)
        print(f"SUCCESS: DataFrame loaded from '{final_path_to_try}'. Initial shape: {df.shape}")
        print(f"DEBUG: Initial DataFrame Info:")
        df.info(verbose=False) # Less verbose for now
        print(f"DEBUG: Initial Null values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}") # Show only columns with NaNs
    except FileNotFoundError:
        print(f"CRITICAL FAILURE: Framingham.csv NOT FOUND at '{final_path_to_try}'.")
        print(f"DEBUG: Current Working Directory: {os.getcwd()}")
        try:
            print(f"DEBUG: Files in CWD: {os.listdir('.')}")
        except: pass
        return None, None, None, None, None
    except Exception as e:
        print(f"CRITICAL FAILURE: Error loading CSV '{final_path_to_try}': {e}")
        return None, None, None, None, None

    # --- CRITICAL DATA CLEANING ---
    print(f"DEBUG: Number of rows before df.dropna(): {len(df)}")
    df.dropna(inplace=True) # Ensure this is active
    print(f"DEBUG: Number of rows AFTER df.dropna(): {len(df)}")
    
    if df.empty:
        print("ERROR: DataFrame is empty after df.dropna(). Cannot proceed.")
        return None, None, None, None, None

    if "TenYearCHD" not in df.columns:
        print("ERROR: 'TenYearCHD' column not found in DataFrame AFTER dropna.")
        print(f"DEBUG: Available columns: {df.columns.tolist()}")
        return None, None, None, None, None
        
    feature_names = df.drop(columns=["TenYearCHD"]).columns.tolist()
    
    min_samples_per_client_train = 2 
    min_data_for_pool = NUM_CLIENTS * min_samples_per_client_train
    min_data_needed = FIXED_TEST_SIZE + min_data_for_pool 
    
    if len(df) < min_data_needed:
        print(f"ERROR: Dataset has only {len(df)} rows after NA drop. Need at least {min_data_needed}.")
        return None, None, None, None, None

    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    if X.isnull().sum().sum() > 0: # Check if X still has NaNs BEFORE split
        print(f"WARNING: X contains NaNs even after global dropna. Sum of NaNs: {X.isnull().sum().sum()}")
        print(f"DEBUG: NaNs per column in X before split:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
        # Optionally, apply a more aggressive drop or imputation here if this occurs
        # X.fillna(X.mean(), inplace=True) # Example: Mean imputation - CHOOSE STRATEGY CAREFULLY

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
        return None, None, None, None, None

    print(f"DEBUG: Data split successful. X_pool shape: {X_pool.shape}, X_test_fixed shape: {X_test_fixed.shape}")
    print("--- load_and_prep_data END (SUCCESS) ---")
    return X_pool, y_pool, X_test_fixed, y_test_fixed, feature_names

# Step 2: Add specific NaN handling within `train_initial_client_models`
def train_initial_client_models(X_pool, y_pool):
    print("--- train_initial_client_models START ---")
    if X_pool is None or y_pool is None:
        print("ERROR (train_initial_client_models): X_pool or y_pool is None.")
        return [] 
    client_models = [] 
    columns_for_dummy = X_pool.columns if not X_pool.empty else [f'f{i}' for i in range(10)]
    num_features_for_dummy = len(columns_for_dummy)
    dummy_X_fallback = pd.DataFrame(np.random.rand(2, num_features_for_dummy), columns=columns_for_dummy)
    dummy_y_fallback = pd.Series([0, 1])

    if X_pool.empty or len(X_pool) < NUM_CLIENTS:
        print(f"WARNING (train_initial_client_models): Pool data insufficient. Using dummy models.")
        for _ in range(NUM_CLIENTS):
            model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, max_iter=100)
            model.fit(dummy_X_fallback, dummy_y_fallback) 
            client_models.append(model)
        return client_models
    
    shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(len(X_pool))
    X_pool_shuffled = X_pool.iloc[shuffled_indices].reset_index(drop=True)
    y_pool_shuffled = y_pool.iloc[shuffled_indices].reset_index(drop=True)
    split_size = max(1, len(X_pool_shuffled) // NUM_CLIENTS) 
    print(f"DEBUG (train_initial_client_models): X_pool_shuffled shape: {X_pool_shuffled.shape}, split_size: {split_size}")

    for i in range(NUM_CLIENTS):
        model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, max_iter=100)
        start = i * split_size
        end = (i + 1) * split_size if i < NUM_CLIENTS - 1 else len(X_pool_shuffled)
        
        client_X_train_orig = X_pool_shuffled.iloc[start:end]
        client_y_train_orig = y_pool_shuffled.iloc[start:end]

        print(f"DEBUG: Client {i+1} original train data shape: X={client_X_train_orig.shape}, y={client_y_train_orig.shape}")
        
        # --- ADD NaN HANDLING FOR CLIENT DATA ---
        if client_X_train_orig.isnull().sum().sum() > 0:
            print(f"WARNING: Client {i+1} X_train contains NaNs BEFORE explicit drop/impute. Sum: {client_X_train_orig.isnull().sum().sum()}")
            # Option 1: Drop rows with NaNs specific to this client's slice
            # This might make the client's dataset too small or empty
            client_X_train = client_X_train_orig.copy() # Work on a copy
            client_y_train = client_y_train_orig.copy()
            
            # Get indices of rows with NaNs in X
            nan_rows_in_X = client_X_train[client_X_train.isnull().any(axis=1)].index
            # Drop these rows from both X and y
            client_X_train.drop(nan_rows_in_X, inplace=True)
            client_y_train.drop(nan_rows_in_X, inplace=True)
            print(f"DEBUG: Client {i+1} X_train shape after dropping its NaN rows: {client_X_train.shape}")

            # Option 2: Imputation (example: mean imputation) - CHOOSE ONE STRATEGY
            # client_X_train = client_X_train_orig.fillna(client_X_train_orig.mean())
            # client_y_train = client_y_train_orig # y usually doesn't need imputation unless target is missing
            # print(f"DEBUG: Client {i+1} X_train imputed. Original NaNs sum: {client_X_train_orig.isnull().sum().sum()}, New NaNs sum: {client_X_train.isnull().sum().sum()}")
        else:
            client_X_train = client_X_train_orig
            client_y_train = client_y_train_orig
        # --- END NaN HANDLING FOR CLIENT DATA ---

        print(f"DEBUG: Client {i+1} FINAL train data shape: X={client_X_train.shape}, y={client_y_train.shape}, y unique: {client_y_train.nunique() if not client_y_train.empty else 'N/A'}")

        if not client_X_train.empty and client_y_train.nunique() >= 2:
            model.fit(client_X_train, client_y_train)
        elif not client_X_train.empty and client_y_train.nunique() == 1:
            print(f"WARNING: Client {i+1} has only one class ({client_y_train.iloc[0] if not client_y_train.empty else 'N/A'}). Augmenting.")
            other_class = 0 if client_y_train.iloc[0] == 1 else 1
            temp_dummy_X_sample_values = np.random.rand(1, len(client_X_train.columns))
            temp_dummy_X_sample = pd.DataFrame(temp_dummy_X_sample_values, columns=client_X_train.columns)
            temp_X = pd.concat([client_X_train, temp_dummy_X_sample], ignore_index=True)
            temp_y = pd.concat([client_y_train, pd.Series([other_class])], ignore_index=True)
            model.fit(temp_X, temp_y)
        else: 
            print(f"WARNING: Client {i+1} has insufficient/empty data after NaN handling. Training on dummy fallback.")
            model.fit(dummy_X_fallback, dummy_y_fallback)
        client_models.append(model)
    print("--- train_initial_client_models END ---")
    return client_models

# Keep get_client_predictions_proba and poison_predictions_simple_flip as they are
# from your last working version.
def get_client_predictions_proba(model, X_data):
    # ... (your existing get_client_predictions_proba code) ...
    print("--- get_client_predictions_proba START ---")
    if X_data.empty:
        print("WARN (get_client_predictions_proba): Received empty X_data. Returning empty array.")
        return np.array([])
    try:
        if hasattr(model, 'n_features_in_') and X_data.shape[1] != model.n_features_in_:
            print(f"ERROR (get_client_predictions_proba): Feature mismatch. Model expects {model.n_features_in_}, got {X_data.shape[1]}.")
            return np.full(len(X_data), 0.5)
    except AttributeError:
        pass
    try:
        return model.predict_proba(X_data)[:, 1] 
    except ValueError as e:
        print(f"ERROR (ValueError) during predict_proba: {e}.")
        return np.full(len(X_data), 0.5)
    except Exception as e:
        print(f"UNEXPECTED ERROR during predict_proba: {e}")
        return np.full(len(X_data), 0.5)

def poison_predictions_simple_flip(predictions_proba):
    if not isinstance(predictions_proba, np.ndarray):
        predictions_proba = np.array(predictions_proba)
    return 1.0 - predictions_proba

if __name__ == '__main__':
    print("--- Running federated_simulation.py directly for testing ---")
    load_results = load_and_prep_data()
    if load_results and load_results[0] is not None:
        X_p, y_p, X_t_f, y_t_f, f_names = load_results
        print(f"Data loaded: X_pool shape {X_p.shape}, X_test_fixed shape {X_t_f.shape}, Features: {f_names}")
        print(f"X_pool NaNs: {X_p.isnull().sum().sum()}, y_pool NaNs: {y_p.isnull().sum()}") # Check NaNs
        
        models = train_initial_client_models(X_p, y_p)
        print(f"Trained {len(models)} client models.")
        if models and not X_t_f.empty:
            if set(X_t_f.columns) == set(f_names):
                sample_patient_data_df = X_t_f[f_names].iloc[[0]]
                print(f"Test patient data (1 sample) columns: {sample_patient_data_df.columns.tolist()}")
                for i, model_instance in enumerate(models):
                    try:
                        pred = get_client_predictions_proba(model_instance, sample_patient_data_df)
                        print(f"Client {i+1} prediction for sample: {pred}")
                    except Exception as e:
                        print(f"Error predicting with client {i+1} for sample patient: {e}")
            else:
                print("ERROR: Test data columns do not match expected feature names.")
        elif X_t_f.empty: print("Test data (X_t_f) is empty.")
        elif not models: print("Client models not trained.")
    else:
        print("Data loading failed.")
    print("--- End of federated_simulation.py direct test ---")