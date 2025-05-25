# federated_simulation.py (SUPER SIMPLIFIED for CSV loading debug)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Keep for later
from sklearn.linear_model import LogisticRegression # Keep for later
import os

NUM_CLIENTS = 5
POISONED_CLIENT_INDEX = 4
FIXED_TEST_SIZE = 100
RANDOM_STATE = 42

def load_and_prep_data():
    print("--- load_and_prep_data START (SUPER SIMPLIFIED DEBUG) ---")
    csv_file_name = "Framingham.csv"
    
    # Try to construct absolute path (good for local, might vary on cloud)
    try:
        abs_path_current_file_dir = os.path.dirname(os.path.abspath(__file__))
        path_option_1 = os.path.join(abs_path_current_file_dir, csv_file_name)
        print(f"DEBUG: Path Option 1 (based on __file__): {path_option_1}")
    except Exception as e_path1:
        print(f"DEBUG: Error constructing Path Option 1: {e_path1}")
        path_option_1 = "ERROR_CONSTRUCTING_PATH_1"

    # Path relative to CWD (often repo root on cloud)
    path_option_2 = csv_file_name
    print(f"DEBUG: Path Option 2 (relative to CWD): {path_option_2}")
    print(f"DEBUG: Current Working Directory (os.getcwd()): {os.getcwd()}")

    # List files in CWD for more context
    try:
        print(f"DEBUG: Files in CWD: {os.listdir('.')}")
    except Exception as e_ls_cwd:
        print(f"DEBUG: Error listing files in CWD: {e_ls_cwd}")

    # List files in __file__ directory for more context
    if path_option_1 != "ERROR_CONSTRUCTING_PATH_1":
        try:
            parent_dir_of_option1 = os.path.dirname(path_option_1)
            if os.path.exists(parent_dir_of_option1):
                 print(f"DEBUG: Files in __file__ directory ({parent_dir_of_option1}): {os.listdir(parent_dir_of_option1)}")
            else:
                print(f"DEBUG: __file__ directory ({parent_dir_of_option1}) does not exist.")
        except Exception as e_ls_file_dir:
            print(f"DEBUG: Error listing files in __file__ directory: {e_ls_file_dir}")


    df = None
    loaded_path = None

    # Attempt 1: Using path based on __file__
    print(f"DEBUG: Attempting to load with Path Option 1: {path_option_1}")
    if path_option_1 != "ERROR_CONSTRUCTING_PATH_1" and os.path.exists(path_option_1):
        try:
            df = pd.read_csv(path_option_1)
            loaded_path = path_option_1
            print(f"SUCCESS: Loaded CSV using Path Option 1: {path_option_1}")
        except Exception as e1:
            print(f"ERROR: Failed to load CSV with Path Option 1 ({path_option_1}): {e1}")
            df = None # Ensure df is None if load fails
    else:
        print(f"INFO: Path Option 1 ({path_option_1}) does not exist or error in path construction.")

    # Attempt 2: Using path relative to CWD (if Attempt 1 failed)
    if df is None:
        print(f"DEBUG: Attempting to load with Path Option 2: {path_option_2}")
        if os.path.exists(path_option_2):
            try:
                df = pd.read_csv(path_option_2)
                loaded_path = path_option_2
                print(f"SUCCESS: Loaded CSV using Path Option 2: {path_option_2}")
            except Exception as e2:
                print(f"ERROR: Failed to load CSV with Path Option 2 ({path_option_2}): {e2}")
                df = None # Ensure df is None
        else:
            print(f"ERROR: Path Option 2 ({path_option_2}) does not exist.")
            print("CRITICAL FAILURE: Framingham.csv NOT FOUND using multiple path strategies.")
            print("--- load_and_prep_data END (FAILURE DUE TO FILE NOT FOUND) ---")
            return None, None, None, None, None

    if df is None:
        print("CRITICAL FAILURE: DataFrame is still None after all attempts to load CSV.")
        print("--- load_and_prep_data END (FAILURE - DF IS NONE) ---")
        return None, None, None, None, None

    print(f"SUCCESS: DataFrame loaded from '{loaded_path}'.")
    print(f"DEBUG: DataFrame Info:")
    df.info(verbose=True, show_counts=True) # Get detailed info
    print(f"DEBUG: DataFrame Head:\n{df.head()}")
    print(f"DEBUG: DataFrame Shape: {df.shape}")
    print(f"DEBUG: DataFrame Columns: {df.columns.tolist()}")
    print(f"DEBUG: Null values per column:\n{df.isnull().sum()}")

    # --- TEMPORARILY COMMENT OUT ALL FURTHER PROCESSING ---
    # df.dropna(inplace=True)
    # print(f"SIM_FED: Dataset shape after dropna: {df.shape}")

    # if "TenYearCHD" not in df.columns:
    #     print("ERROR: 'TenYearCHD' column not found.")
    #     return None, None, None, None, None
        
    # feature_names = df.drop(columns=["TenYearCHD"]).columns.tolist()
    
    # min_data_needed = FIXED_TEST_SIZE + (NUM_CLIENTS * 2)
    # if len(df) < min_data_needed:
    #     print(f"ERROR: Dataset has only {len(df)} rows after NA drop. Need at least {min_data_needed} for this configuration.")
    #     return None, None, None, None, None

    # X = df.drop(columns=["TenYearCHD"])
    # y = df["TenYearCHD"]
    
    # stratify_y = y if y.nunique() > 1 else None
    
    # X_pool, X_test_fixed, y_pool, y_test_fixed = train_test_split(
    #     X, y, test_size=FIXED_TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_y
    # )
    # print(f"SIM_FED: X_pool shape: {X_pool.shape}, X_test_fixed shape: {X_test_fixed.shape}")
    # print("--- load_and_prep_data END (SUCCESS - RETURNING RAW DF FOR NOW) ---")
    # For this debug step, let's return something very basic if load succeeds,
    # or the app will crash later. We'll make the app handle this.
    # We just want to confirm the CSV is read.

    # To make the app.py not crash immediately after this simplified debug version,
    # we need to return the expected number of items, even if they are mostly None or placeholder.
    # We'll assume if df is loaded, we pass it as X_pool and X_test_fixed for now.
    # This is *highly temporary* for debugging CSV load.
    if df is not None and not df.empty:
        print("--- load_and_prep_data END (DEBUG SUCCESS - RETURNING RAW DF AS PLACEHOLDERS) ---")
        # Create placeholder y values if 'TenYearCHD' exists, else dummy
        if "TenYearCHD" in df.columns:
            dummy_y = df["TenYearCHD"]
            feature_names_temp = df.drop(columns=["TenYearCHD"]).columns.tolist()
            df_features_temp = df.drop(columns=["TenYearCHD"])
        else:
            print("WARNING: 'TenYearCHD' not in df for placeholder return. Using dummy y and features.")
            dummy_y = pd.Series(np.zeros(len(df))) # dummy y
            feature_names_temp = [f"col_{i}" for i in range(df.shape[1])] # dummy features
            df_features_temp = df.copy()
            df_features_temp.columns = feature_names_temp

        # Ensure we return 5 items
        return df_features_temp, dummy_y, df_features_temp.copy(), dummy_y.copy(), feature_names_temp
    else:
        print("--- load_and_prep_data END (FAILURE - DF IS NONE OR EMPTY BEFORE RETURN) ---")
        return None, None, None, None, None


# --- Keep other functions as they were, but they might not be called if load_and_prep_data fails early ---
def train_initial_client_models(X_pool, y_pool):
    # ... (your existing train_initial_client_models code) ...
    # Add a print at the start
    print("--- train_initial_client_models START ---")
    if X_pool is None or y_pool is None:
        print("ERROR (train_initial_client_models): X_pool or y_pool is None. Cannot train.")
        return [] # Return empty list of models
    client_models = [] 
    columns_for_dummy = X_pool.columns if not X_pool.empty else [f'f{i}' for i in range(10)]
    num_features_for_dummy = len(columns_for_dummy)
    dummy_X_fallback = pd.DataFrame(np.random.rand(2, num_features_for_dummy), columns=columns_for_dummy)
    dummy_y_fallback = pd.Series([0, 1])

    if X_pool.empty or len(X_pool) < NUM_CLIENTS:
        print(f"WARNING (train_initial_client_models): Pool data insufficient. Using dummy models.")
        for i in range(NUM_CLIENTS):
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
        client_X_train = X_pool_shuffled.iloc[start:end]
        client_y_train = y_pool_shuffled.iloc[start:end]
        print(f"DEBUG: Client {i+1} train data shape: X={client_X_train.shape}, y={client_y_train.shape}, y unique: {client_y_train.nunique() if not client_y_train.empty else 'N/A'}")
        if not client_X_train.empty and client_y_train.nunique() >= 2:
            model.fit(client_X_train, client_y_train)
        elif not client_X_train.empty and client_y_train.nunique() == 1:
            print(f"WARNING: Client {i+1} has only one class. Augmenting.")
            other_class = 0 if client_y_train.iloc[0] == 1 else 1
            temp_dummy_X_sample_values = np.random.rand(1, len(client_X_train.columns))
            temp_dummy_X_sample = pd.DataFrame(temp_dummy_X_sample_values, columns=client_X_train.columns)
            temp_X = pd.concat([client_X_train, temp_dummy_X_sample], ignore_index=True)
            temp_y = pd.concat([client_y_train, pd.Series([other_class])], ignore_index=True)
            model.fit(temp_X, temp_y)
        else: 
            print(f"WARNING: Client {i+1} has insufficient data. Training on dummy fallback.")
            model.fit(dummy_X_fallback, dummy_y_fallback)
        client_models.append(model)
    print("--- train_initial_client_models END ---")
    return client_models

def get_client_predictions_proba(model, X_data):
    # ... (your existing get_client_predictions_proba code, add print at start) ...
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
    # ... (your existing poison_predictions_simple_flip code) ...
    if not isinstance(predictions_proba, np.ndarray):
        predictions_proba = np.array(predictions_proba)
    return 1.0 - predictions_proba

if __name__ == '__main__':
    # ... (your existing if __name__ == '__main__' code, will likely fail with simplified load_and_prep_data) ...
    print("--- Running federated_simulation.py directly for SUPER SIMPLIFIED testing ---")
    load_results = load_and_prep_data()
    if load_results and load_results[0] is not None:
        print(f"Simplified load_and_prep_data returned: X_pool type: {type(load_results[0])}, X_pool shape (if df): {load_results[0].shape if isinstance(load_results[0], pd.DataFrame) else 'N/A'}")
        # The rest of this test block might fail due to the simplified return, that's OK for now.
    else:
        print("Simplified data loading failed during script test.")
    print("--- End of federated_simulation.py direct test ---")