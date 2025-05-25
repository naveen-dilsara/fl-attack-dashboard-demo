# federated_simulation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Simpler model for speed
from sklearn.metrics import accuracy_score, mean_squared_error
import os

NUM_CLIENTS = 5
POISONED_CLIENT_INDEX = 4 # Client 5 (0-indexed)
FIXED_TEST_SIZE = 100 
RANDOM_STATE = 42

def load_and_prep_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(current_dir, "Framingham.csv")
        print(f"SIM_FED: Attempting to load CSV from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"ERROR in federated_simulation.py: Framingham.csv not found at: {csv_file_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"ERROR loading Framingham.csv: {e}")
        return None, None, None, None, None

    df.dropna(inplace=True)
    print(f"SIM_FED: Dataset shape after dropna: {df.shape}")

    if "TenYearCHD" not in df.columns:
        print("ERROR: 'TenYearCHD' column not found.")
        return None, None, None, None, None
        
    feature_names = df.drop(columns=["TenYearCHD"]).columns.tolist()
    
    min_data_needed = FIXED_TEST_SIZE + (NUM_CLIENTS * 2) # *2 for at least 2 samples per client for training
    if len(df) < min_data_needed:
        print(f"ERROR: Dataset has only {len(df)} rows after NA drop. Need at least {min_data_needed} for this configuration.")
        return None, None, None, None, None

    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    stratify_y = y if y.nunique() > 1 else None
    
    X_pool, X_test_fixed, y_pool, y_test_fixed = train_test_split(
        X, y, test_size=FIXED_TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_y
    )
    print(f"SIM_FED: X_pool shape: {X_pool.shape}, X_test_fixed shape: {X_test_fixed.shape}")
    return X_pool, y_pool, X_test_fixed, y_test_fixed, feature_names

def train_initial_client_models(X_pool, y_pool):
    client_models = [] # Returns a LIST of models
    
    num_features = len(X_pool.columns) if not X_pool.empty else 10 # Fallback if X_pool is empty
    columns_for_dummy = X_pool.columns if not X_pool.empty else [f'f{i}' for i in range(num_features)]
    dummy_X_fallback = pd.DataFrame(np.random.rand(2, num_features), columns=columns_for_dummy)
    dummy_y_fallback = pd.Series([0, 1])


    if X_pool.empty or len(X_pool) < NUM_CLIENTS:
        print(f"WARNING: Pool data ({len(X_pool)} samples) is insufficient. Using dummy models for all clients.")
        for _ in range(NUM_CLIENTS):
            model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, max_iter=100)
            model.fit(dummy_X_fallback, dummy_y_fallback) 
            client_models.append(model)
        return client_models

    shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(len(X_pool))
    X_pool_shuffled = X_pool.iloc[shuffled_indices].reset_index(drop=True)
    y_pool_shuffled = y_pool.iloc[shuffled_indices].reset_index(drop=True)

    split_size = max(1, len(X_pool_shuffled) // NUM_CLIENTS) 
    
    for i in range(NUM_CLIENTS):
        model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=0.1, max_iter=100)
        start = i * split_size
        end = (i + 1) * split_size if i < NUM_CLIENTS - 1 else len(X_pool_shuffled)
        
        client_X_train = X_pool_shuffled.iloc[start:end]
        client_y_train = y_pool_shuffled.iloc[start:end]

        if not client_X_train.empty and client_y_train.nunique() >= 2:
            model.fit(client_X_train, client_y_train)
        elif not client_X_train.empty and client_y_train.nunique() == 1:
            print(f"WARNING: Client {i+1} has only one class ({client_y_train.iloc[0]}). Augmenting for LR training.")
            other_class = 0 if client_y_train.iloc[0] == 1 else 1
            temp_dummy_X_sample = pd.DataFrame([dummy_X_fallback.iloc[0].values], columns=client_X_train.columns)
            temp_X = pd.concat([client_X_train, temp_dummy_X_sample], ignore_index=True)
            temp_y = pd.concat([client_y_train, pd.Series([other_class])], ignore_index=True)
            model.fit(temp_X, temp_y)
        else: 
            print(f"WARNING: Client {i+1} has insufficient/empty data. Training on minimal dummy data.")
            model.fit(dummy_X_fallback, dummy_y_fallback)
            
        client_models.append(model)
    return client_models

def get_client_predictions_proba(model, X_data):
    if X_data.empty:
        print("WARN: get_client_predictions_proba received empty X_data.")
        return np.array([0.5]) 
    try:
        return model.predict_proba(X_data)[:, 1] 
    except ValueError as e:
        print(f"ERROR during predict_proba: {e}. X_data columns: {X_data.columns.tolist()}")
        return np.full(len(X_data), 0.5)
    except Exception as e:
        print(f"UNEXPECTED ERROR during predict_proba: {e}")
        return np.full(len(X_data), 0.5)

def poison_predictions_simple_flip(predictions_proba):
    return 1.0 - predictions_proba

if __name__ == '__main__':
    print("--- Running federated_simulation.py directly for testing ---")
    load_results = load_and_prep_data()
    if load_results and load_results[0] is not None:
        X_p, y_p, X_t_f, y_t_f, f_names = load_results
        print(f"Data loaded: X_pool shape {X_p.shape}, X_test_fixed shape {X_t_f.shape}")
        models = train_initial_client_models(X_p, y_p)
        print(f"Trained {len(models)} client models.")
        if models and not X_t_f.empty:
            sample_patient_data_df = X_t_f.iloc[[0]]
            print(f"Test patient data columns: {sample_patient_data_df.columns.tolist()}")
            for i, model_instance in enumerate(models):
                try:
                    pred = get_client_predictions_proba(model_instance, sample_patient_data_df)
                    print(f"Client {i+1} prediction for sample: {pred}")
                except Exception as e:
                    print(f"Error predicting with client {i+1}: {e}")
        elif X_t_f.empty:
            print("Test data (X_t_f) is empty, cannot run prediction test.")
    else:
        print("Data loading failed or returned None components during script test.")
    print("--- End of federated_simulation.py direct test ---")