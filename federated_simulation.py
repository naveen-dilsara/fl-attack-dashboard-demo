# federated_simulation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import os # <-- ADDED THIS IMPORT

def run_full_simulation_pipeline():
    # --- Configuration ---
    NUM_CLIENTS = 5
    POISONED_CLIENT_INDEX = 4 # Client 5 is at index 4 (0-indexed)
    TEST_SIZE_GLOBAL = 0.2
    RANDOM_STATE = 42

    # --- Step 1: Load and Prepare Dataset ---
    # Construct the absolute path to the CSV file relative to the script's location
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(current_dir, "framingham.csv")
        # print(f"SIM: Attempting to load CSV from: {csv_file_path}") # For debugging
        
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"ERROR in federated_simulation.py: Framingham.csv not found at expected path: {csv_file_path}")
        print("Please make sure 'Framingham.csv' is in the same folder as the script in your GitHub repository.")
        return None # Indicate failure
    except Exception as e:
        print(f"ERROR in federated_simulation.py: Could not load Framingham.csv. Reason: {e}")
        return None

    df.dropna(inplace=True)

    if "TenYearCHD" not in df.columns:
        print("ERROR in federated_simulation.py: 'TenYearCHD' column not found in the dataset.")
        return None
    if len(df) < (NUM_CLIENTS * 2): # Adjusted minimum check
        print(f"WARNING in federated_simulation.py: Dataset has only {len(df)} rows after dropping NA. This might be too small for {NUM_CLIENTS} clients.")
        if len(df) == 0:
            print("ERROR: Dataset is empty after dropping NA values. Cannot proceed.")
            return None

    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]

    # Ensure y has at least two unique classes for stratification if possible
    stratify_y = y if y.nunique() > 1 else None

    # --- Step 2: Baseline Accuracy (Centralized Clean Model) ---
    if len(X) < 2 or len(y) < 2: # Need at least 2 samples for train_test_split
        print("ERROR: Not enough data to perform train_test_split for baseline.")
        return None
        
    X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
        X, y, test_size=TEST_SIZE_GLOBAL, random_state=RANDOM_STATE, stratify=stratify_y
    )
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    if len(X_train_baseline) == 0:
        print("ERROR: Baseline training set is empty. Check data splitting or original data size.")
        return None
    baseline_model.fit(X_train_baseline, y_train_baseline)
    y_pred_baseline = baseline_model.predict(X_test_baseline)
    initial_accuracy_val = accuracy_score(y_test_baseline, y_pred_baseline)

    # --- Step 3: Split Data for Federated Learning ---
    if len(X) < 2 or len(y) < 2:
        print("ERROR: Not enough data to perform train_test_split for FL simulation.")
        return None

    X_train_fl_full, X_test_global, y_train_fl_full, y_test_global = train_test_split(
        X, y, test_size=TEST_SIZE_GLOBAL, random_state=RANDOM_STATE, stratify=stratify_y
    )

    client_data_train = []
    client_labels_train = []

    if len(X_train_fl_full) < NUM_CLIENTS and len(X_train_fl_full) > 0:
        print(f"WARNING: Not enough data in X_train_fl_full ({len(X_train_fl_full)} samples) to distribute ideally among {NUM_CLIENTS} clients. Adjusting split.")
        # Give all data to the first few clients if dataset is smaller than num_clients
        for i in range(NUM_CLIENTS):
            if i < len(X_train_fl_full):
                 # Give one sample per client if possible, else some get empty
                start_idx = i
                end_idx = i + 1
                client_data_train.append(X_train_fl_full.iloc[start_idx:end_idx].reset_index(drop=True))
                client_labels_train.append(y_train_fl_full.iloc[start_idx:end_idx].reset_index(drop=True))
            else:
                client_data_train.append(pd.DataFrame(columns=X.columns))
                client_labels_train.append(pd.Series(dtype=y.dtype))
    elif len(X_train_fl_full) == 0:
        print("ERROR: FL training dataset is empty after split.")
        # Create empty dataframes/series for all clients to avoid errors later
        for i in range(NUM_CLIENTS):
            client_data_train.append(pd.DataFrame(columns=X.columns))
            client_labels_train.append(pd.Series(dtype=y.dtype))
    else:
        shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(len(X_train_fl_full))
        X_train_fl_shuffled = X_train_fl_full.iloc[shuffled_indices]
        y_train_fl_shuffled = y_train_fl_full.iloc[shuffled_indices]
        
        split_size = len(X_train_fl_shuffled) // NUM_CLIENTS
        for i in range(NUM_CLIENTS):
            start = i * split_size
            end = (i + 1) * split_size if i < NUM_CLIENTS - 1 else len(X_train_fl_shuffled)
            client_data_train.append(X_train_fl_shuffled.iloc[start:end].reset_index(drop=True))
            client_labels_train.append(y_train_fl_shuffled.iloc[start:end].reset_index(drop=True))

    # --- Step 4: Poison a Client ---
    if not client_data_train[POISONED_CLIENT_INDEX].empty:
        poisoned_client_training_data = client_data_train[POISONED_CLIENT_INDEX].copy()
        poisoned_client_training_labels = 1 - client_labels_train[POISONED_CLIENT_INDEX]
    else:
        print(f"WARNING: Client {POISONED_CLIENT_INDEX + 1} has no data, cannot be poisoned effectively.")
        poisoned_client_training_data = client_data_train[POISONED_CLIENT_INDEX].copy()
        poisoned_client_training_labels = client_labels_train[POISONED_CLIENT_INDEX].copy()

    # --- Step 5: Train Local Models ---
    local_models = []
    # Fallback data if everything else is empty (should ideally not be needed with proper data checks)
    dummy_X_fallback = pd.DataFrame(np.random.rand(2, len(X.columns)), columns=X.columns) if X.empty else X.iloc[[0]] if len(X) == 1 else X.iloc[:2]
    dummy_y_fallback = pd.Series([0,1]) if y.empty or y.nunique() < 2 else y.iloc[[0]] if len(y) == 1 else y.iloc[:2]
    
    if len(dummy_X_fallback) < 2 and len(y.unique()) > 1: # Ensure dummy_y has at least two classes for RF if possible
        dummy_y_fallback = pd.Series(y.unique()[:2]) if len(y.unique()) >=2 else pd.Series([0,1])
        dummy_X_fallback = pd.DataFrame(np.random.rand(len(dummy_y_fallback), len(X.columns)), columns=X.columns)


    for i in range(NUM_CLIENTS):
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        current_client_data = client_data_train[i]
        current_client_labels = client_labels_train[i]

        if i == POISONED_CLIENT_INDEX and not poisoned_client_training_data.empty:
            current_client_data = poisoned_client_training_data
            current_client_labels = poisoned_client_training_labels
        
        if not current_client_data.empty and current_client_labels.nunique() >= 2:
            model.fit(current_client_data, current_client_labels)
        elif not current_client_data.empty and current_client_labels.nunique() == 1:
            # If only one class, RF might complain. Add a dummy sample of another class (simulation hack)
            # This is a tricky edge case for small client datasets.
            print(f"WARNING: Client {i+1} has only one class. Augmenting slightly for RF training.")
            temp_data = pd.concat([current_client_data, dummy_X_fallback.iloc[[0]]])
            other_class = 0 if current_client_labels.iloc[0] == 1 else 1
            temp_labels = pd.concat([current_client_labels, pd.Series([other_class])])
            model.fit(temp_data, temp_labels)
        else: # Client data is empty
            print(f"WARNING: Client {i+1} training data is empty. Training on minimal dummy data.")
            model.fit(dummy_X_fallback, dummy_y_fallback)
        local_models.append(model)

    # --- Step 6 & 7 & 8 will only work if X_test_global is not empty ---
    if X_test_global.empty:
        print("ERROR: Global test set is empty. Cannot evaluate models.")
        attacked_accuracy_mean_agg_val = 0.0
        client_mse_values_dict = {f"Client {i+1}{' (Attacker)' if i==POISONED_CLIENT_INDEX else ''}": 0.0 for i in range(NUM_CLIENTS)}
        defended_accuracy_median_agg_val = 0.0
    else:
        all_client_probas_on_global_test = np.array(
            [model.predict_proba(X_test_global)[:, 1] for model in local_models]
        )
        global_avg_proba_attacked = all_client_probas_on_global_test.mean(axis=0)
        global_preds_attacked_mean_agg = (global_avg_proba_attacked >= 0.5).astype(int)
        attacked_accuracy_mean_agg_val = accuracy_score(y_test_global, global_preds_attacked_mean_agg)

        client_mse_values_dict = {}
        for i, client_proba in enumerate(all_client_probas_on_global_test):
            mse = mean_squared_error(global_avg_proba_attacked, client_proba)
            client_name = f"Client {i+1}"
            if i == POISONED_CLIENT_INDEX:
                client_name += " (Attacker)"
            client_mse_values_dict[client_name] = mse

        global_median_proba_defended = np.median(all_client_probas_on_global_test, axis=0)
        global_preds_defended_median_agg = (global_median_proba_defended >= 0.5).astype(int)
        defended_accuracy_median_agg_val = accuracy_score(y_test_global, global_preds_defended_median_agg)

    return (
        initial_accuracy_val,
        attacked_accuracy_mean_agg_val,
        client_mse_values_dict,
        defended_accuracy_median_agg_val,
        X.columns.tolist()
    )

if __name__ == '__main__':
    results = run_full_simulation_pipeline()
    if results:
        initial_acc, attacked_acc_mean, client_mses, defended_acc_median, features = results
        print("\n--- Simulation Script Test Output ---")
        print(f"Initial Clean Accuracy (Centralized): {initial_acc:.4f}")
        print(f"Attacked Accuracy (Mean Aggregation): {attacked_acc_mean:.4f}")
        print("Client MSEs:")
        for client, mse in client_mses.items():
            print(f"  {client}: {mse:.4f}")
        print(f"Defended Accuracy (Median Aggregation): {defended_acc_median:.4f}")
    else:
        print("Simulation script test failed to produce results.")