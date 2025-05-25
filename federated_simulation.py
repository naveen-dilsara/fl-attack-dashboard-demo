# federated_simulation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

def run_full_simulation_pipeline():
    # --- Configuration ---
    NUM_CLIENTS = 5
    POISONED_CLIENT_INDEX = 4 # Client 5 is at index 4 (0-indexed)
    TEST_SIZE_GLOBAL = 0.2
    RANDOM_STATE = 42

    # --- Step 1: Load and Prepare Dataset ---
    try:
        # Ensure "Framingham.csv" is in the same directory as this script
        df = pd.read_csv("Framingham.csv")
    except FileNotFoundError:
        print("ERROR in federated_simulation.py: Framingham.csv not found.")
        print("Please make sure 'Framingham.csv' is in the same folder as the script.")
        return None # Indicate failure
    except Exception as e:
        print(f"ERROR in federated_simulation.py: Could not load Framingham.csv. Reason: {e}")
        return None


    # Basic preprocessing: drop rows with any NaN values
    # A more robust solution would involve imputation, but dropna is simpler for this example.
    # print(f"SIM: Dataset shape before dropna: {df.shape}")
    df.dropna(inplace=True)
    # print(f"SIM: Dataset shape after dropna: {df.shape}")

    if "TenYearCHD" not in df.columns:
        print("ERROR in federated_simulation.py: 'TenYearCHD' column not found in the dataset.")
        return None
    if len(df) < (NUM_CLIENTS * 10): # Basic check for enough data
        print(f"WARNING in federated_simulation.py: Dataset has only {len(df)} rows after dropping NA. This might be too small for {NUM_CLIENTS} clients.")
        if len(df) == 0:
            print("ERROR: Dataset is empty after dropping NA values. Cannot proceed.")
            return None


    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]

    # --- Step 2: Baseline Accuracy (Centralized Clean Model) ---
    # This acts as an approximate upper bound or reference.
    X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
        X, y, test_size=TEST_SIZE_GLOBAL, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
    )
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    if len(X_train_baseline) == 0:
        print("ERROR: Baseline training set is empty. Check data splitting or original data size.")
        return None
    baseline_model.fit(X_train_baseline, y_train_baseline)
    y_pred_baseline = baseline_model.predict(X_test_baseline)
    initial_accuracy_val = accuracy_score(y_test_baseline, y_pred_baseline)
    # print(f"SIM: Baseline Accuracy (Centralized Clean): {initial_accuracy_val:.4f}")


    # --- Step 3: Split Data for Federated Learning ---
    # Using the full dataset (after NA drop) for FL simulation, then split into FL train/test
    X_train_fl_full, X_test_global, y_train_fl_full, y_test_global = train_test_split(
        X, y, test_size=TEST_SIZE_GLOBAL, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
    )

    client_data_train = []
    client_labels_train = []

    # Ensure there's enough data for each client after splitting
    if len(X_train_fl_full) < NUM_CLIENTS:
        print(f"ERROR: Not enough data in X_train_fl_full ({len(X_train_fl_full)} samples) to distribute among {NUM_CLIENTS} clients.")
        return None

    # Distribute X_train_fl_full and y_train_fl_full among clients
    # This uses a simple non-IID split for demonstration. A real FL setup would have naturally partitioned data.
    # Here, we're simulating how data might be distributed.
    
    # Shuffle data before splitting to make it more IID-like for this simulation
    shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(len(X_train_fl_full))
    X_train_fl_shuffled = X_train_fl_full.iloc[shuffled_indices]
    y_train_fl_shuffled = y_train_fl_full.iloc[shuffled_indices]

    split_size = len(X_train_fl_shuffled) // NUM_CLIENTS
    if split_size == 0 and len(X_train_fl_shuffled) > 0 : # If dataset too small, give all to client 1
        print("WARNING: Dataset too small for even splits, giving all training data to client 1 for simulation purposes.")
        client_data_train.append(X_train_fl_shuffled.reset_index(drop=True))
        client_labels_train.append(y_train_fl_shuffled.reset_index(drop=True))
        for i in range(1, NUM_CLIENTS): # Other clients get empty data
             client_data_train.append(pd.DataFrame(columns=X.columns))
             client_labels_train.append(pd.Series(dtype=y.dtype))

    else:
        for i in range(NUM_CLIENTS):
            start = i * split_size
            end = (i + 1) * split_size if i < NUM_CLIENTS - 1 else len(X_train_fl_shuffled)
            if start >= len(X_train_fl_shuffled): # handles edge case if len is not perfectly divisible
                client_data_train.append(pd.DataFrame(columns=X.columns))
                client_labels_train.append(pd.Series(dtype=y.dtype))
            else:
                client_data_train.append(X_train_fl_shuffled.iloc[start:end].reset_index(drop=True))
                client_labels_train.append(y_train_fl_shuffled.iloc[start:end].reset_index(drop=True))


    # --- Step 4: Poison a Client ---
    # Ensure the poisoned client has data to poison
    if not client_data_train[POISONED_CLIENT_INDEX].empty:
        poisoned_client_training_data = client_data_train[POISONED_CLIENT_INDEX].copy()
        poisoned_client_training_labels = 1 - client_labels_train[POISONED_CLIENT_INDEX] # Flip labels
    else:
        print(f"WARNING: Client {POISONED_CLIENT_INDEX + 1} has no data, so cannot be poisoned.")
        poisoned_client_training_data = client_data_train[POISONED_CLIENT_INDEX].copy() # will be empty
        poisoned_client_training_labels = client_labels_train[POISONED_CLIENT_INDEX].copy() # will be empty


    # --- Step 5: Train Local Models ---
    local_models = []
    for i in range(NUM_CLIENTS):
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        # Only train if client has data
        if not client_data_train[i].empty:
            if i == POISONED_CLIENT_INDEX and not poisoned_client_training_data.empty:
                model.fit(poisoned_client_training_data, poisoned_client_training_labels)
            else:
                model.fit(client_data_train[i], client_labels_train[i])
        else:
             # If client has no data, we can't train a model.
             # For predict_proba, we need a "fitted" model.
             # So, fit on a tiny dummy dataset (this is a simulation hack)
             # or ensure all clients have at least some data.
             # Here, we'll make it predict a default probability if it has no data.
             # A better way for a real FL sim would be to ensure all clients get some data
             # or handle clients that don't participate in a round.
             # For this dashboard, we need predict_proba to work for all local_models.
             # Simplest hack: train on a single sample from the overall training data if client is empty.
             if not X_train_fl_full.empty:
                 model.fit(X_train_fl_full.iloc[[0]], y_train_fl_full.iloc[[0]])
             else:
                 # This is a very bad fallback, means dataset is extremely small.
                 # Dashboard will likely show 0 for everything.
                 print("CRITICAL WARNING: Training FL model on dummy data due to empty client set and empty global FL train set.")
                 dummy_X = pd.DataFrame(np.random.rand(1, len(X.columns)), columns=X.columns)
                 dummy_y = pd.Series([0])
                 model.fit(dummy_X, dummy_y)


        local_models.append(model)

    # --- Step 6: Aggregate Predictions (Mean) - Simulating Attack Impact ---
    all_client_probas_on_global_test = np.array(
        [model.predict_proba(X_test_global)[:, 1] for model in local_models]
    )
    global_avg_proba_attacked = all_client_probas_on_global_test.mean(axis=0)
    global_preds_attacked_mean_agg = (global_avg_proba_attacked >= 0.5).astype(int)
    attacked_accuracy_mean_agg_val = accuracy_score(y_test_global, global_preds_attacked_mean_agg)
    # print(f"SIM: Attacked Accuracy (Mean Aggregation): {attacked_accuracy_mean_agg_val:.4f}")

    # --- Step 7: MSE Calculation for Detection ---
    client_mse_values_dict = {}
    for i, client_proba in enumerate(all_client_probas_on_global_test):
        mse = mean_squared_error(global_avg_proba_attacked, client_proba)
        client_name = f"Client {i+1}"
        if i == POISONED_CLIENT_INDEX:
            client_name += " (Attacker)"
        client_mse_values_dict[client_name] = mse
    # print(f"SIM: Client MSEs: {client_mse_values_dict}")

    # --- Step 8: Defense (Median Aggregation) ---
    global_median_proba_defended = np.median(all_client_probas_on_global_test, axis=0)
    global_preds_defended_median_agg = (global_median_proba_defended >= 0.5).astype(int)
    defended_accuracy_median_agg_val = accuracy_score(y_test_global, global_preds_defended_median_agg)
    # print(f"SIM: Defended Accuracy (Median Aggregation): {defended_accuracy_median_agg_val:.4f}")

    return (
        initial_accuracy_val,
        attacked_accuracy_mean_agg_val,
        client_mse_values_dict,
        defended_accuracy_median_agg_val,
        X.columns.tolist() # For feature names
    )

# This part allows you to test the script directly by running: python federated_simulation.py
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
        # print(f"Features: {features}")
    else:
        print("Simulation script test failed to produce results.")