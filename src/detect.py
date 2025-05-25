# fl_dashboard/src/detect.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import argparse
import os

# Import utility functions from utils.py
try:
    from utils import load_and_preprocess_data, split_data_into_client_shards, RANDOM_STATE
except ImportError:
    print("Error: Could not import from utils.py. Make sure it's in the src/ directory and has no errors.")
    exit()


NUM_CLIENTS = 5
DATASET_FILENAME = "framingham.csv" # Ensure this matches your dataset
DATA_DIR = "data"

def get_client_predictions(X_train_shards, y_train_shards, X_test_global, poisoned_client_index=4):
    """
    Trains local models (one poisoned) and gets their predictions on X_test_global.
    Returns a list of prediction probability arrays and the list of trained models.
    """
    local_models = []
    client_predictions_on_test = [] 

    print("Training client models for detection phase...")
    for i in range(NUM_CLIENTS):
        X_client_train = X_train_shards[i]
        y_client_train = y_train_shards[i].copy()

        # Initialize local_model with a dummy in case of insufficient data early on
        class DummyModel:
            def fit(self, X,y): pass
            def predict_proba(self, X): return np.full((len(X), 2), 0.5) if not X.empty else np.array([])
        
        local_model = DummyModel() # Default to dummy

        if X_client_train.empty or y_client_train.empty or len(X_client_train) < 2:
            print(f"Warning: Client {i+1} has insufficient training data ({len(X_client_train)} samples). Using dummy predictor.")
        else:
            if i == poisoned_client_index:
                print(f"Poisoning Client {i+1} for detection phase: Flipping labels.")
                y_client_train = 1 - y_client_train
                if y_client_train.nunique() < 2:
                     # CORRECTED PRINT STATEMENT:
                     print(f"Warning: Client {i+1} (poisoned) has only one class ({y_client_train.unique()}) after label flip.")
            
            if y_client_train.nunique() < 2:
                print(f"Client {i+1} training data has only one class ({y_client_train.unique()}). Using dummy predictor.")
                # DummyModel already initialized
            else:
                local_model = RandomForestClassifier(
                    n_estimators=50, random_state=RANDOM_STATE, max_depth=10,
                    min_samples_split=5, min_samples_leaf=2,
                    class_weight='balanced' if y_client_train.value_counts(normalize=True).min() < 0.4 else None,
                    n_jobs=-1
                )
                try:
                    local_model.fit(X_client_train, y_client_train)
                except ValueError as e:
                    print(f"Error training model for client {i+1}: {e}. Using dummy predictor.")
                    local_model = DummyModel() # Re-assign dummy if fit fails

        local_models.append(local_model)
        
        if not X_test_global.empty:
            try:
                # Handle cases where dummy model might return empty if X_test_global is empty, though we check X_test_global
                client_pred_proba_all_classes = local_model.predict_proba(X_test_global)
                if client_pred_proba_all_classes.shape[1] > 1: # Ensure it's 2 classes
                    client_pred_proba = client_pred_proba_all_classes[:, 1]
                else: # Fallback if only one class predicted (shouldn't happen with proper dummy)
                    client_pred_proba = np.full(len(X_test_global), 0.5)
                client_predictions_on_test.append(client_pred_proba)
            except AttributeError: # For dummy model if predict_proba is not well-defined for some reason
                 client_predictions_on_test.append(np.full(len(X_test_global), 0.5))
            except Exception as e:
                 print(f"Unexpected error during predict_proba for client {i+1}: {e}")
                 client_predictions_on_test.append(np.full(len(X_test_global), 0.5))

        else: # X_test_global is empty
            client_predictions_on_test.append(np.array([]))

    return client_predictions_on_test, local_models


def detect_malicious_clients_by_mse(client_predictions_on_test, y_test_global_labels, poisoned_client_index=4):
    """
    Calculates MSE for each client's predictions against the global average prediction.
    """
    print("\n--- Phase 4a: Attack Detection using MSE ---")

    if not client_predictions_on_test or not any(p.size > 0 for p in client_predictions_on_test):
        print("Error: No client predictions available for MSE calculation.")
        return None

    # Filter out empty prediction arrays and ensure they match y_test_global_labels length
    valid_predictions = [p for p in client_predictions_on_test if p.size == len(y_test_global_labels)]
    
    if not valid_predictions:
        print("Error: No valid client predictions match the length of the global test set labels for MSE.")
        return None
    
    if len(valid_predictions) < NUM_CLIENTS:
         print(f"Warning: Only {len(valid_predictions)} clients had valid predictions for MSE. Results based on these.")


    global_avg_proba = np.mean(np.array(valid_predictions), axis=0)
    if global_avg_proba.size == 0: # Should not happen if valid_predictions is not empty
        print("Error: Global average probability is empty.")
        return None

    mse_scores_data = []
    print("\nClient MSE vs. Global Average Prediction Probabilities:")
    
    # Determine which original clients correspond to the valid_predictions
    # This is a bit simplified; a more robust way would be to pass client IDs along with predictions
    original_indices_of_valid_preds = [i for i, p in enumerate(client_predictions_on_test) if p.size == len(y_test_global_labels)]


    for idx, client_proba_valid in enumerate(valid_predictions):
        original_client_idx = original_indices_of_valid_preds[idx] # Get original client index
        
        mse = mean_squared_error(global_avg_proba, client_proba_valid)
        client_name = f"Client {original_client_idx + 1}"
        is_attacker = (original_client_idx == poisoned_client_index)
        
        if is_attacker:
            client_name += " (Attacker)"
        
        print(f"{client_name}: {mse:.4f}")
        mse_scores_data.append({"Client": client_name, "MSE_Value": mse, 
                                "Flag": "Attacker (High MSE Expected)" if is_attacker else "Normal"})
        
    mse_df = pd.DataFrame(mse_scores_data)
    if not mse_df.empty:
        mse_df = mse_df.sort_values(by="MSE_Value", ascending=False)
    
    print("\n MSE Table (Detection Results):")
    print(mse_df)
    
    return mse_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect malicious clients in FL.")
    parser.add_argument("--client", type=int, default=5, help="Client to simulate as poisoned (1-indexed).")
    args = parser.parse_args()

    poisoned_client_idx_arg = args.client - 1
    if not (0 <= poisoned_client_idx_arg < NUM_CLIENTS):
        print(f"Error: Invalid client index. Must be between 1 and {NUM_CLIENTS}.")
        exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file_path = os.path.join(project_root, DATA_DIR, DATASET_FILENAME)
    
    print(f"Loading data from: {data_file_path}")
    load_result = load_and_preprocess_data(file_path=data_file_path)

    if load_result is None or load_result[0] is None:
        print("Exiting due to data loading/preprocessing failure.")
        exit()
    X, y, _ = load_result
    
    X_train_shards, y_train_shards, X_test_global, y_test_global = \
        split_data_into_client_shards(X, y, num_clients=NUM_CLIENTS)

    if X_test_global.empty or y_test_global.empty:
        print("Global test set is empty. Cannot run detection.")
        exit()

    client_probas_list, _ = get_client_predictions( # _ ignores the returned models list
        X_train_shards, y_train_shards, X_test_global, poisoned_client_index=poisoned_client_idx_arg
    )
    
    if client_probas_list:
        detect_malicious_clients_by_mse(client_probas_list, y_test_global, poisoned_client_index=poisoned_client_idx_arg)
    else:
        print("Could not get client predictions for MSE detection.")