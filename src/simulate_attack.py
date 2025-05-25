# fl_dashboard/src/simulate_attack.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # CHANGED MODEL
from sklearn.metrics import accuracy_score
import argparse
import os

from utils import load_and_preprocess_data, split_data_into_client_shards, RANDOM_STATE

DATA_DIR = "data"
NUM_CLIENTS = 5
DATASET_FILENAME = "framingham.csv" # Ensure this matches your 100k dataset file if different

def simulate_label_flip_attack(X_train_shards, y_train_shards, X_test_global, y_test_global, poisoned_client_index=4):
    print(f"\n--- Phase 3: Attack Simulation (Label Flipping on Client {poisoned_client_index + 1}) ---")
    print(f"Using RandomForestClassifier for client models.")

    local_models = []
    client_predictions_on_test = []

    for i in range(NUM_CLIENTS):
        X_client_train = X_train_shards[i]
        y_client_train = y_train_shards[i].copy()

        if X_client_train.empty or y_client_train.empty or len(X_client_train) < 2 : # Need at least 2 samples
            print(f"Warning: Client {i+1} has insufficient training data ({len(X_client_train)} samples). Skipping its model training.")
            continue 

        if i == poisoned_client_index:
            print(f"Poisoning Client {i+1}: Flipping labels.")
            y_client_train = 1 - y_client_train
            if y_client_train.nunique() < 2:
                 print(f"Warning: Client {i+1} (poisoned) has only one class after label flip. Model might be trivial.")

        if y_client_train.nunique() < 2:
            print(f"Warning: Client {i+1} training data has only one class ({y_client_train.unique()}). Using a dummy predictor.")
            class DummyModel: # Simple dummy
                def __init__(self, const_val=0.5): self.const_val = const_val
                def fit(self, X,y): pass
                def predict_proba(self, X): return np.full((len(X), 2), self.const_val)
            local_model = DummyModel()
        else:
            local_model = RandomForestClassifier(
                n_estimators=50, # Can adjust this; higher might be better but slower
                random_state=RANDOM_STATE,
                max_depth=10,      
                min_samples_split=5,    
                min_samples_leaf=2,     
                class_weight='balanced' if y_client_train.value_counts(normalize=True).min() < 0.4 else None, # Basic balance check
                n_jobs=-1               
            )
            try:
                local_model.fit(X_client_train, y_client_train)
            except ValueError as e:
                print(f"Error training RandomForest for client {i+1}: {e}. Using dummy predictor.")
                class DummyModel:
                    def __init__(self, const_val=0.5): self.const_val = const_val
                    def fit(self, X,y): pass
                    def predict_proba(self, X): return np.full((len(X), 2), self.const_val)
                local_model = DummyModel()
        
        local_models.append(local_model)
        
        if not X_test_global.empty:
            try:
                client_pred_proba = local_model.predict_proba(X_test_global)[:, 1]
                client_predictions_on_test.append(client_pred_proba)
            except AttributeError: 
                 client_predictions_on_test.append(np.full(len(X_test_global), 0.5)) # For dummy model

    if not client_predictions_on_test or X_test_global.empty or not any(len(p) > 0 for p in client_predictions_on_test):
        print("Error: No valid client predictions generated or test set is empty. Cannot calculate attacked accuracy.")
        return None

    # Ensure all prediction arrays are of the same length before averaging
    # This can happen if some clients failed to produce predictions (e.g., dummy model on empty X_test_global)
    # or if X_test_global itself was modified unexpectedly.
    # However, X_test_global should be fixed. The issue is more likely with some client_pred_proba being empty.
    valid_predictions = [p for p in client_predictions_on_test if len(p) == len(y_test_global)]
    if not valid_predictions:
        print("Error: No client predictions match the length of the global test set labels.")
        return None
    
    if len(valid_predictions) < NUM_CLIENTS:
        print(f"Warning: Only {len(valid_predictions)} out of {NUM_CLIENTS} clients provided valid predictions.")


    global_avg_proba = np.mean(np.array(valid_predictions), axis=0)
    global_predictions = (global_avg_proba >= 0.5).astype(int)

    if len(y_test_global) != len(global_predictions):
        print(f"Error: Length mismatch between y_test_global ({len(y_test_global)}) and global_predictions ({len(global_predictions)}). Cannot calculate accuracy.")
        return None

    attacked_accuracy = accuracy_score(y_test_global, global_predictions)
    print(f"Attacked Global Model Accuracy (Mean Aggregation with RF clients): {attacked_accuracy:.4f} (approx {(attacked_accuracy*100):.0f}%)")
    
    return attacked_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate Federated Learning Attacks.")
    parser.add_argument("--attack", type=str, default="label_flip", choices=["label_flip"],
                        help="Type of attack to simulate.")
    parser.add_argument("--client", type=int, default=5,
                        help="Client to poison (1-indexed).")
    args = parser.parse_args()

    poisoned_client_idx = args.client - 1 
    if not (0 <= poisoned_client_idx < NUM_CLIENTS):
        print(f"Error: Invalid client index {args.client}. Must be between 1 and {NUM_CLIENTS}.")
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

    if not X_test_global.empty and not y_test_global.empty:
        if args.attack == "label_flip":
            simulate_label_flip_attack(
                X_train_shards, y_train_shards,
                X_test_global, y_test_global,
                poisoned_client_index=poisoned_client_idx
            )
        else:
            print(f"Attack type '{args.attack}' not implemented.")
    else:
        print("Global test set is empty. Cannot run attack simulation.")