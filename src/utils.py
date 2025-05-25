# fl_dashboard/src/utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Added LabelEncoder

# Define a consistent random state for reproducibility
RANDOM_STATE = 42

def load_and_preprocess_data(file_path="data/framingham.csv"):
    """
    Loads the Framingham dataset, performs basic preprocessing.
    - Handles missing values (mean imputation for numeric, mode for categorical).
    - Encodes categorical features.
    - Scales numerical features.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None, None, None # Return None for X, y, and original_X_columns
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

    print(f"Original dataset shape: {df.shape}")

    # Separate features and target
    if "TenYearCHD" not in df.columns:
        print("Error: Target column 'TenYearCHD' not found in the dataset.")
        return None, None, None
    
    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]
    original_X_columns = X.columns.tolist() # Store original column names

    # Impute missing values
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].mean(), inplace=True)
                print(f"Imputed missing values in numeric column '{col}' with mean ({X[col].mean():.2f}).")
            else: # Assume categorical
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
                print(f"Imputed missing values in categorical column '{col}' with mode ('{mode_val}').")
    
    # Handle missing values in target (if any, though dropna is often preferred for target)
    if y.isnull().any():
        print(f"Warning: Target variable 'TenYearCHD' has {y.isnull().sum()} missing values. Dropping these rows.")
        # Align X and y after dropping NaNs from y
        valid_y_indices = y.dropna().index
        X = X.loc[valid_y_indices]
        y = y.loc[valid_y_indices]
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

    if X.empty or y.empty:
        print("Error: Dataset became empty after preprocessing. Check missing value handling.")
        return None, None, None

    # Identify categorical and numerical features (example - adjust based on your actual dataset)
    # Based on the feature image: 'sex', 'education' are categorical. Binary features are also treated as categorical for encoding.
    categorical_cols = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
    # Ensure all listed categorical columns exist in X, handle missing ones
    actual_categorical_cols = [col for col in categorical_cols if col in X.columns]
    if len(actual_categorical_cols) != len(categorical_cols):
        print(f"Warning: Some defined categorical columns not found in dataset. Found: {actual_categorical_cols}")

    numerical_cols = [col for col in X.columns if col not in actual_categorical_cols]

    # Encode categorical features (Label Encoding for simplicity here, One-Hot might be better for some models)
    # If models like RandomForest are used, they can often handle label encoded categoricals well.
    # For Logistic Regression, One-Hot Encoding is generally preferred for nominal categoricals.
    # For simplicity in this utils script, we'll use Label Encoding for all categoricals identified.
    # The original dataset description shows 'sex' as 1/0, 'currentSmoker' 1/0, etc. 
    # 'education' might be ordinal or nominal. We'll assume it can be label encoded.
    
    label_encoders = {}
    for col in actual_categorical_cols:
        if X[col].dtype == 'object': # Only encode if it's an object type
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le # Store encoder if needed for inverse transform later
            print(f"Label encoded column '{col}'.")
        elif not pd.api.types.is_numeric_dtype(X[col]):
             # If it's not object and not numeric, it might be already encoded or needs specific handling
            print(f"Warning: Column '{col}' is categorical but not of object type. Assuming pre-encoded or can be used as is.")


    # Scale numerical features
    if numerical_cols:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        print(f"Scaled numerical columns: {numerical_cols}")
    
    print(f"Processed dataset shape: X - {X.shape}, y - {y.shape}")
    return X, y, original_X_columns


def split_data_into_client_shards(X, y, num_clients=5, test_size=0.2):
    """
    Splits the data into a global test set and then shards the training data for clients.
    Returns:
        X_train_shards (list of DataFrames): Training features for each client.
        y_train_shards (list of Series): Training labels for each client.
        X_test_global (DataFrame): Global test features.
        y_test_global (Series): Global test labels.
    """
    if X is None or y is None or X.empty or y.empty:
        print("Error in split_data: Input X or y is None or empty.")
        return [pd.DataFrame()] * num_clients, [pd.Series()] * num_clients, pd.DataFrame(), pd.Series()

    stratify_option = y if y.nunique() > 1 else None

    # First, split into a global training set and a global test set
    X_train_full, X_test_global, y_train_full, y_test_global = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_option
    )

    if X_train_full.empty:
        print("Error in split_data: Training set is empty after initial split.")
        return [pd.DataFrame()] * num_clients, [pd.Series()] * num_clients, X_test_global, y_test_global

    X_train_shards = []
    y_train_shards = []
    
    # Shuffle training data before sharding
    shuffled_indices = np.random.RandomState(seed=RANDOM_STATE).permutation(len(X_train_full))
    X_train_shuffled = X_train_full.iloc[shuffled_indices].reset_index(drop=True)
    y_train_shuffled = y_train_full.iloc[shuffled_indices].reset_index(drop=True)

    shard_size = max(1, len(X_train_shuffled) // num_clients) # Ensure at least 1 sample per shard if data allows

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else len(X_train_shuffled)
        if start >= len(X_train_shuffled): # Not enough data for this shard
            X_train_shards.append(pd.DataFrame(columns=X.columns))
            y_train_shards.append(pd.Series(dtype=y.dtype))
        else:
            X_train_shards.append(X_train_shuffled.iloc[start:end])
            y_train_shards.append(y_train_shuffled.iloc[start:end])
        print(f"Client {i+1} train shard shape: {X_train_shards[-1].shape}")

    print(f"Global test set shape: X_test - {X_test_global.shape}, y_test - {y_test_global.shape}")
    return X_train_shards, y_train_shards, X_test_global, y_test_global


# --- Test functions (can be run if this file is executed directly) ---
if __name__ == '__main__':
    print("--- Testing data_utils.py ---")
    
    # Test load_and_preprocess_data
    X_processed, y_processed, _ = load_and_preprocess_data("data/framingham.csv") # Adjust path if utils.py is in src/
    if X_processed is not None and y_processed is not None:
        print("\n--- Data Loading & Preprocessing Test ---")
        print("X_processed head:\n", X_processed.head())
        print("y_processed head:\n", y_processed.head())
        print(f"X_processed shape: {X_processed.shape}, y_processed shape: {y_processed.shape}")

        # Test split_data_into_client_shards
        print("\n--- Client Data Sharding Test ---")
        X_shards, y_shards, X_test, y_test = split_data_into_client_shards(X_processed, y_processed, num_clients=5)
        
        if X_shards and y_shards:
            for i in range(len(X_shards)):
                print(f"Client {i+1} data shape: X_shard - {X_shards[i].shape}, y_shard - {y_shards[i].shape}")
            print(f"Global test data shape: X_test - {X_test.shape}, y_test - {y_test.shape}")
        else:
            print("Client data sharding failed or produced empty shards.")
    else:
        print("Data loading and preprocessing failed.")