import pandas as pd
import data_loader # Your script to load data
import data_preprocessor # Your script with the DataPreprocessor class
import evaluation_metrics as my_metrics # Import your new functional metrics module

# Import necessary sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Keep these sklearn metrics for convenient formatted output
from sklearn.metrics import confusion_matrix, classification_report

print("--- Running Main Pipeline Script (Simplified: Logistic Regression Only) ---")

# --- Initialization ---
df_real = None
X = None
y = None
X_processed = None
X_train, X_test, y_train, y_test = None, None, None, None
log_reg_model = None
data_ok = True # Flag to track if steps are successful

# --- Workflow ---

# 1. Load the Real Data
print("Loading real data using data_loader...")
df_real = data_loader.load_heart_data()

if df_real is None:
    print("Halting script: Failed to load data.")
    data_ok = False
else:
    print(f"Real data loaded successfully. Shape: {df_real.shape}")

    # 2. Ensure 'target_binary' exists (Create if necessary)
    if 'target_binary' not in df_real.columns:
        print("Creating 'target_binary' column...")
        if 'target' in df_real.columns:
            df_real['target_binary'] = df_real['target'].apply(lambda x: 0 if x == 0 else 1)
            print("'target_binary' created.")
        else:
            print("ERROR: Original 'target' column not found in loaded data.")
            data_ok = False # Stop processing

# Proceed only if data loading and target creation were okay
if data_ok:
    # 3. Separate Features (X) and Target (y)
    print("Separating features (X) and target (y='target_binary')...")
    try:
        if 'target_binary' in df_real.columns:
            y = df_real['target_binary']
            # Drop original target and binary target to get features X
            X = df_real.drop(['target', 'target_binary'], axis=1)
            print(f"Features X shape: {X.shape}")
            print(f"Target y shape: {y.shape}")
        else:
            print("ERROR: 'target_binary' column not found or created. Cannot separate.")
            data_ok = False
    except KeyError as e:
        print(f"Error separating features/target. Missing column: {e}")
        data_ok = False

# Proceed only if separation was okay
if data_ok and X is not None and y is not None:
    # 4. Initialize and Apply Preprocessor
    print("Initializing DataPreprocessor...")
    # Create an instance of the class from data_preprocessor.py
    preprocessor = data_preprocessor.DataPreprocessor()

    # Fit the preprocessor and transform the features X
    # This applies imputation, scaling, and encoding
    print("Applying preprocessing (fit_transform) to real data X...")
    try:
        X_processed = preprocessor.fit_transform(X)

        # 5. Verify Processed Data
        print("\n--- Preprocessing Complete ---")
        print(f"Shape of processed features (X_processed): {X_processed.shape}")
        # print("First 5 rows of processed data (X_processed):") # Optional: Uncomment to view
        # print(X_processed[:5, :])

        # 6. Train/Test Split
        print("\n--- Splitting data into Train/Test sets ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.20, random_state=42, stratify=y
        )
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        # 7. Train Logistic Regression Model
        print("\n--- Training Logistic Regression model ---")
        log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
        log_reg_model.fit(X_train, y_train)
        print("Logistic Regression model trained successfully.")

        # 8. Evaluate the Logistic Regression Model
        print("\n--- Evaluating Logistic Regression model on Test Set ---")
        y_pred_lr = log_reg_model.predict(X_test)

        # --- Calculate Performance Metrics using custom functions ---
        accuracy_lr = my_metrics.calculate_accuracy(y_test, y_pred_lr)
        precision_lr = my_metrics.calculate_precision(y_test, y_pred_lr)
        recall_lr = my_metrics.calculate_recall(y_test, y_pred_lr)
        f1_lr = my_metrics.calculate_f1(y_test, y_pred_lr)

        print("--- METRICS (Logistic Regression - Functional Calculation) ---")
        print(f"Accuracy:  {accuracy_lr:.4f}")
        print(f"Precision: {precision_lr:.4f}")
        print(f"Recall:    {recall_lr:.4f}")
        print(f"F1-score:  {f1_lr:.4f}")

        # --- Use sklearn for formatted reports ---
        print("\nConfusion Matrix (Logistic Regression):")
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        print(cm_lr)
        print("\nClassification Report (Logistic Regression):")
        print(classification_report(y_test, y_pred_lr, target_names=['No Disease (0)', 'Disease (1)'], zero_division=0))
        print("\n--- Evaluation Complete for Logistic Regression ---")

        # --- Random Forest Section Removed for Simplification ---

    except Exception as e:
         print(f"\nAn error occurred during preprocessing, training, or evaluation: {e}")
         data_ok = False # Indicate an error occurred

# Check if any step failed along the way
if not data_ok:
     print("\nScript halted or did not complete fully due to errors.")


print("\n--- End of Main Pipeline Script ---")