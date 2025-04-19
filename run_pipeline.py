import pandas as pd
import data_loader # Your script to load data
import data_preprocessor # Your script with the DataPreprocessor class

# Import necessary sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

print("--- Running Main Pipeline Script ---")

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
            # X, y remain None if error occurs here
    except KeyError as e:
        print(f"Error separating features/target. Missing column: {e}")
        data_ok = False
        # X, y remain None

# Proceed only if separation was okay
if data_ok:
    # 4. Initialize and Apply Preprocessor
    print("Initializing DataPreprocessor...")
    preprocessor = data_preprocessor.DataPreprocessor()

    print("Applying preprocessing (fit_transform) to real data X...")
    try:
        X_processed = preprocessor.fit_transform(X)

        # 5. Verify Processed Data
        print("\n--- Preprocessing Complete ---")
        print(f"Shape of processed features (X_processed): {X_processed.shape}")
        print("First 5 rows of processed data (X_processed):")
        print(X_processed[:5, :]) # Displaying numpy array

        # 6. Train/Test Split
        print("\n--- Splitting data into Train/Test sets ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.20, random_state=42, stratify=y
        )
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        # 7. Train Initial Model (Logistic Regression)
        print("\n--- Training Logistic Regression model ---")
        log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
        log_reg_model.fit(X_train, y_train)
        print("Logistic Regression model trained successfully.")

        # 8. Evaluate the Model
        print("\n--- Evaluating Logistic Regression model on Test Set ---")
        # Use the trained model to make predictions on the TEST FEATURES (X_test)
        y_pred = log_reg_model.predict(X_test)

        # --- Calculate Performance Metrics ---
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        # --- Confusion Matrix ---
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        # [[TN, FP],
        #  [FN, TP]]

        # --- Classification Report ---
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)'], zero_division=0))

        # --- Evaluation Complete ---
        print("\n--- Evaluation Complete for Logistic Regression ---")
        # (Code for other models or final steps would go here)

    except Exception as e:
         print(f"An error occurred during preprocessing, training, or evaluation: {e}")
         data_ok = False # Indicate an error occurred

if not data_ok: # Check if any step failed along the way
     print("\nScript halted or did not complete fully due to errors.")

print("\n--- End of Main Pipeline Script ---")
# --- Train and Evaluate Random Forest Model ---
print("\n--- Training Random Forest model ---")
from sklearn.ensemble import RandomForestClassifier
 # Instantiate with default parameters (keeping it simple)
rf_model = RandomForestClassifier(random_state=42)

            # Train on the SAME training data
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

print("\n--- Evaluating Random Forest model on Test Set ---")
            # Make predictions
y_pred_rf = rf_model.predict(X_test)

            # Calculate Metrics (reuse or call functional metric functions)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print(f"Accuracy:  {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall:    {recall_rf:.4f}")
print(f"F1-score:  {f1_rf:.4f}")

print("\nConfusion Matrix (Random Forest):")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['No Disease (0)', 'Disease (1)'], zero_division=0))
print("\n--- Evaluation Complete for Random Forest ---")

        # (Outer indentation continues...)