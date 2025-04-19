import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    A class to preprocess the heart disease dataset.
    Handles missing values, scales numerical features, and encodes categorical features.
    """
    def __init__(self):
        # Define feature lists based on EDA
        # Numerical features to be scaled
        self.numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        # Categorical features to be imputed (if needed) and one-hot encoded
        # Note: 'ca' and 'thal' are included here as they are categorical representations
        # even though they appear numeric sometimes. They also have missing values.
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        # --- Build the Preprocessing Pipelines ---

        # Pipeline for numerical features: Just scale them
        # (No imputation needed here as EDA showed no NaNs in these specific columns)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Pipeline for categorical features:
        # 1. Impute missing values (using the most frequent value)
        # 2. One-Hot Encode the categories
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Handles NaN in 'ca', 'thal'
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # Converts categories to 0/1 columns
        ])

        # --- Assemble preprocessing steps using ColumnTransformer ---
        # Apply numeric_transformer to numerical_features
        # Apply categorical_transformer to categorical_features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough' # Optional: Keep other columns if any (shouldn't be any here)
        )
        print("DataPreprocessor initialized.")

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline to the data X.
        Note: y is typically ignored in unsupervised preprocessing steps.
        """
        print("Fitting preprocessor...")
        # We only fit on X (features), not y (target)
        self.preprocessor.fit(X)
        print("Preprocessor fitting complete.")
        return self

    def transform(self, X):
        """
        Transform the data X using the fitted preprocessing pipeline.
        """
        print("Transforming data...")
        # Transform X using the fitted preprocessor
        X_processed = self.preprocessor.transform(X)
        print(f"Data transformed to shape: {X_processed.shape}")
        return X_processed

    def fit_transform(self, X, y=None):
        """
        Fit the pipeline to the data X and then transform X.
        """
        print("Fitting and transforming data...")
        X_processed = self.preprocessor.fit_transform(X, y)
        print(f"Data fitted and transformed to shape: {X_processed.shape}")
        return X_processed

# Example of how to potentially use this class later (in run_pipeline.py)
if __name__ == '__main__':
    # This part is just for demonstration if you run this file directly
    # In the actual project, you'll import and use this class in run_pipeline.py
    print("Demonstrating DataPreprocessor usage (if run directly)")

    # Load sample data (replace with actual loading using data_loader)
    # Create a dummy DataFrame for demonstration structure
    dummy_data = {
        'age': [63, 67, 37], 'sex': [1, 1, 1], 'cp': [1, 4, 3], 'trestbps': [145, 160, 130],
        'chol': [233, 286, 250], 'fbs': [1, 0, 0], 'restecg': [2, 2, 0], 'thalach': [150, 108, 187],
        'exang': [0, 1, 0], 'oldpeak': [2.3, 1.5, 3.5], 'slope': [3, 2, 3],
        'ca': [0, 3, 0], 'thal': [6, 3, 3], 'target': [0, 2, 0], 'target_binary': [0, 1, 0] # Assuming target_binary exists
    }
    dummy_df = pd.DataFrame(dummy_data)
    X_dummy = dummy_df.drop(['target', 'target_binary'], axis=1) # Features only
    y_dummy = dummy_df['target_binary'] # Target only

    print("\nDummy Features (X):")
    print(X_dummy)

    # Instantiate the preprocessor
    preprocessor_instance = DataPreprocessor()

    # Fit and transform the dummy data
    X_processed_dummy = preprocessor_instance.fit_transform(X_dummy)

    print("\nProcessed Dummy Features (X_processed):")
    print(X_processed_dummy)
    # Note: Output will be a NumPy array, likely sparse if OneHotEncoder is used heavily.
    # The number of columns will increase due to one-hot encoding.