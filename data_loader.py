import pandas as pd

def load_heart_data(file_path='./data_base/processed.cleveland.data'):
    """Loads the Cleveland heart disease dataset from the specified path."""

    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        df = pd.read_csv(file_path, header=None, names=column_names, na_values='?')
        print(f"--- Data loaded successfully from {file_path} ---")
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

if __name__ == '__main__':
    # Example of how to use the function if you run this script directly
    data = load_heart_data()
    if data is not None:
        print("\n--- Running data_loader.py directly ---")
        print("First 5 rows:")
        print(data.head())
        print("\nInfo:")
        data.info()
        print("\nMissing Values:")
        print(data.isnull().sum())