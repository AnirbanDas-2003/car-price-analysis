import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the CarDekho dataset"""
    # Load the dataset
    df = pd.read_csv('cardekho_dataset.csv')
    
    # Drop the unnamed index column
    df = df.drop('Unnamed: 0', axis=1)
    
    # Basic data cleaning
    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Feature engineering
    # Calculate current year (2025) - vehicle_age to get manufacturing year
    df['year'] = 2025 - df['vehicle_age']
    
    # Create price per km feature
    df['price_per_km'] = df['selling_price'] / (df['km_driven'] + 1)  # +1 to avoid division by zero
    
    # Create power to weight ratio (approximation)
    df['power_to_engine_ratio'] = df['max_power'] / df['engine']
    
    # Drop car_name only (keep model for prediction)
    df = df.drop(['car_name'], axis=1)
    
    return df

def encode_categorical_features(df):
    """Encode categorical features using one-hot encoding"""
    # Categorical columns to encode (include model)
    categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
    # One-hot encode categorical variables (do NOT drop first, so every model/brand/etc. gets a column)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    return df_encoded

def train_models(X_train, X_test, y_train, y_test):
    """Train XGBoost and Ridge Regression models"""
    
    models = {}
    results = {}
    
    # XGBoost Model (improved hyperparameters and early stopping)
    print("Training XGBoost model with improved hyperparameters and early stopping...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        verbosity=1
    )
    # Convert to numpy arrays for XGBoost early stopping compatibility
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    xgb_model.fit(
        X_train_np, y_train_np
    )
    xgb_pred = xgb_model.predict(X_test_np)
    models['xgboost'] = xgb_model
    results['xgboost'] = {
        'mse': mean_squared_error(y_test_np, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_np, xgb_pred)),
        'mae': mean_absolute_error(y_test_np, xgb_pred),
        'r2': r2_score(y_test_np, xgb_pred)
    }
    
    # Ridge Regression Model
    print("Training Ridge Regression model...")
    
    # Scale features for Ridge Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred = ridge_model.predict(X_test_scaled)
    
    models['ridge'] = ridge_model
    models['scaler'] = scaler
    results['ridge'] = {
        'mse': mean_squared_error(y_test, ridge_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
        'mae': mean_absolute_error(y_test, ridge_pred),
        'r2': r2_score(y_test, ridge_pred)
    }
    
    return models, results

def main():
    """Main function to run the complete pipeline"""
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Separate features and target
    X = df_encoded.drop('selling_price', axis=1)
    y = df_encoded['selling_price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Feature columns:", X.columns.tolist())
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"  RMSE: ₹{metrics['rmse']:,.2f}")
        print(f"  MAE: ₹{metrics['mae']:,.2f}")
        print(f"  R² Score: {metrics['r2']:.4f}")
    
    # Save models and preprocessing info
    joblib.dump(models, 'car_price_models.pkl')
    joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
    
    # Save sample data for the app
    sample_data = {
        'brands': sorted(df['brand'].unique()),
        'fuel_types': sorted(df['fuel_type'].unique()),
        'transmission_types': sorted(df['transmission_type'].unique()),
        'seller_types': sorted(df['seller_type'].unique()),
        'year_range': (int(df['year'].min()), int(df['year'].max())),
        'km_range': (int(df['km_driven'].min()), int(df['km_driven'].max())),
        'engine_range': (int(df['engine'].min()), int(df['engine'].max())),
        'power_range': (float(df['max_power'].min()), float(df['max_power'].max())),
        'mileage_range': (float(df['mileage'].min()), float(df['mileage'].max())),
        'seats_range': (int(df['seats'].min()), int(df['seats'].max()))
    }
    
    joblib.dump(sample_data, 'sample_data.pkl')
    
    print("\n" + "="*50)
    print("Models and preprocessing data saved successfully!")
    print("Files created:")
    print("- car_price_models.pkl")
    print("- feature_columns.pkl")
    print("- sample_data.pkl")
    print("="*50)

if __name__ == "__main__":
    main()
