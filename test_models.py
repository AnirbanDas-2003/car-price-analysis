import joblib
import pandas as pd
import numpy as np

def test_prediction():
    """Test the trained models with sample data"""
    
    # Load models and data
    models = joblib.load('car_price_models.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    sample_data = joblib.load('sample_data.pkl')
    
    print("✅ Models loaded successfully!")
    print(f"Available models: {list(models.keys())}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Test with sample car data
    test_cases = [
        {
            'brand': 'Maruti',
            'year': 2020,
            'fuel_type': 'Petrol',
            'transmission_type': 'Manual',
            'seller_type': 'Individual',
            'km_driven': 30000,
            'engine': 1197,
            'max_power': 82.0,
            'mileage': 18.9,
            'seats': 5
        },
        {
            'brand': 'Hyundai',
            'year': 2019,
            'fuel_type': 'Diesel',
            'transmission_type': 'Manual',
            'seller_type': 'Dealer',
            'km_driven': 45000,
            'engine': 1582,
            'max_power': 126.0,
            'mileage': 22.3,
            'seats': 5
        },
        {
            'brand': 'BMW',
            'year': 2018,
            'fuel_type': 'Petrol',
            'transmission_type': 'Automatic',
            'seller_type': 'Dealer',
            'km_driven': 25000,
            'engine': 1998,
            'max_power': 184.0,
            'mileage': 14.8,
            'seats': 5
        }
    ]
    
    print("\n🔍 Testing predictions:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['brand']} {test_case['year']} ({test_case['fuel_type']})")
        
        # Create input features
        vehicle_age = 2025 - test_case['year']
        price_per_km = 1.0
        power_to_engine_ratio = test_case['max_power'] / test_case['engine']
        
        # Create base features
        input_data = {
            'vehicle_age': vehicle_age,
            'km_driven': test_case['km_driven'],
            'mileage': test_case['mileage'],
            'engine': test_case['engine'],
            'max_power': test_case['max_power'],
            'seats': test_case['seats'],
            'year': test_case['year'],
            'price_per_km': price_per_km,
            'power_to_engine_ratio': power_to_engine_ratio
        }
        
        # Add all categorical features with default 0
        for col in feature_columns:
            if col not in input_data:
                input_data[col] = 0
        
        # Set the selected categorical features to 1
        brand_col = f"brand_{test_case['brand']}"
        fuel_col = f"fuel_type_{test_case['fuel_type']}"
        transmission_col = f"transmission_type_{test_case['transmission_type']}"
        seller_col = f"seller_type_{test_case['seller_type']}"
        
        if brand_col in input_data:
            input_data[brand_col] = 1
        if fuel_col in input_data:
            input_data[fuel_col] = 1
        if transmission_col in input_data:
            input_data[transmission_col] = 1
        if seller_col in input_data:
            input_data[seller_col] = 1
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Make predictions
        xgb_pred = models['xgboost'].predict(input_df)[0]
        
        input_scaled = models['scaler'].transform(input_df)
        ridge_pred = models['ridge'].predict(input_scaled)[0]
        
        print(f"  XGBoost Prediction: ₹{xgb_pred:,.0f}")
        print(f"  Ridge Prediction: ₹{ridge_pred:,.0f}")
        print(f"  Difference: ₹{abs(xgb_pred - ridge_pred):,.0f}")
    
    print("\n✅ All tests completed successfully!")
    print("\nYou can now run the Streamlit app with:")
    print("streamlit run streamlit_app.py")


# === New: Evaluate and plot predictions on a larger test set ===
def evaluate_on_test_set(sample_size=100):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load models and data
    models = joblib.load('car_price_models.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    # Load and preprocess the dataset (reuse logic from train_model.py)
    df = pd.read_csv('cardekho_dataset.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df['year'] = 2025 - df['vehicle_age']
    df['price_per_km'] = df['selling_price'] / (df['km_driven'] + 1)
    df['power_to_engine_ratio'] = df['max_power'] / df['engine']
    if 'car_name' in df.columns:
        df = df.drop(['car_name'], axis=1)

    # One-hot encode categorical variables
    categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    # Align columns to match training features
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns + ['selling_price']]

    # Sample from the test set
    df_sample = df_encoded.sample(n=min(sample_size, len(df_encoded)), random_state=42)
    X = df_sample.drop('selling_price', axis=1)
    y_true = df_sample['selling_price']

    # XGBoost predictions
    xgb_pred = models['xgboost'].predict(X)
    # Ridge predictions (scale features)
    X_scaled = models['scaler'].transform(X)
    ridge_pred = models['ridge'].predict(X_scaled)

    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, xgb_pred, alpha=0.6, label='XGBoost', color='royalblue')
    plt.scatter(y_true, ridge_pred, alpha=0.6, label='Ridge', color='orange')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()

    # Plot prediction errors
    plt.subplot(1, 2, 2)
    xgb_err = xgb_pred - y_true
    ridge_err = ridge_pred - y_true
    sns.histplot(xgb_err, color='royalblue', label='XGBoost', kde=True, stat='density', alpha=0.5)
    sns.histplot(ridge_err, color='orange', label='Ridge', kde=True, stat='density', alpha=0.5)
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.title('Prediction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nXGBoost Mean Absolute Error:", np.mean(np.abs(xgb_err)))
    print("Ridge Mean Absolute Error:", np.mean(np.abs(ridge_err)))

if __name__ == "__main__":
    test_prediction()
    print("\n--- Evaluating on a larger test set and plotting errors ---\n")
    evaluate_on_test_set(sample_size=100)
