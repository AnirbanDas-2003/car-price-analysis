import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
/* Make selectbox text white and background dark for visibility */
.stSelectbox > div > div > div {
    background-color: #222 !important;
    color: #fff !important;
}
/* Dropdown menu background and text color */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #222 !important;
    color: #fff !important;
}
.stSelectbox [data-baseweb="select"] span {
    color: #fff !important;
}
.stSelectbox [data-baseweb="popover"] {
    background-color: #222 !important;
    color: #fff !important;
}
.price-prediction {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
}
.comparison-box {
    background-color: #f8f9fa;
    color: #222;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}
.metric-box {
    background-color: #ffffff;
    color: #222;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models_and_data():
    """Load trained models and preprocessing data"""
    try:
        models = joblib.load('car_price_models.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        sample_data = joblib.load('sample_data.pkl')
        return models, feature_columns, sample_data
    except FileNotFoundError:
        st.error("Model files not found. Please run train_model.py first.")
        return None, None, None

def create_input_features(brand, year, fuel_type, transmission_type, seller_type, 
                         km_driven, engine, max_power, mileage, seats, feature_columns, model=None):
    """Create input features for prediction"""
    
    # Calculate vehicle age and derived features
    vehicle_age = 2025 - year
    price_per_km = 1.0  # This will be calculated after prediction
    power_to_engine_ratio = max_power / engine
    
    # Create base features
    input_data = {
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
        'year': year,
        'price_per_km': price_per_km,
        'power_to_engine_ratio': power_to_engine_ratio
    }
    
    # Add all categorical features with default 0
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0
    
    # Set the selected categorical features to 1
    brand_col = f'brand_{brand}'
    fuel_col = f'fuel_type_{fuel_type}'
    transmission_col = f'transmission_type_{transmission_type}'
    seller_col = f'seller_type_{seller_type}'
    if brand_col in input_data:
        input_data[brand_col] = 1
    if fuel_col in input_data:
        input_data[fuel_col] = 1
    if transmission_col in input_data:
        input_data[transmission_col] = 1
    if seller_col in input_data:
        input_data[seller_col] = 1
    # Set the selected model feature to 1 if it exists
    if model is not None:
        model_col = f'model_{model}'
        if model_col in input_data:
            input_data[model_col] = 1
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    return input_df

def predict_price(input_df, models, model_choice):
    """Predict car price using the selected model"""
    if model_choice == 'XGBoost':
        prediction = models['xgboost'].predict(input_df)[0]
    else:  # Ridge Regression
        input_scaled = models['scaler'].transform(input_df)
        prediction = models['ridge'].predict(input_scaled)[0]
    
    return max(0, prediction)  # Ensure non-negative prediction

def get_similar_cars_comparison(brand, year, fuel_type, sample_data):
    """Generate comparison with similar cars"""
    # This is a simplified comparison - in a real app, you'd query actual data
    base_price = 500000  # Base price for comparison
    
    # Price adjustments based on features
    brand_multiplier = {
        'Maruti': 0.8, 'Hyundai': 0.9, 'Toyota': 1.2, 'Honda': 1.1,
        'Mahindra': 0.95, 'Tata': 0.85, 'Ford': 0.9, 'Volkswagen': 1.1,
        'BMW': 2.5, 'Mercedes-Benz': 2.8, 'Audi': 2.6, 'Skoda': 1.15
    }
    
    fuel_multiplier = {
        'Petrol': 1.0, 'Diesel': 1.1, 'CNG': 0.9, 'LPG': 0.88, 'Electric': 1.5
    }
    
    age_factor = max(0.5, 1 - (2025 - year) * 0.08)  # Depreciation factor
    
    estimated_price = base_price * brand_multiplier.get(brand, 1.0) * fuel_multiplier.get(fuel_type, 1.0) * age_factor
    
    return estimated_price

def main():
    """Main Streamlit app"""
    
    # Load models and data
    models, feature_columns, sample_data = load_models_and_data()
    
    if models is None:
        st.stop()
    
    # Header
    st.title("🚗 Car Price Predictor")
    st.markdown("### Get accurate resale value predictions for your car")
    
    # Sidebar for inputs
    st.sidebar.header("Car Details")
    # Input fields
    brand = st.sidebar.selectbox("Brand", sample_data['brands'])

    # Brand to model mapping (from dataset)
    brand_to_models = {
        'Audi': ['A4', 'A6', 'Q7', 'A8'],
        'BMW': ['5', '3', 'Z4', '6', 'X5', 'X1', '7', 'X3', 'X4'],
        'Bentley': ['Continental'],
        'Datsun': ['RediGO', 'GO', 'redi-GO'],
        'Ferrari': ['GTC4Lusso'],
        'Force': ['Gurkha'],
        'Ford': ['Ecosport', 'Aspire', 'Figo', 'Endeavour', 'Freestyle'],
        'Honda': ['City', 'Amaze', 'CR-V', 'Jazz', 'Civic', 'WR-V', 'CRR'],
        'Hyundai': ['Grand', 'i20', 'i10', 'Venue', 'Verna', 'Creta', 'Santro', 'Elantra', 'Aura', 'Tucson'],
        'ISUZU': ['MUX'],
        'Isuzu': ['D-Max', 'MUX'],
        'Jaguar': ['XF', 'F-PACE', 'XE'],
        'Jeep': ['Wrangler', 'Compass'],
        'Kia': ['Seltos', 'Carnival'],
        'Land Rover': ['Rover'],
        'Lexus': ['ES', 'NX', 'RX'],
        'MG': ['Hector'],
        'Mahindra': ['Bolero', 'XUV500', 'KUV100', 'Scorpio', 'Marazzo', 'KUV', 'Thar', 'XUV300', 'Alturas'],
        'Maruti': ['Alto', 'Wagon R', 'Swift', 'Ciaz', 'Baleno', 'Swift Dzire', 'Ignis', 'Vitara', 'Cellerio', 'Ertiga', 'Eeco', 'Dzire VXI', 'XL6', 'S-Presso', 'Dzire LXI', 'Dzire ZXI'],
        'Maserati': ['Ghibli', 'Quattroporte'],
        'Mercedes-AMG': ['C'],
        'Mercedes-Benz': ['C-Class', 'E-Class', 'GL-Class', 'S-Class', 'CLS', 'GLS'],
        'Mini': ['Cooper'],
        'Nissan': ['Kicks', 'X-Trail'],
        'Porsche': ['Cayenne', 'Macan', 'Panamera'],
        'Renault': ['Duster', 'KWID', 'Triber'],
        'Rolls-Royce': ['Ghost'],
        'Skoda': ['Rapid', 'Superb', 'Octavia'],
        'Tata': ['Tiago', 'Tigor', 'Safari', 'Hexa', 'Nexon', 'Harrier', 'Altroz'],
        'Toyota': ['Innova', 'Fortuner', 'Camry', 'Yaris', 'Glanza'],
        'Volkswagen': ['Vento', 'Polo'],
        'Volvo': ['S90', 'XC', 'XC90', 'XC60']
    }
    models_for_brand = brand_to_models.get(brand, [])
    model = st.sidebar.selectbox("Model", models_for_brand if models_for_brand else ['N/A'])

    year = st.sidebar.slider("Manufacturing Year", 
                            min_value=sample_data['year_range'][0],
                            max_value=sample_data['year_range'][1],
                            value=2020)
    
    # Set default values for selectboxes if present
    fuel_type_options = sample_data['fuel_types']
    fuel_type_default = fuel_type_options.index('Petrol') if 'Petrol' in fuel_type_options else 0
    fuel_type = st.sidebar.selectbox("Fuel Type", fuel_type_options, index=fuel_type_default)

    transmission_options = sample_data['transmission_types']
    transmission_default = transmission_options.index('Manual') if 'Manual' in transmission_options else 0
    transmission_type = st.sidebar.selectbox("Transmission", transmission_options, index=transmission_default)

    seller_options = sample_data['seller_types']
    seller_default = seller_options.index('Individual') if 'Individual' in seller_options else 0
    seller_type = st.sidebar.selectbox("Seller Type", seller_options, index=seller_default)
    
    km_driven = st.sidebar.slider("Kilometers Driven", 
                                 min_value=sample_data['km_range'][0],
                                 max_value=min(200000, sample_data['km_range'][1]),
                                 value=50000,
                                 step=1000)
    
    engine = st.sidebar.slider("Engine (CC)", 
                              min_value=sample_data['engine_range'][0],
                              max_value=sample_data['engine_range'][1],
                              value=1200)
    
    max_power = st.sidebar.slider("Max Power (bhp)", 
                                 min_value=sample_data['power_range'][0],
                                 max_value=sample_data['power_range'][1],
                                 value=80.0)
    
    mileage = st.sidebar.slider("Mileage (km/l)", 
                               min_value=sample_data['mileage_range'][0],
                               max_value=sample_data['mileage_range'][1],
                               value=15.0)
    
    seats = st.sidebar.selectbox(
        "Number of Seats",
        range(sample_data['seats_range'][0], sample_data['seats_range'][1] + 1),
        index=(5 - sample_data['seats_range'][0]) if 5 in range(sample_data['seats_range'][0], sample_data['seats_range'][1] + 1) else 0
    )
    
    model_choice = st.sidebar.selectbox("Model", ['XGBoost', 'Ridge Regression'], index=1)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Predict Price", type="primary"):
            # Create input features
            input_df = create_input_features(
                brand, year, fuel_type, transmission_type, seller_type,
                km_driven, engine, max_power, mileage, seats, feature_columns, model=model
            )
            # Make prediction
            predicted_price = predict_price(input_df, models, model_choice)
            # Display prediction
            st.markdown(f"""
            <div class="price-prediction">
                <h2>🎯 Predicted Price</h2>
                <h1>₹{predicted_price:,.0f}</h1>
                <p>Model used: {model_choice}</p>
            </div>
            """, unsafe_allow_html=True)
            # Car summary
            st.markdown("### 📋 Car Summary")
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>Basic Info</h4>
                    <p><strong>Brand:</strong> {brand}</p>
                    <p><strong>Model:</strong> {model}</p>
                    <p><strong>Year:</strong> {year}</p>
                    <p><strong>Age:</strong> {2025-year} years</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col1_2:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>Performance</h4>
                    <p><strong>Engine:</strong> {engine} CC</p>
                    <p><strong>Power:</strong> {max_power} bhp</p>
                    <p><strong>Mileage:</strong> {mileage} km/l</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col1_3:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>Other Details</h4>
                    <p><strong>Fuel:</strong> {fuel_type}</p>
                    <p><strong>Transmission:</strong> {transmission_type}</p>
                    <p><strong>KM Driven:</strong> {km_driven:,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Price comparison
            st.markdown("### 📊 Price Analysis")
            
            # Get market comparison
            market_estimate = get_similar_cars_comparison(brand, year, fuel_type, sample_data)
            
            comparison_data = {
                'Metric': ['Our Prediction', 'Market Estimate', 'Difference'],
                'Value': [f"₹{predicted_price:,.0f}", f"₹{market_estimate:,.0f}", 
                         f"₹{abs(predicted_price - market_estimate):,.0f}"],
                'Status': ['Predicted', 'Estimated', 
                          'Higher' if predicted_price > market_estimate else 'Lower']
            }
            
            st.dataframe(comparison_data, hide_index=True)
            
            # Price factors
            st.markdown("### 💡 Price Factors")
            st.markdown(f"""
            <div class="comparison-box">
                <h4>Key factors affecting your car's price:</h4>
                <ul>
                    <li><strong>Brand Premium:</strong> {brand} has {'high' if brand in ['BMW', 'Mercedes-Benz', 'Audi'] else 'moderate' if brand in ['Toyota', 'Honda'] else 'standard'} brand value</li>
                    <li><strong>Age Impact:</strong> {2025-year} year old car - depreciation factor applied</li>
                    <li><strong>Fuel Type:</strong> {fuel_type} engines have {'high' if fuel_type == 'Electric' else 'good' if fuel_type == 'Diesel' else 'standard'} market demand</li>
                    <li><strong>Mileage:</strong> {km_driven:,} km driven - {'low' if km_driven < 30000 else 'moderate' if km_driven < 80000 else 'high'} usage</li>
                    <li><strong>Performance:</strong> {max_power} bhp power with {mileage} km/l efficiency</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Model Performance")
        # Display model info
        if model_choice == 'XGBoost':
            st.info("""
            **XGBoost Model**
            - Advanced gradient boosting
            - Handles non-linear relationships
            - Feature importance analysis
            - Good for complex patterns
            """)
        else:
            st.info("""
            **Ridge Regression Model**
            - Linear regression with regularization
            - Prevents overfitting
            - Faster predictions
            - Interpretable results
            """)

        # Only show price visualization and tips after prediction
        if 'predicted_price' in locals() or 'predicted_price' in globals():
            try:
                _ = predicted_price  # check if defined
                st.markdown("### 🎨 Price Visualization")
                # Create a simple price range chart
                fig, ax = plt.subplots(figsize=(8, 6))
                # Sample price ranges for visualization
                price_ranges = ['0-3L', '3-5L', '5-8L', '8-12L', '12L+']
                car_counts = [3500, 4200, 3800, 2400, 1500]  # Sample data
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
                bars = ax.bar(price_ranges, car_counts, color=colors, alpha=0.8)
                # Highlight the predicted price range
                if predicted_price < 300000:
                    bars[0].set_color('#FF4444')
                elif predicted_price < 500000:
                    bars[1].set_color('#FF4444')
                elif predicted_price < 800000:
                    bars[2].set_color('#FF4444')
                elif predicted_price < 1200000:
                    bars[3].set_color('#FF4444')
                else:
                    bars[4].set_color('#FF4444')
                ax.set_title('Car Price Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Price Range (₹)', fontsize=12)
                ax.set_ylabel('Number of Cars', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
                # Quick tips
                st.markdown("### 💰 Selling Tips")
                st.markdown("""
                - **Service History**: Keep all service records
                - **Clean Condition**: Well-maintained cars get better prices
                - **Market Timing**: Avoid festive seasons for selling
                - **Documentation**: Keep all papers ready
                - **Multiple Quotes**: Get quotes from different dealers
                """)
            except Exception:
                pass

if __name__ == "__main__":
    main()
