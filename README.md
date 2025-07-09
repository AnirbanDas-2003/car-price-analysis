# Car Price Prediction Project

A comprehensive car price prediction system using machine learning models (XGBoost and Ridge Regression) with a user-friendly Streamlit interface.

## 🚗 Project Overview

This project predicts the resale value of cars based on various features like brand, year, fuel type, mileage, engine specifications, and more. It uses the CarDekho dataset and provides both XGBoost and Ridge Regression models for comparison.

## 📋 Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Multiple Models**: XGBoost and Ridge Regression for comparison
- **One-hot Encoding**: Proper handling of categorical variables
- **Interactive Web App**: Beautiful Streamlit interface for predictions
- **Price Comparison**: Market analysis and price factors
- **Visualization**: Charts and graphs for better insights

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project uses the CarDekho dataset (`cardekho_dataset.csv`) with the following features:

- **car_name**: Name of the car
- **brand**: Car manufacturer
- **model**: Car model
- **vehicle_age**: Age of the vehicle
- **km_driven**: Kilometers driven
- **seller_type**: Type of seller (Individual/Dealer/Trustmark Dealer)
- **fuel_type**: Fuel type (Petrol/Diesel/CNG/LPG/Electric)
- **transmission_type**: Transmission type (Manual/Automatic)
- **mileage**: Fuel efficiency in km/l
- **engine**: Engine capacity in CC
- **max_power**: Maximum power in bhp
- **seats**: Number of seats
- **selling_price**: Target variable (price in ₹)

## 🚀 Usage

### 1. Train the Models

First, train the machine learning models:

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Perform feature engineering
- Train XGBoost and Ridge Regression models
- Save the trained models and preprocessing data
- Display model performance metrics

### 2. Run the Streamlit App

Launch the web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Using the Web App

1. **Input Car Details**: Use the sidebar to enter car specifications
   - Brand, manufacturing year, fuel type
   - Transmission type, seller type
   - Kilometers driven, engine capacity
   - Maximum power, mileage, number of seats

2. **Select Model**: Choose between XGBoost or Ridge Regression

3. **Get Prediction**: Click "Predict Price" to get the estimated resale value

4. **View Analysis**: Review the price analysis, market comparison, and factors affecting the price

## 📈 Model Performance

The project includes two machine learning models:

### XGBoost
- **Advantages**: Handles non-linear relationships, feature importance analysis
- **Use Case**: Complex patterns and high accuracy predictions

### Ridge Regression
- **Advantages**: Linear model with regularization, faster predictions
- **Use Case**: Interpretable results and baseline comparisons

## 🎯 Key Features of the App

### Price Prediction
- Accurate price estimates using trained ML models
- Real-time predictions based on user inputs
- Model comparison functionality

### Market Analysis
- Price comparison with market estimates
- Factors affecting car prices
- Price distribution visualization

### User Interface
- Clean, modern design with custom CSS
- Responsive layout with sidebar inputs
- Interactive charts and visualizations
- Comprehensive car summary display

## 📁 Project Structure

```
car-price-analysis/
├── cardekho_dataset.csv          # Dataset
├── train_model.py                # Model training script
├── streamlit_app.py              # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── car_price_models.pkl          # Trained models (generated)
├── feature_columns.pkl           # Feature columns (generated)
└── sample_data.pkl               # Sample data for app (generated)
```

## 🔧 Technical Details

### Data Preprocessing
- **Feature Engineering**: Created year, price_per_km, power_to_engine_ratio
- **One-hot Encoding**: Categorical variables properly encoded
- **Data Cleaning**: Handled missing values and outliers

### Model Training
- **Train-Test Split**: 80-20 split for model evaluation
- **Feature Scaling**: StandardScaler for Ridge Regression
- **Model Evaluation**: RMSE, MAE, and R² metrics

### Web Application
- **Streamlit Framework**: Modern web interface
- **Interactive Inputs**: Sliders, selectboxes, and buttons
- **Data Visualization**: Matplotlib and Seaborn integration
- **Custom Styling**: CSS for enhanced user experience

## 💡 Usage Tips

1. **Accurate Inputs**: Provide precise car details for better predictions
2. **Model Selection**: Try both models and compare results
3. **Market Context**: Consider the price factors and market analysis
4. **Regular Updates**: Retrain models with new data for better accuracy

## 🔮 Future Enhancements

- **More Models**: Integration of other ML algorithms
- **Real-time Data**: Integration with live car market data
- **Advanced Features**: Image recognition for car condition assessment
- **Location-based Pricing**: Regional price variations
- **Historical Trends**: Price trend analysis over time

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**: Run `train_model.py` first
2. **Import errors**: Check if all packages are installed
3. **Data loading issues**: Ensure `cardekho_dataset.csv` is in the project directory
4. **Streamlit port issues**: Use `streamlit run --server.port 8502 streamlit_app.py`

### Performance Tips

- **Model Training**: May take 2-3 minutes depending on system
- **Streamlit Loading**: First load might take longer due to model loading
- **Data Caching**: Models are cached for faster subsequent predictions

## 📞 Support

For any issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure the dataset file is present
4. Make sure models are trained before running the app

## 📄 License

This project is for educational and demonstration purposes. The dataset is from CarDekho and should be used according to their terms of service.

---

**Happy Predicting! 🚗💰**
