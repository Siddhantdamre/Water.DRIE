from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
def load_and_preprocess_data():
    np.random.seed(42)
    data_size = 1000
    X = pd.DataFrame({
    'rainfall_mm': np.random.uniform(0, 100, data_size),  # More variability
    'temperature_avg': np.random.uniform(10, 40, data_size),
    'evaporation_rate_mm': np.random.uniform(0, 10, data_size),  # More variability
    'soil_moisture': np.random.uniform(0, 1, data_size),
    'ndwi': np.random.uniform(0, 1, data_size)


    })
    
    # Classification target: More realistic drying-up condition
    y_classification = (
        (X['rainfall_mm'] < 30) &  # Low rainfall
        (X['evaporation_rate_mm'] > 3) &  # High evaporation
        (X['soil_moisture'] < 0.4)  # Low soil moisture
    ).astype(int)
    
    # Regression target: Time to dry up is inversely related to rainfall and soil moisture
    y_regression = X['rainfall_mm'] / (X['evaporation_rate_mm'] + 1) * (1 + X['soil_moisture'])
    
    # Split the dataset (before SMOTE)
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_classification, y_regression, test_size=0.2, random_state=42
    )
    
    # Apply SMOTE only to classification data
    smote = SMOTE(random_state=42)
    X_resampled, y_train_class_resampled = smote.fit_resample(X_train, y_train_class)
    
    # Adjust the regression target to match resampled X
    y_train_reg_resampled = np.repeat(y_train_reg, np.ceil(len(X_resampled) / len(y_train_reg)).astype(int))[:len(X_resampled)]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_class_resampled, y_test_class, y_train_reg_resampled, y_test_reg, scaler

# Build models
def build_classification_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def build_regression_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Streamlit app
def run_app():
    st.title("Water Body Dry-Up Prediction System")
    
    # Input fields in Streamlit
    st.sidebar.subheader("Input Environmental Data")
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=10.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    evaporation_rate = st.sidebar.number_input("Evaporation Rate (mm)", min_value=0.0, max_value=10.0, value=2.5)
    soil_moisture = st.sidebar.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=50.0) / 100.0
    ndwi = st.sidebar.number_input("NDWI", min_value=0.0, max_value=1.0, value=0.5)
    
    # Create a user input dataframe
    user_data = pd.DataFrame({
        'rainfall_mm': [rainfall],
        'temperature_avg': [temperature],
        'evaporation_rate_mm': [evaporation_rate],
        'soil_moisture': [soil_moisture],
        'ndwi': [ndwi]
    

    })
    
    # Load and preprocess data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg, scaler = load_and_preprocess_data()
    
    # Scale user input data
    user_data_scaled = scaler.transform(user_data)
    
    # Build models
    class_model = build_classification_model(X_train, y_train_class)
    reg_model = build_regression_model(X_train, y_train_reg)

    
    # Add this section after building the classification model
    y_train_class_pred = class_model.predict(X_train)
    st.write(classification_report(y_train_class, y_train_class_pred))
    
    # After building the classification model
    class_model = build_classification_model(X_train, y_train_class)

    # Print predictions for the training set
    predictions = class_model.predict(X_train)
    st.write(f"Training set predictions: {predictions}")

    # Make predictions
    if st.sidebar.button("Predict"):
        # Classification: Will the water body dry up?
        will_dry = class_model.predict(user_data_scaled)
        dry_prob = class_model.predict_proba(user_data_scaled)[0][1]
        
        if will_dry == 1:
            st.write(f"Prediction: The water body is **likely to dry up** with a probability of {dry_prob:.2f}.")            
            # Regression: Time to dry up
            time_to_dry = reg_model.predict(user_data_scaled)[0]
            st.write(f"Estimated time to dry up: {time_to_dry:.2f} days.")

        else:
            st.write(f"Prediction: The water body is **not likely to dry up**.")
        

# Run the app
if __name__ == "__main__":
    run_app()
