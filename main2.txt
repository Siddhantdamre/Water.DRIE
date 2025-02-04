import streamlit as st
import numpy as np
import joblib

# Load model and scaler once
@st.cache_resource
def load_model_and_scaler():
    # Load the model and scaler from pre-trained files (adjust the paths as needed)
    model = joblib.load('path/to/your_model.pkl')  # Adjust path to your model
    scaler = joblib.load('path/to/your_scaler.pkl')  # Adjust path to your scaler
    return model, scaler

# Function to render Streamlit UI and make predictions
def run_streamlit_app():
    st.title('Water Body Dry-Up Prediction')

    # Collecting user input
    rainfall_mm = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=100.0)
    temperature_avg = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)
    evaporation_rate_mm = st.number_input('Evaporation Rate (mm)', min_value=0.0, max_value=10.0, value=2.0)
    soil_moisture = st.number_input('Soil Moisture', min_value=0.0, max_value=1.0, value=0.5)
    ndwi = st.number_input('NDWI', min_value=-1.0, max_value=1.0, value=0.0)
    rainfall_7day_avg = st.number_input('7-Day Avg Rainfall (mm)', min_value=0.0, max_value=500.0, value=100.0)
    temperature_7day_avg = st.number_input('7-Day Avg Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)

    # Input Data Preparation
    input_data = np.array([[rainfall_mm, temperature_avg, evaporation_rate_mm, soil_moisture, ndwi, rainfall_7day_avg, temperature_7day_avg]])

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    # Standardize input data
    try:
        input_data_scaled = scaler.transform(input_data)

        # Prediction on user input
        if st.button('Predict'):
            prediction = model.predict(input_data_scaled)
            if prediction[0] == 1:
                st.write("The water body is likely to dry up.")
            else:
                st.write("The water body is not likely to dry up.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    run_streamlit_app()
