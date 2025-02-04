from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

def load_and_preprocess_data():
    # Simulated dataset creation
    data = pd.DataFrame({
        'rainfall_mm': [5, 7, 6, 5, 8, 6],
        'temperature_avg': [20, 21.5, 20.5, 19, 22.5, 22.5],
        'evaporation_rate_mm': [2, 2.5, 2.2, 2.3, 2.6, 2.4],
        'soil_moisture': [30, 28, 29, 32, 31, 30],
        'ndwi': [0.2, 0.25, 0.23, 0.2, 0.3, 0.28],
        'water_level': [5.0, 5.2, 5.1, 5.3, 5.4, 5.2],
        'rainfall_7day_avg': [5.0, 6.0, 6.5, 5.5, 7.0, 6.0],
        'temperature_7day_avg': [20.0, 21.0, 21.5, 20.5, 22.0, 21.5]
    })

    # Creating binary classification target
    y = (data['water_level'] <= 5.1).astype(int)
    
    # Features selection
    X = data[['rainfall_mm', 'temperature_avg', 'evaporation_rate_mm', 'soil_moisture', 
              'ndwi', 'rainfall_7day_avg', 'temperature_7day_avg']]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Balancing the training data with SMOTE
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Return preprocessed data
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler
