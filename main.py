import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(data_size=6, random_state=42):
    try:
        # Create a simulated dataset
        data = pd.DataFrame({
            'rainfall_mm': [5, 7, 6, 5, 8, 6],
            'temperature_avg': [20, 21.5, 20.5, 19, 22.5, 22.5],
            'evaporation_rate_mm': [2, 2.5, 2.2, 2.3, 2.6, 2.4],
            'soil_moisture': [30, 28, 29, 32, 31, 30],
            'ndwi': [0.2, 0.25, 0.23, 0.2, 0.3, 0.28],
            'water_level': [4.9, 5.0, 5.1, 5.3, 5.4, 5.2],
            'rainfall_7day_avg': [5.0, 6.0, 6.5, 5.5, 7.0, 6.0],
            'temperature_7day_avg': [20.0, 21.0, 21.5, 20.5, 22.0, 21.5]
        }).sample(n=data_size, random_state=random_state)

        # Creating a binary classification target
        y = (data['water_level'] <= 5.1).astype(int)

        # Features
        X = data[['rainfall_mm', 'temperature_avg', 'evaporation_rate_mm', 'soil_moisture',
                  'ndwi', 'rainfall_7day_avg', 'temperature_7day_avg']]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Check class distribution before applying SMOTE
        class_counts = y_train.value_counts()
        st.write(f"Class distribution before SMOTE: \n{class_counts}")

        # Set k_neighbors to 1 if any class has less than 2 samples
        k_neighbors = min(class_counts) - 1 if min(class_counts) > 1 else 1

        # Apply SMOTE
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)

        # Visualize the class distribution before and after SMOTE
        st.write("### Class Distribution Before and After SMOTE")
        visualize_class_distribution(y_train, y_train_balanced)

        # Return the preprocessed data
        return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, X.columns.tolist()

    except Exception as e:
        st.error(f"An error occurred during data loading/preprocessing: {e}")
        return None

def visualize_class_distribution(original_y, balanced_y):
    # Plot before SMOTE
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.countplot(x=original_y, ax=ax[0])
    ax[0].set_title("Class Distribution Before SMOTE")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["Not Dry", "Likely Dry"])

    # Plot after SMOTE
    sns.countplot(x=balanced_y, ax=ax[1])
    ax[1].set_title("Class Distribution After SMOTE")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["Not Dry", "Likely Dry"])

    st.pyplot(fig)

# Call your function in the Streamlit app
if __name__ == "__main__":
    st.title('Water Body Dry-Up Prediction')
    X_train, X_test, y_train_balanced, y_test, scaler, feature_names = load_and_preprocess_data()
