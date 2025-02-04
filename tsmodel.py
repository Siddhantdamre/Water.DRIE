from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def train_arima_model(water_level_ts, order=(5, 1, 0), steps=10):
    try:
        # Fit ARIMA model on water level time series data
        arima_model = ARIMA(water_level_ts, order=order)
        arima_model_fit = arima_model.fit()

        # Forecast future water levels
        arima_forecast = arima_model_fit.forecast(steps=steps)
        print("ARIMA Forecast:\n", arima_forecast)

        # Plot the forecast
        plot_forecast(water_level_ts, arima_forecast)

        return arima_model_fit, arima_forecast

    except Exception as e:
        print(f"Error in fitting ARIMA model: {e}")
        return None

def plot_forecast(actual, forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual", color='blue')
    plt.plot(range(len(actual), len(actual) + len(forecast)), forecast, label="Forecast", linestyle='--', color='orange')
    plt.title("Water Level Forecast")
    plt.xlabel("Time Steps")
    plt.ylabel("Water Level")
    plt.legend()
    plt.grid()
    plt.show()
