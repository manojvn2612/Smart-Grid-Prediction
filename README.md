# Grid Load Prediction Using LSTM

## Overview
This project is a **Streamlit-based web application** that predicts the average grid load for a given hour based on various weather parameters. The application utilizes a **pre-trained LSTM model** to make predictions, taking user-input weather conditions and applying **MinMax scaling** for accurate forecasting.

## Features
- **User-Friendly Interface**: Built using Streamlit for ease of interaction.
- **Real-Time Predictions**: Predicts the grid's average load based on provided inputs.
- **Multiple Weather Parameters**: Includes temperature, humidity, wind speed, solar energy, and more.
- **LSTM Model Integration**: Uses a deep learning model for accurate forecasting.
- **One-Hot Encoding for Days**: Accounts for variations in daily energy consumption.

## Tech Stack
- **Python** (Primary Language)
- **Streamlit** (Frontend UI)
- **TensorFlow/Keras** (LSTM Model)
- **Joblib** (For loading MinMax scalers)
- **NumPy** (Data processing)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Bholu877/Grid-Load-Prediction.git
   cd Grid-Load-Prediction

   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit interface in your browser.
2. Input weather parameters such as temperature, humidity, wind speed, and more.
3. Select the day of the week.
4. Click the "Predict" button to get the estimated grid load.

## File Structure
```
├── app.py                  # Main Streamlit application
├── lstm_model.h5           # Pre-trained LSTM model
├── minmax_scaler_load.pkl  # Scaler for load values
├── minmax_scaler_wet.pkl   # Scaler for weather parameters
├── requirements.txt        # List of dependencies
├── README.md               # Project Documentation
```

## Future Improvements
- Enhance UI with better visuals.
- Improve model accuracy with more training data.
- Deploy as a web application using **AWS/GCP/Heroku**.



