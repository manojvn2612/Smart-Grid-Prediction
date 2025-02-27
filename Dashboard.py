import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
scaler_load = joblib.load('minmax_scaler_load.pkl')
scaler_wet = joblib.load('minmax_scaler_wet.pkl')
model = load_model('E:\SIH 2024\lstm_model.h5')

st.set_page_config(page_title='Load Prediction', layout='wide')
st.markdown("<h1 style='text-align: center;'>Load Prediction</h1>", unsafe_allow_html=True)
st.write("""
### Predict the Grid's Average Load
This app takes various weather parameters and predicts the average load that the grid will hold for the given hour of the day.
""")

# Use containers to group related inputs with titles for better UI
st.markdown("---")  # Add a line divider

# Section for time and temperature inputs
st.subheader("Weather Conditions")

container1 = st.container()

with container1:
    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
    with col2:
        temp = st.number_input('Temperature (°F)', value=25.0)
    with col3:
        feelslike = st.number_input('Feels Like (°F)', value=25.0)

# Section for humidity, dew, and precipitation
st.subheader("Atmospheric Conditions")

container2 = st.container()

with container2:
    col1, col2, col3 = st.columns(3)
    with col1:
        humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=50)
    with col2:
        dew = st.number_input('Dew Point (°F)', value=15.0)
    with col3:
        precip = st.number_input('Precipitation (mm)', value=0.0)

# Section for wind and pressure inputs
st.subheader("Wind and Pressure")

container3 = st.container()

with container3:
    col1, col2, col3 = st.columns(3)
    with col1:
        windgust = st.number_input('Wind Gust (km/h)', value=10.0)
    with col2:
        windspeed = st.number_input('Wind Speed (km/h)', value=10.0)
    with col3:
        pressure = st.number_input('Pressure (hPa)', value=1000.0)

# Section for visibility, cloud cover, and solar data
st.subheader("Visibility and Solar Conditions")

container4 = st.container()

with container4:
    col1, col2, col3 = st.columns(3)
    with col1:
        visibility = st.number_input('Visibility (km)', value=10.0)
    with col2:
        cloudcover = st.number_input('Cloud Cover (%)', min_value=0, max_value=100, value=50)
    with col3:
        solarradiation = st.number_input('Solar Radiation (W/m²)', value=500.0)

# Section for energy and UV index
st.subheader("Solar Energy and UV Index")

container5 = st.container()

with container5:
    col1, col2, col3 = st.columns(3)
    with col1:
        solarenergy = st.number_input('Solar Energy (MJ/m²)', value=5.0)
    with col2:
        uvindex = st.number_input('UV Index', min_value=0, max_value=11, value=5)
    with col3:
        day = st.selectbox('Day of the Week', ['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday'])

# Divider before prediction button
st.markdown("---")

# Create a one-hot encoding for the day
days_list = ['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday']
day_one_hot = [1 if day == d else 0 for d in days_list]

# Make a prediction when the button is clicked
if st.button('Predict'):
    if feelslike == 25.0:
        feelslike = temp

    # Create a feature array from the input data, with the day as one-hot encoding
    input_data = np.array([[hour, temp, feelslike, dew, humidity, precip, windgust,
                            windspeed, pressure, cloudcover, visibility, solarradiation, solarenergy, uvindex]])
    input_data_scaled = scaler_wet.transform(input_data)
    input_data_scaled = np.concatenate([input_data_scaled[0], day_one_hot])

    input_data = input_data_scaled.reshape((1, 1, input_data_scaled.shape[0]))

    # Prediction using the custom model
    prediction = model.predict(input_data)

    y_pred = scaler_load.inverse_transform(prediction)

    # Display the prediction result with styling
    st.write(f'### Predicted Avg Load: **{y_pred[0][0]:.2f} Mw**')


# Input fields
# hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
# temp = st.number_input('Temperature (°F)', value=25.0)
# feelslike = st.number_input('Feels Like (°F)', value=25.0)
# humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=50)
# dew = st.number_input('Dew Point (°F)', value=15.0)
# precip = st.number_input('Precipitation (mm)', value=0.0)
# windgust = st.number_input('Wind Gust (km/h)', value=10.0)
# windspeed = st.number_input('Wind Speed (km/h)', value=10.0)
# pressure = st.number_input('Pressure (hPa)', value=1000.0)
# visibility = st.number_input('Visibility (km)', value=10.0)
# cloudcover = st.number_input('Cloud Cover (%)', min_value=0, max_value=100, value=50)
# solarradiation = st.number_input('Solar Radiation (W/m²)', value=500.0)
# solarenergy = st.number_input('Solar Energy (MJ/m²)', value=5.0)
# uvindex = st.number_input('UV Index', min_value=0, max_value=11, value=5)
# Add day of the week using a selectbox
# day = st.selectbox(
#     'Day of the Week',
#     ['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday']
# )
import streamlit as st
import numpy as np


