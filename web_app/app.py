import streamlit as st
import pandas as pd
import joblib


st.write("""
# Intelligent Transportation Optimization Platform

Streamline **Maintenance Predictions** for Enhanced Transport Efficiency

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Use the sliders below to input key parameters for maintenance prediction:
""")

# Collects user input features into dataframe
def user_input_features():
    vehicle_speed_sensor = st.sidebar.slider('Vehicle Speed Sensor', min_value=0.0, max_value=186.0, value=82.0)
    vibration = st.sidebar.slider('Vibration', min_value=240.0, max_value=252.0, value=246.5)
    engine_load = st.sidebar.slider('Engine Load', min_value=0.0, max_value=100.0, value=36.8)
    engine_coolant_temp = st.sidebar.slider('Engine Coolant Temperature', min_value=79.0, max_value=94.0, value=87.0)
    intake_manifold_pressure = st.sidebar.slider('Intake Manifold Pressure', min_value=102.0, max_value=255.0, value=120.0)
    engine_rpm = st.sidebar.slider('Engine RPM', min_value=669, max_value=2597, value=1320)
    speed_obd = st.sidebar.slider('Speed OBD', min_value=0.0, max_value=186.0, value=82.0)
    intake_air_temp = st.sidebar.slider('Intake Air Temperature', min_value=6.0, max_value=18.0, value=10.0)
    mass_air_flow_rate = st.sidebar.slider('Mass Air Flow Rate', min_value=5.86, max_value=117.91, value=25.32)
    throttle_position_manifold = st.sidebar.slider('Throttle Position Manifold', min_value=27.8, max_value=81.2, value=76.)
    voltage_control_module = st.sidebar.slider('Voltage Control Module', min_value=12.9, max_value=14.4, value=14.2)
    ambient_air_temp = st.sidebar.slider('Ambient Air Temperature', min_value=5, max_value=9, value=7)
    accel_pedal_pos_d = st.sidebar.slider('Accelerator Pedal Position D', min_value=14.0, max_value=79.0, value=22.3)
    engine_oil_temp = st.sidebar.slider('Engine Oil Temperature', min_value=78.0, max_value=92.0, value=84.0)
    litres_per_100km_inst = st.sidebar.slider('Litres per 100km Instant', min_value=0.0, max_value=32.0, value=3.4)
    co2_in_g_per_km_inst = st.sidebar.slider('CO2 in g per km Instant', min_value=0.0, max_value=865.2, value=93.6)
    data = {
        'Vehicle_speed_sensor': vehicle_speed_sensor,
        'Vibration': vibration,
        'Engine_Load': engine_load,
        'Engine_Coolant_Temp': engine_coolant_temp,
        'Intake_Manifold_Pressure': intake_manifold_pressure,
        'Engine_RPM': engine_rpm,
        'Speed_OBD': speed_obd,
        'Intake_Air_Temp': intake_air_temp,
        'Mass_Air_Flow_Rate': mass_air_flow_rate,
        'Throttle_Pos_Manifold': throttle_position_manifold,
        'Voltage_Control_Module': voltage_control_module,
        'Ambient_air_temp': ambient_air_temp,
        'Accel_Pedal_Pos_D': accel_pedal_pos_d,
        'Engine_Oil_Temp': engine_oil_temp,
        'Litres_Per_100km_Inst': litres_per_100km_inst,
        'CO2_in_g_per_km_Inst': co2_in_g_per_km_inst,
    }
    features = pd.DataFrame(data, index=[0])
    return features
data = user_input_features()

# Displays the user input features
st.subheader('User Input features')
st.write(data)

# Reads in saved classification model
load_model = joblib.load("itos.pkl")

# Apply model to make predictions
prediction = load_model.predict(data)
prediction_proba = load_model.predict_proba(data)


st.subheader('Prediction')

if prediction == 0:
    st.write("The prediction is **No Maintenance Needed**.")
else:
    st.write("The prediction is **Maintenance Needed**.")
    st.write(f"Probability of Maintenance: **{list(prediction_proba)[0][1]:.2%}**")
