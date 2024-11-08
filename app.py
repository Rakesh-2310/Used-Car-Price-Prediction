import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained XGBoost model
xgboost_model = joblib.load('xgboost_best_model.pkl')

# Encoding mappings based on DataFrame's unique values
encoding_mappings = {
    'fuel_type': {0: 'Petrol', 1: 'Diesel', 2: 'Gas'},
    'body_type': {0: 'Hatchback', 1: 'SUV', 2: 'Sedan', 3: 'MUV', 4: 'Coupe', 5: 'Minivans'},
    'transmission': {0: 'Manual', 1: 'Automatic'},
    'drive_type': {0: 'FWD', 1: '2WD', 2: 'RWD', 3: '4WD', 4: 'AWD'},
    'driver_air_bag': {0: 'No', 1: 'Yes'},
    'passenger_air_bag': {0: 'No', 1: 'Yes'}, 
    'rear_camera': {0: 'No', 1: 'Yes'},
    'front_brake_type_group': {0: 'Drum', 1: 'Disc'},
    'rear_brake_type_group': {0: 'Drum', 1: 'Disc'},
    'car_roof': {0: 'No', 1: 'Yes'},
    'city': {0: 'Bangalore', 1: 'Chennai', 2: 'Delhi', 3: 'Hyderabad', 4: 'Jaipur', 5: 'Kolkata'},
    'oem': {0: 'maruti', 1: 'ford', 2: 'tata', 3: 'hyundai', 4: 'jeep', 5: 'datsun', 6: 'honda', 7: 'bmw', 8: 'renault', 9: 'audi', 10: 'mercedes-benz', 11: 'toyota', 12: 'kia', 13: 'skoda', 14: 'volkswagen', 15: 'mahindra', 16: 'volvo', 17: 'nissan', 18: 'mahindra ssangyong', 19: 'mitsubishi', 20: 'mg', 21: 'chevrolet', 22: 'fiat', 23: 'jaguar', 24: 'mini', 25: 'isuzu', 26: 'lexus', 27: 'land rover', 28: 'porsche'}, 
    'model': {0: 'maruti celerio', 1: 'ford ecosport', 2: 'tata tiago', 3: 'hyundai xcent', 4: 'maruti sx4 s cross', 5: 'jeep compass', 6: 'datsun go', 7: 'maruti ciaz', 8: 'maruti baleno', 9: 'hyundai grand i10', 10: 'honda jazz', 11: 'hyundai i20', 12: 'tata nexon', 13: 'honda city', 14: 'bmw 5 series', 15: 'maruti swift', 16: 'renault duster', 17: 'hyundai santro', 18: 'hyundai santro xing', 19: 'audi a4', 20: 'maruti wagon r', 21: 'maruti ertiga', 22: 'mercedes-benz c-class', 23: 'toyota fortuner', 24: 'hyundai elantra', 25: 'audi a6', 26: 'kia seltos', 27: 'maruti alto', 28: 'tata new safari', 29: 'renault kwid', 30: 'skoda rapid', 31: 'hyundai creta', 32: 'tata harrier', 33: 'bmw 3 series gt', 34: 'renault lodgy', 35: 'skoda octavia', 36: 'maruti ritz', 37: 'volkswagen polo', 38: 'mahindra kuv 100', 39: 'hyundai i10', 40: 'volvo s60', 41: 'mahindra xuv300', 42: 'honda brio', 43: 'maruti alto k10', 44: 'renault kiger', 45: 'hyundai eon', 46: 'volkswagen vento', 47: 'toyota yaris', 48: 'volkswagen t-roc', 49: 'bmw 3 series', 50: 'audi q5', 51: 'ford endeavour', 52: 'ford figo', 53: 'maruti ignis', 54: 'hyundai tucson', 55: 'hyundai verna', 56: 'mercedes-benz glc', 57: 'nissan terrano', 58: 'honda cr-v', 59: 'toyota innova', 60: 'hyundai santa fe', 61: 'maruti baleno rs', 62: 'maruti vitara brezza', 63: 'hyundai i20 active', 64: 'mercedes-benz e-class', 65: 'honda wr-v', 66: 'mahindra ssangyong rexton', 67: 'toyota corolla altis', 68: 'mitsubishi cedia', 69: 'hyundai venue', 70: 'audi a3', 71: 'skoda kushaq', 72: 'maruti swift dzire', 73: 'hyundai accent', 74: 'mercedes-benz b class', 75: 'skoda laura', 76: 'skoda superb', 77: 'kia sonet', 78: 'mahindra verito', 79: 'maruti s-presso', 80: 'volkswagen jetta', 81: 'ford aspire', 82: 'ford freestyle', 83: 'audi q3', 84: 'tata tigor', 85: 'mg hector', 86: 'mercedes-benz a class', 87: 'toyota glanza', 88: 'maruti celerio x', 89: 'mercedes-benz cla', 90: 'volkswagen tiguan', 91: 'tata indica v2', 92: 'toyota innova crysta', 93: 'volkswagen ameo', 94: 'bmw x1', 95: 'chevrolet cruze', 96: 'fiat punto abarth', 97: 'mahindra tuv 300', 98: 'chevrolet beat', 99: 'maruti eeco', 100: 'hyundai grand i10 nios', 101: 'tata zest', 102: 'honda new accord', 103: 'maruti alto 800', 104: 'skoda yeti', 105: 'maruti sx4', 106: 'jaguar xe', 107: 'chevrolet spark', 108: 'tata nano', 109: 'honda amaze', 110: 'tata manza', 111: 'tata hexa', 112: 'nissan micra active', 113: 'ford fiesta', 114: 'bmw x3', 115: 'fiat punto', 116: 'kia carens', 117: 'chevrolet enjoy', 118: 'volvo xc40', 119: 'renault triber', 120: 'skoda slavia', 121: 'mahindra marazzo', 122: 'tata indigo', 123: 'skoda fabia', 124: 'nissan sunny', 125: 'datsun redigo', 126: 'fiat palio', 127: 'toyota etios', 128: 'bmw 1 series', 129: 'bmw x5', 130: 'nissan micra', 131: 'fiat punto evo', 132: 'mini cooper countryman', 133: 'renault fluence', 134: 'maruti a-star', 135: 'chevrolet sail', 136: 'fiat linea', 137: 'maruti xl6', 138: 'hyundai sonata', 139: 'honda civic', 140: 'mini cooper', 141: 'volvo s90', 142: 'honda br-v', 143: 'skoda kodiaq', 144: 'tata tiago nrg', 145: 'datsun go plus', 146: 'toyota camry', 147: 'maruti wagon r stingray', 148: 'mini 5 door', 149: 'fiat grande punto', 150: 'mahindra kuv 100 nxt', 151: 'chevrolet aveo', 152: 'tata indica', 153: 'toyota hyryder', 154: 'maruti zen estilo', 155: 'isuzu mu-x', 156: 'fiat punto pure', 157: 'honda mobilio', 158: 'mitsubishi pajero', 159: 'lexus es', 160: 'nissan kicks', 161: 'mercedes-benz gla class', 162: 'toyota etios cross', 163: 'toyota etios liva', 164: 'lexus rx', 165: 'mercedes-benz cls-class', 166: 'maruti jimny', 167: 'mini cooper clubman', 168: 'maruti grand vitara', 169: 'chevrolet optra', 170: 'mitsubishi outlander', 171: 'ford fiesta classic', 172: 'maruti 800', 173: 'volvo xc60', 174: 'mahindra alturas g4', 175: 'volkswagen passat', 176: 'fiat avventura', 177: 'renault scala', 178: 'tata aria', 179: 'volvo v40', 180: 'tata bolt', 181: 'fiat abarth avventura', 182: 'mahindra bolero neo', 183: 'chevrolet captiva', 184: 'volvo s60 cross country', 185: 'chevrolet aveo u-va', 186: 'land rover range rover evoque', 187: 'renault pulse', 188: 'volkswagen crosspolo', 189: 'porsche macan', 190: 'porsche panamera'},
}

inverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in encoding_mappings.items()}

# Streamlit user interface
st.title("🚗 **Car Price Prediction**")
st.markdown("### Enter the car details below to get an estimated price:")

# Layout and grouping fields
with st.form("car_details_form"):
    col1, col2 = st.columns(2)
    with col1:
        fuel_type = st.selectbox('Fuel Type', list(encoding_mappings['fuel_type'].values()))
        body_type = st.selectbox('Body Type', list(encoding_mappings['body_type'].values()))
        transmission = st.selectbox('Transmission', list(encoding_mappings['transmission'].values()))
        owners = st.number_input('Number of Owners', min_value=1, max_value=5, value=1)
        oem = st.selectbox('OEM (Manufacturer)', list(encoding_mappings['oem'].values()))
        model = st.selectbox('Model', list(encoding_mappings['model'].values()))

    with col2:
        km_driven = st.number_input('Kilometers Driven', min_value=0, value=60000, step=5000)
        modelyear = st.number_input('Model Year', min_value=1900, max_value=2024, value=2020)
        registration_year = st.number_input('Registration Year', min_value=1900, max_value=2024, value=2020)
        seats = st.number_input('Seats', min_value=2, max_value=10, value=4, step=1)
        engine_displacement = st.slider('Engine Displacement (in cc)', 624, 5000, step=100, value=1500)
        mileage = st.number_input('Mileage (in km/l)', min_value=5.0, max_value=50.0, value=20.0, step=0.5)

    # Additional Features
    st.markdown("### **Safety & Features**")
    col3, col4 = st.columns(2)
    with col3:
        driver_air_bag = st.selectbox('Driver Air Bag', list(encoding_mappings['driver_air_bag'].values()))
        passenger_air_bag = st.selectbox('Passenger Air Bag', list(encoding_mappings['passenger_air_bag'].values()))
        rear_camera = st.selectbox('Rear Camera', list(encoding_mappings['rear_camera'].values()))
        car_roof = st.selectbox('Car Roof', list(encoding_mappings['car_roof'].values()))

    with col4:
        front_brake_type_group = st.selectbox('Front Brake Type', list(encoding_mappings['front_brake_type_group'].values()))
        rear_brake_type_group = st.selectbox('Rear Brake Type', list(encoding_mappings['rear_brake_type_group'].values()))
        drive_type = st.selectbox('Drive Type', list(encoding_mappings['drive_type'].values()))
        city = st.selectbox('City', list(encoding_mappings['city'].values()))
        gears = st.selectbox('Number of Gears', list(range(4, 11)))

    submitted = st.form_submit_button('Predict Price 💰')
    if submitted:
        # Encode the selected inputs using inverse mappings
        input_data = {
            'fuel_type': inverse_mappings['fuel_type'][fuel_type],
            'body_type': inverse_mappings['body_type'][body_type],
            'km_driven': km_driven,
            'transmission': inverse_mappings['transmission'][transmission],
            'owners': owners,
            'oem': inverse_mappings['oem'][oem],
            'model': inverse_mappings['model'][model],
            'modelyear': modelyear,
            'registration_year': registration_year,
            'seats': seats,
            'engine_displacement': engine_displacement,
            'driver_air_bag': inverse_mappings['driver_air_bag'][driver_air_bag],
            'passenger_air_bag': inverse_mappings['passenger_air_bag'][passenger_air_bag],
            'rear_camera': inverse_mappings['rear_camera'][rear_camera],
            'mileage': mileage,
            'drive_type': inverse_mappings['drive_type'][drive_type],
            'city': inverse_mappings['city'][city],
            'car_roof': inverse_mappings['car_roof'][car_roof],
            'gears': gears,
            'front_brake_type_group': inverse_mappings['front_brake_type_group'][front_brake_type_group],
            'rear_brake_type_group': inverse_mappings['rear_brake_type_group'][rear_brake_type_group],
        }
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        try:
            prediction = xgboost_model.predict(input_df)
            st.success(f"### Estimated Price: **₹{prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")
