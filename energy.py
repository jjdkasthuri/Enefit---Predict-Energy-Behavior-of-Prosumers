import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model_path = r'C:\Users\ASUS\Downloads\new_model/model_test.h5'
model = load_model(model_path)

# Initialize scaler with example parameters (mean and std)
mean = np.array([0.0] * 8)  # Example means for all 8 features
std = np.array([1.0] * 8)   # Example standard deviations for all 8 features

scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = std
scaler.n_features_in_ = len(mean)

# Function to render the home page
def home():
    st.title("Welcome to the Energy Prediction App")
    
    st.markdown(
        """
        <style>
        .home-main {
            background-image: url("https://etimg.etb2bimg.com/photo/105585282.cms");
            background-size: cover;
            background-position: center;
            min-height: 400px; /* Set minimum height */
            padding: 40px;
            border-radius: 50px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .about-section {
            margin-top: 20px;
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 10px;
        }
        </style>
        <div class="home-main">
        
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="about-section">
            <h2>About This App</h2>
            <p>This app helps you predict energy usage and determine eligibility for becoming a prosumer.</p>
            <p>Use the sidebar to navigate to the prediction tool.</p>
            <h3>Features:</h3>
            <ul>
                <li>Predict energy values based on user inputs</li>
                <li>Compare predicted values with actual ground truth</li>
                <li>Determine eligibility for further consideration as a prosumer</li>
            </ul>
            <p>Enefit is one of the biggest energy companies in Baltic region. As experts in the field of energy, we help customers plan their green journey in a personal and flexible manner as well as implement it by using environmentally friendly energy solutions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Function to render the prediction page
def prediction():
    # Apply background image and style using custom HTML/CSS
    st.markdown(
        """
        <style>
        .main {
            background-image: url("https://mltfwbciccuo.i.optimole.com/cb:n4OZ.4d52/w:auto/h:auto/q:mauto/f:best/https://www.evalueserve.com/wp-content/uploads/2023/06/renewables_coverphoto.jpg");
            background-position: center;
            padding: 20px;
            border-radius: 50px;
            background-repeat: no-repeat;
            background-attachment: fixed;
           
            background-size: cover;
            background-position: center;
            min-height: 700px; /* Set minimum height */
        }
        .input-box {
            background-color: rgba(255, 255, 255, 0.8); /* White background with transparency */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        
        </style>
        <div class="main">
        """,
        unsafe_allow_html=True,
    )

    st.title("Energy Prediction")

    st.markdown(
        """
 
        """,
        unsafe_allow_html=True,
    )

    # Input fields for energy amount, time step, and ground truth
    energy_amount = st.number_input("Energy Amount", min_value=0.0, format="%.2f", step=0.01)
    time_step = st.number_input("Time Step", min_value=0.0, format="%.2f", step=0.01)
    ground_truth = st.number_input("Ground Truth", min_value=0.0, format="%.2f", step=0.01)

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    if st.button("Predict"):
        try:
            # Prepare the input data for the model
            input_data = pd.DataFrame([[energy_amount, time_step, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
                                      columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8'])
            
            # Standardize the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Make a prediction
            prediction = model.predict(input_data_scaled)[0][0]  # Assuming the model outputs a single prediction
            
            # Display the results
            st.write(f"Ground Truth Value: {ground_truth:.2f}")
            st.write(f"Predicted Energy Value: {prediction:.2f}")
            
            # Check condition
            if ground_truth > prediction:
                st.write("Eligible for further consideration to become a prosumer.")
            else:
                st.write("Not eligible for further consideration.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Navigation menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# Page rendering
if page == "Home":
    home()
elif page == "Prediction":
    prediction()
