import streamlit as st
import pandas as pd
import joblib

# Load the trained logistic regression model
model = joblib.load('logistic_model.pkl')

# Title of the app
st.title("Titanic Survival Prediction")

# Input fields for user data
st.header("Enter Passenger Information")

passenger_id = st.number_input("Passenger ID", min_value=1, value=1)
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=7.25)
embarked = st.selectbox("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)", ["C", "Q", "S"])

# Encode categorical variables exactly as done during training
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create a DataFrame for the input data with all necessary features
input_data = pd.DataFrame({
    'PassengerId': [passenger_id],
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'male': [sex_male],
    'Q': [embarked_Q],
    'S': [embarked_S]
})

# Make prediction when button is clicked
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The passenger is likely to survive.")
        else:
            st.error("The passenger is unlikely to survive.")
    except ValueError as e:
        st.error(f"Prediction error: {e}")
