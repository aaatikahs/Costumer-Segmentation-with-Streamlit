import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Load the trained Random Forest model
model = joblib.load("Random Forest_model.pkl")

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Classification")
st.markdown("This Streamlit application performs customer segmentation in the automobile industry using [Kaggle's Customer Segmentation Classification Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation?select=Train.csv).")
st.markdown("It uses a trained Random Forest model to predict customer segments (A, B, C, and D) based on user input. The definitions of these segments are as follows: ")

# Load segment descriptions
with open("descriptions.json") as f:
    segment_descriptions = json.load(f)
segment_descriptions = pd.DataFrame(segment_descriptions.values(), index=segment_descriptions.keys(), columns=["description"])
st.table(segment_descriptions)

st.markdown("These descriptions are **by no means definitive**. They are simply the result of Exploratory Data Analysis (EDA) process.")

# User input form
st.subheader("Input Customer Data")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, step=1, value=30)
    married = st.selectbox("Have you ever been married?", ["Yes", "No"])
    work_experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, step=1, value=5)

with col2:
    graduated = st.selectbox("Have you ever graduated?", ["Yes", "No"])
    profession = st.selectbox("What is your profession?", ["Artist", "Doctor", "Engineer", "Entertainment", "Executive", 
                                              "Healthcare", "Homemaker", "Lawyer", "Marketing", "Undefined", "Unemployed"])
    gender = st.selectbox("What is your gender?", ["Female", "Male"])

with col3:
    spending_score = st.selectbox("How big of a spender on automobiles are you?", ["Low", "Average", "High"])
    family_size = st.number_input("Number of family members (including the customer)", min_value=1, max_value=20, step=1, value=3)

# Add Submit button
if st.button("Submit"):
    # Define label and one-hot encoding mappings
    label_encodings = {
        "Ever_Married": {"Yes": 1, "No": 0},
        "Graduated": {"Yes": 1, "No": 0},
        "Spending_Score": {"Low": 2, "Average": 0, "High": 1}
    }

    # Process user inputs
    inputs = {
        "Ever_Married": label_encodings["Ever_Married"][married],
        "Graduated": label_encodings["Graduated"][graduated],
        "Spending_Score": label_encodings["Spending_Score"][spending_score],
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        **{f"Profession_{p}": 1 if profession == p else 0 for p in ["Artist", "Doctor", "Engineer", "Entertainment", 
                                                                   "Executive", "Healthcare", "Homemaker", "Lawyer", 
                                                                   "Marketing", "Undefined", "Unemployed"]},
        "Age": age,
        "Work_Experience": work_experience,
        "Family_Size": family_size
    }

    # Convert inputs to a DataFrame
    input_df = pd.DataFrame([inputs])

    # Load the feature names used during training
    scaler = joblib.load("scaler.pkl")  # Ensure this scaler was fitted with consistent column order
    trained_feature_names = scaler.feature_names_in_

    # Reorder columns to match the feature names used during training
    input_df = input_df[trained_feature_names]

    # Scale the inputs
    scaled_inputs = scaler.transform(input_df)

    # Convert scaled inputs to a DataFrame for prediction
    scaled_df = pd.DataFrame(scaled_inputs, columns=input_df.columns)

    # Map the one-hot encoding back to segment labels
    segment_mapping = {
    "Segmentation_A": "A",
    "Segmentation_B": "B",
    "Segmentation_C": "C",
    "Segmentation_D": "D"
    }

    # Predict the segmentation
    prediction = model.predict(scaled_df)
    # Ensure the prediction is 2D by squeezing any extra dimensions
    prediction = np.squeeze(prediction)  # Shape becomes (1, 4) or (4,)
    # Convert the prediction to a DataFrame
    segment_onehot = pd.DataFrame([prediction], columns=["Segmentation_A", "Segmentation_B", "Segmentation_C", "Segmentation_D"])
    segment = segment_onehot.idxmax(axis=1).iloc[0]  # Get the column name with the max value
    segment_label = segment_mapping[segment]  # Map to label (A, B, C, D)

    # Retrieve and display the description
    if segment_label in segment_descriptions.index:
        st.success(f"The customer belongs to Segment {segment_label}.")
        st.markdown(f"**Segment Description:** {segment_descriptions.loc[segment_label, 'description']}")
    else:
        st.error(f"Segment {segment_label} not found in segment descriptions.")
        st.markdown("Please check your model or descriptions file.")