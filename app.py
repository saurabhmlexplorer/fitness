import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
#st.image("", use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg): ", 30, 150, 70)
    height = st.sidebar.slider("Height (cm): ", 100, 220, 170)
    bmi = weight / ((height / 100) ** 2)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": round(bmi, 2),
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")

st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")

st.write("---")
st.header("Personalized Recommendations: ")

recommendations = []

# Debugging: Print values to ensure they are being read correctly
st.write(f"**BMI:** {df['BMI'].values[0]}")
st.write(f"**Heart Rate:** {df['Heart_Rate'].values[0]}")
st.write(f"**Duration:** {df['Duration'].values[0]}")
st.write(f"**Body Temperature:** {df['Body_Temp'].values[0]}")

# BMI Analysis
bmi_value = df["BMI"].values[0]
if bmi_value < 18.5:
    recommendations.append("🔹 Your BMI suggests you are underweight. Consider a balanced diet with calorie-dense, nutritious foods to maintain energy levels.")
elif 18.5 <= bmi_value < 24.9:
    recommendations.append("✅ Your BMI is in the healthy range! Maintain it with a good balance of nutrition and exercise.")
elif 25 <= bmi_value < 29.9:
    recommendations.append("⚠️ Your BMI indicates that you are overweight. Try incorporating more cardio workouts and a balanced diet to reduce excess weight.")
else:
    recommendations.append("⚠️ Your BMI falls in the obesity range. Regular physical activity and a monitored diet plan may help improve your fitness levels.")

# Heart Rate Analysis
heart_rate_value = df["Heart_Rate"].values[0]
if heart_rate_value > 100:
    recommendations.append("🔴 Your heart rate is higher than average during exercise. Consider reducing intensity or consulting a doctor if it remains elevated.")
elif heart_rate_value < 70:
    recommendations.append("🟢 Your heart rate is lower than average. Ensure you are engaging in effective workouts to raise cardiovascular endurance.")
else:
    recommendations.append("✅ Your heart rate is within the normal range for exercise.")

# Exercise Duration Analysis
duration_value = df["Duration"].values[0]
if duration_value < 10:
    recommendations.append("🟠 Your workout duration is quite short. Aim for at least **30 minutes of moderate exercise** per session for better results.")
elif duration_value > 30:
    recommendations.append("✅ Great job! You are meeting the recommended exercise duration. Make sure to stay hydrated and allow time for recovery.")
else:
    recommendations.append("👍 Your workout duration is good. Keep it up!")

# Body Temperature Analysis
body_temp_value = df["Body_Temp"].values[0]
if body_temp_value > 39:
    recommendations.append("⚠️ Your body temperature is quite high. Stay hydrated, avoid overheating, and rest when necessary.")
elif body_temp_value < 36.5:
    recommendations.append("🔵 Your body temperature is lower than normal. Make sure to warm up properly before workouts.")
else:
    recommendations.append("✅ Your body temperature is in the normal range.")

# Display All Recommendations
for rec in recommendations:
    st.write(rec)

