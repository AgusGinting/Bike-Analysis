import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df_day = pd.read_csv("day.csv")

# Feature engineering
df_day['date'] = pd.to_datetime(df_day['dteday'], format='%Y-%m-%d')
df_day.drop(['instant', 'windspeed', 'weekday', 'holiday', 'workingday', 'dteday', 'registered', 'casual'], axis=1, inplace=True)
df_day['Year'] = pd.to_datetime(df_day['date']).dt.year
df_day['Month'] = pd.to_datetime(df_day['date']).dt.month
df_day['Day'] = pd.to_datetime(df_day['date']).dt.day
df_day.drop(columns=['date'], inplace=True)

# Split data
X = df_day.drop(columns=['cnt'])
Y = df_day['cnt']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Train Random Forest Classifier
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, Y_train)

# Save the model to file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest_model, f)

# Load the model from file
with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Streamlit App
st.title("Bike Rental Prediction")

# Sidebar
st.sidebar.subheader("Predict Bike Rentals")

# Inputs for prediction
season = st.sidebar.slider("Season", min_value=1, max_value=4, value=3)
yr = st.sidebar.slider("Year", min_value=2011, max_value=2012, value=2011)
mnth = st.sidebar.slider("Month", min_value=1, max_value=12, value=6)
holiday = st.sidebar.selectbox("Holiday", [0, 1], index=0)
weekday = st.sidebar.slider("Weekday", min_value=0, max_value=6, value=3)
workingday = st.sidebar.selectbox("Workingday", [0, 1], index=1)
weathersit = st.sidebar.slider("Weathersit", min_value=1, max_value=4, value=2)
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
hum = st.sidebar.slider("Humidity", min_value=0.0, max_value=1.0, value=0.5)
windspeed = st.sidebar.slider("Windspeed", min_value=0.0, max_value=1.0, value=0.5)

# Predict function
def predict_rentals(season, yr, mnth, holiday, weekday, workingday, weathersit, temp, hum, windspeed):
    input_data = pd.DataFrame({
        'season': [season],
        'yr': [yr],
        'mnth': [mnth],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'hum': [hum],
        'windspeed': [windspeed]
    })
    prediction = loaded_model.predict(input_data)
    return prediction[0]

# Prediction
if st.sidebar.button("Predict"):
    prediction = predict_rentals(season, yr, mnth, holiday, weekday, workingday, weathersit, temp, hum, windspeed)
    st.subheader(f"Predicted Bike Rentals: {prediction}")

# About
st.sidebar.subheader("About")
st.sidebar.write("This app predicts the number of bike rentals based on user input.")

# Show original data
st.subheader("Original Data")
st.write(df_day)

# Show correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_day.corr(), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt.gcf())
