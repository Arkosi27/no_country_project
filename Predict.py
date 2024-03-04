# app.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib
import streamlit as st

# Load the dataset
df_pred = pd.read_csv('datasets/df_pred.csv')

# Convert non-numeric values in the 'Month' column to numerical values
df_pred['Month'] = pd.to_datetime(df_pred['Month'], format='%B').dt.month

# Renaming the Name State column into Sname for easy Handling
df_pred = df_pred.rename(columns={'State Name': 'Sname'})

# Drop columns
cols_to_drop = ['Period', 'Indicator', 'State', 'Data Value', 'Footnote Symbol', 'Predicted Value', 'Footnote']
df_pred = df_pred.drop(cols_to_drop, axis=1)

# Select relevant features, excluding "Data Value"
features_to_use = ['Year', 'Month', 'Percent Complete', 'Percent Pending Investigation', 'Sname']
X = df_pred[features_to_use]

# Encode categorical feature 'Sname'
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X = pd.get_dummies(X)

# Train-test split (for potential future evaluation)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train a model (Random Forest Regressor)
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Save the trained model
model_path = "Models/random_forest_model.joblib"
joblib.dump(rf_model, model_path)

# Streamlit app
def main():
    st.title("Drug Overdose Death Prediction App")
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Show Instructions", "Run the Prediction"])
    if app_mode == "Show Instructions":
        st.markdown("## Instructions")
        st.markdown("This app predicts drug overdose death counts based on selected features.")
    elif app_mode == "Run the Prediction":
        st.markdown("## Prediction")
        # Get input values
        year = st.number_input("Enter the year", min_value=2015, max_value=2022, value=2022)
        month = st.selectbox("Select the month", range(1, 13), format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
        percent_complete = st.slider("Percent Complete", 0, 100, 50)
        percent_pending = st.slider("Percent Pending Investigation", 0, 100, 50)
        state = st.selectbox("Select the state", df_pred['Sname'].unique())

        # Prepare input data for prediction
        input_data = {'Year': year, 'Month': month, 'Percent Complete': percent_complete, 'Percent Pending Investigation': percent_pending, 'Sname': state}
        input_df = pd.DataFrame([input_data])

        # Encode input data
        input_df_encoded = pd.get_dummies(input_df)

        # Load the trained model
        loaded_model = joblib.load(model_path)

        # Make prediction
        prediction = loaded_model.predict(input_df_encoded)[0]

        st.write(f"Predicted drug overdose death count: {prediction}")

if __name__ == "__main__":
    main()
