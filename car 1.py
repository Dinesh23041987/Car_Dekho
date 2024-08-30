import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache_data
def load_data():
    combined_df = pd.read_csv('cleaned_combined_cars.csv')
    return combined_df

# Train a model
@st.cache_resource
def train_model(data):
    # Selecting features and target variable
    features = ['Kilometers_Driven', 'Registration_Year', 'Mileage', 'Engine_Power', 'Car_Model', 'City']
    X = data[features]
    y = data['Price']

    # Handle missing values and encode categorical variables
    numeric_features = ['Kilometers_Driven', 'Registration_Year', 'Mileage', 'Engine_Power']
    categorical_features = ['Car_Model', 'City']

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that first transforms the data and then fits the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, features

def main():
    st.title("Used Car Price Prediction")

    # Load data
    data = load_data()

    # Input options in the center of the page
    st.subheader("Input Details")
    car_make = st.selectbox("Select Car Make", options=data['Car_Model'].unique())
    fuel_type = st.selectbox("Select Fuel Type", options=data['Fuel_Type'].unique())
    registration_year = st.selectbox("Select Registration Year", options=data['Registration_Year'].unique())
    engine_power = st.number_input("Engine Power (in bhp)", min_value=30, max_value=500, value=100)
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=10000)
    city = st.selectbox("Select City", options=data['City'].unique())

    # Calculate the average mileage for the selected car make and city
    mileage = data.loc[
        (data['Car_Model'] == car_make) & 
        (data['City'] == city) & 
        (data['Registration_Year'] == registration_year), 'Mileage'
    ]

    if mileage.empty:
        mileage = data['Mileage'].median()
    else:
        mileage = mileage.mean()

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Kilometers_Driven': [kilometers_driven],
        'Registration_Year': [registration_year],
        'Engine_Power': [engine_power],
        'Mileage': [mileage],
        'Car_Model': [car_make],
        'City': [city]
    })

    # Train the model
    model, mse, r2, features = train_model(data)

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Display the prediction result with larger, bold font size
    st.markdown(f"<h1 style='text-align: center; color: black; font-size: 32px;'><b>Predicted Car Price: ₹{predicted_price:,.2f}</b></h1>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
