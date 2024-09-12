import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_combined_cars.csv')

# Train the model
@st.cache_resource
def train_model(data):
    features = ['Kilometers_Driven', 'Registration_Year', 'Mileage', 'Engine_Power', 'City']
    X = data[features]
    y = data['Price']

    # Preprocessing for numerical data
    numerical_features = ['Kilometers_Driven', 'Registration_Year', 'Mileage', 'Engine_Power']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_features = ['City']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define model (using RandomForestRegressor)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(random_state=42))])

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

    # Train the model
    model, mse, r2, features = train_model(data)

    # Input options in the center of the page
    st.subheader("Input Details")
    car_make = st.selectbox("Select Car Make", options=data['Car_Model'].unique())
    fuel_type = st.selectbox("Select Fuel Type", options=data['Fuel_Type'].unique())
    
    # Ensure that the registration year is cast as an integer
    registration_year = int(st.selectbox("Select Registration Year", options=data['Registration_Year'].unique().astype(int)))
    
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
        'Registration_Year': [registration_year],  # Ensure year is integer
        'Engine_Power': [engine_power],
        'Mileage': [mileage],
        'City': [city]
    })

    # Add a button for prediction
    if st.button("Predict Car Price"):
        # Make prediction
        predicted_price = model.predict(input_data)[0]

        # Display the prediction result with larger, bold font size
        st.markdown(f"<h1 style='text-align: center; color: black; font-size: 32px;'><b>Predicted Car Price: ₹{predicted_price:,.2f}</b></h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
