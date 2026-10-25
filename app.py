import pickle

import pandas as pd
import streamlit as st

MODEL_FILE = "property_price_model_full_dataset.pkl"

st.set_page_config(page_title="Property Price Predictor", page_icon="🏠")
st.title("🏠 Pakistan Property Price Predictor")
st.write("Estimate the price of a property (Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad) "
         "using a machine-learning model trained on Zameen.com listings.")


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_locations():
    return pd.read_csv("locations.csv")


model = load_model()
locations = load_locations()

purpose = st.radio("Purpose", ["For Sale", "For Rent"], horizontal=True)

c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("City", sorted(locations["city"].unique()))
    city_locations = locations[locations["city"] == city]
    location = st.selectbox("Location / Society", city_locations["location"].tolist())
    property_type = st.selectbox(
        "Property type",
        ["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House", "Penthouse"],
    )
with c2:
    unit = st.selectbox("Area unit", ["Marla", "Kanal", "Square feet"])
    area_value = st.number_input(f"Area ({unit})", min_value=0.5, value=10.0 if unit == "Marla" else 1.0, step=0.5)
    bedrooms = st.number_input("Bedrooms", 0, 15, 3)
    baths = st.number_input("Bathrooms", 0, 15, 3)

# Zameen convention used in the dataset: 1 Marla = 272.25 sq ft, 1 Kanal = 20 Marla
area_sqft = area_value * {"Marla": 272.25, "Kanal": 5445.0, "Square feet": 1.0}[unit]
row_loc = city_locations[city_locations["location"] == location].iloc[0]

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "latitude": row_loc["latitude"],
        "longitude": row_loc["longitude"],
        "baths": baths,
        "bedrooms": bedrooms,
        "area_sqft": area_sqft,
        "property_type": property_type,
        "city": city,
        "purpose": purpose,
        "location": location,
    }])
    price = float(model.predict(row)[0])
    label = "monthly rent" if purpose == "For Rent" else "price"
    st.success(f"Estimated {label}: **Rs {price:,.0f}**  (about Rs {price / 1e6:,.2f} million)")
    st.caption("Latitude/longitude are taken from the selected location. "
               "Predictions are estimates based on 2018-2019 listings and can be off by a large margin for luxury properties.")
