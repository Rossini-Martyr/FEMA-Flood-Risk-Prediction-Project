import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. CUSTOM TRANSFORMER ---
# Must be defined before loading the model so joblib can find it
class DataPrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Identify boolean columns and convert to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        # Ensure only the required features are passed to the next pipeline step
        cols_to_keep = [
            'elevatedBuildingIndicator', 'waterDepth', 'occupancyType', 
            'age_of_property', 'yearOfLoss', 'elevationDifference', 
            'floodZoneCurrent', 'primaryResidenceIndicator', 
            'postFIRMConstructionIndicator'
        ]
        return X[cols_to_keep]

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('flood_model_v1.pkl')
    meta = joblib.load('model_metadata.pkl')
    return model, meta

try:
    model, meta = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 3. UI SETUP ---
st.set_page_config(page_title="Florida Flood Risk Predictor", layout="wide")
st.title("🌊 Florida Flood Risk Predictor")
st.markdown("""
This tool predicts the **Building Damage Ratio** (Claim Amount / Building Value) 
based on FEMA NFIP historical patterns. 
""")

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("Property & Financial Details")

# Financial Input
prop_value = st.sidebar.number_input(
    "Structure Replacement Value ($)", 
    min_value=50000, 
    max_value=5000000, 
    value=1000000, 
    step=50000,
    format="%d"
)

st.sidebar.divider()

# Numerical Inputs
age = st.sidebar.number_input("Property Age (Years)", 0, 150, int(meta['defaults'].get('age_of_property', 30)))
water_depth = st.sidebar.slider("Expected Flood Depth (ft)", 0.0, 15.0, float(meta['defaults'].get('waterDepth', 2.0)))
year_loss = st.sidebar.slider("Year of Loss", 1978, 2030, 2024)

# Categorical Dropdown for Flood Zone
selected_zone = st.sidebar.selectbox(
    "Select Flood Zone", 
    options=meta['flood_zones'],
    index=meta['flood_zones'].index(meta['defaults']['floodZoneCurrent']) 
)

# Binary Toggles (Dropdown style as requested)
post_firm = st.sidebar.selectbox("Post-FIRM Construction?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
primary_res = st.sidebar.selectbox("Primary Residence?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

# --- 5. PREDICTION LOGIC ---
if st.button("Generate Risk Assessment", type="primary"):
    # 1. Initialize with defaults (this handles 'occupancyType' and 'elevationDifference' automatically)
    input_df = pd.DataFrame([meta['defaults']])
    
    # 2. Overwrite defaults with user selections from sidebar
    input_df['age_of_property'] = age
    input_df['waterDepth'] = water_depth
    input_df['yearOfLoss'] = year_loss
    input_df['floodZoneCurrent'] = selected_zone
    input_df['postFIRMConstructionIndicator'] = post_firm
    input_df['primaryResidenceIndicator'] = primary_res
    
    # 3. Ensure column order matches the metadata/pipeline expectation
    input_df = input_df[meta['required_raw_features']]

    # 4. Predict
    prediction = model.predict(input_df)[0]
    display_pred = max(0.0, min(float(prediction), 1.0))
    estimated_loss = prop_value * display_pred

    # 5. Results UI
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    
    res_col1.metric("Predicted Damage Ratio", f"{display_pred:.2%}")
    res_col2.metric("Estimated Structural Loss", f"${estimated_loss:,.0f}")
    
    risk_level = "High" if display_pred > 0.4 else "Moderate" if display_pred > 0.15 else "Low"
    res_col3.markdown(f"**Risk Category:** {'🔴' if risk_level == 'High' else '🟡' if risk_level == 'Moderate' else '🟢'} {risk_level}")
 
else:
    st.write("Adjust settings in the sidebar and click the button to see the risk assessment.")