import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and the metadata

try:
    model = joblib.load('flood_model_v1.pkl')
    meta = joblib.load('model_metadata.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are in the project directory.")

st.title("🌊 Florida Flood Risk Predictor")
st.markdown("""
This tool predicts the **Building Damage Ratio** (Claim Amount / Building Value) 
based on FEMA NFIP historical patterns. 
""")

# --- SIDEBAR INPUTS (TOP 5 SHAP FEATURES) ---
st.sidebar.header("Property Inputs")

# Age of Property
age = st.sidebar.number_input("Property Age", min_value=0, max_value=150, value=30)

# Year of Loss (Top 5 feature)
year_loss = st.sidebar.slider("Year of the Flood Event", 1978, 2024, 2024)

# Water Depth (Top 5 feature)
water_depth = st.sidebar.slider("Flood Depth above Ground (ft)", 0.0, 15.0, 2.0)

# Post-FIRM Indicator (Top 5 feature)
post_firm = st.sidebar.selectbox(
    "Was it built after the Flood Map (Post-FIRM)?", 
    options=[1, 0], 
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Flood Zone AE (Top 5 feature)
zone_ae = st.sidebar.selectbox(
    "Is the property in Flood Zone AE?", 
    options=[1, 0], 
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# --- PREDICTION LOGIC ---
if st.button("Generate Risk Assessment"):
    
    # Create a DataFrame with all columns used during training, initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=meta['all_features'])
    
    # 2. FILL BACKGROUND FEATURES 
    for feature, value in meta['defaults'].items():
        if feature in input_df.columns:
            input_df[feature] = value
    
    # 3. OVERWRITE WITH USER INPUTS (From Sidebar)
    input_df['age_of_property'] = age
    input_df['yearOfLoss'] = year_loss
    input_df['waterDepth'] = water_depth
    input_df['postFIRMConstructionIndicator'] = post_firm
    input_df['floodZone_grouped_AE'] = zone_ae

    # 4. RUN PREDICTION
    prediction = model.predict(input_df)[0]
    
    # Bound the prediction between 0 and 100%
    display_pred = max(0, min(prediction, 1.0))

    # 5. DISPLAY RESULTS
    st.subheader("Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Damage Ratio", f"{display_pred:.2%}")
    
    with col2:
        risk_level = "High" if display_pred > 0.4 else "Moderate" if display_pred > 0.15 else "Low"
        st.write(f"**Risk Level:** {risk_level}")

    # Interpretation based on SHAP findings
    st.info(f"""
    **Insight:** A damage ratio of {display_pred:.2%} suggests that for a house valued at $300k, 
    estimated structural losses would be around ${300000 * display_pred:,.0f}.
    """)