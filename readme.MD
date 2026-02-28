# 🌊 Florida Flood Damage Prediction (NFIP Data)

An end-to-end machine learning project predicting building damage ratios using FEMA's National Flood Insurance Program (NFIP) claims data.

![FEMA Risk App Demo](./project_demo.gif)

## 🚀 Project Overview
This project uses FEMA data to predict building damage ratios for Florida properties at risk of flooding. Key challenges found were: data imbalance with many zero damage findings, high missingness (60%), and physical data anomalies.**

## Key Technical Achievements
* Advanced Data Sanitization: Conducted a forensic audit of FEMA NRI records to identify and isolate physically impossible values (e.g., negative water depths/elevations), ensuring baseline central tendencies remained unbiased for imputation.

* Targeted Feature Engineering: Derived high-signal variables including age_of_property. SHAP validation confirmed these as top-tier predictors of structural resilience.

* Predictive Optimization: Leveraged GridSearchCV to tune a Random Forest Regressor, prioritizing Mean Absolute Error (MAE) to ensure the model's error margins are interpretable in real-world insurance/financial contexts.

* Full-Stack Deployment: Engineered a Streamlit dashboard that encapsulates the entire data pipeline—from raw user input to final loss estimation—enabling non-technical stakeholders to interact with the model and glean insights.

## 🛠️ Tech Stack
* **Modeling:** RandomForest Regressor
* **Interpretability:** SHAP
* **Deployment:** Streamlit, Joblib
* **Data Handling:** Pandas, NumPy, Scikit-Learn

## 🔍 Model Interpretability & Risk Drivers
The model was validated using SHAP (SHapley Additive exPlanations) to ensure that its logic aligns with the physical and economic realities of flood risk.

1. Mitigation Efficacy: The "Post-FIRM" Success
The model identified Post-FIRM construction (buildings erected after a community joined the NFIP) as one of the strongest mitigators of loss. Properties built to these modern standards consistently showed lower damage ratios, providing quantitative evidence that federal elevation requirements are successfully reducing financial vulnerability.

2. Building Vulnerability: The "Vintage" Factor
Property Age emerged as a critical vulnerability driver. The analysis showed a clear trend: older properties—built before modern flood-resistant materials and stricter coastal codes—experience significantly higher damage ratios when compared to newer structures under identical flood depths.

3. Economic Trends: Construction Inflation
A compelling "Year of Loss" trend was observed, showing that recent flood events result in higher relative payouts. This aligns with macroeconomic shifts, specifically the significant spike in construction labor and material costs over the last decade, which has pushed structural repair costs higher relative to original building values.

4. Physical Ground Truth: Water Depth & Geography
* Water Depth: The model correctly learned the fundamental physics of flooding; lower water depths consistently mapped to lower SHAP values (reduced risk), while increased depth accelerated damage exponentially.

* Flood Zone AE: Properties in the Special Flood Hazard Area (Zone AE) were penalized by the model with higher risk scores, while those in shaded X zones showed a consistently reduced risk profile.

## Data Source
This project utilizes the National Risk Index (NRI) dataset provided by FEMA.

[Link to FEMA NRI](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2)

## ✍️ Author
**Rossini Martyr** [https://www.linkedin.com/in/rossinimartyr/]


