import streamlit as st
import pandas as pd
import joblib

# Load Regression Model
revenue_model = joblib.load("revenue_prediction_model.pkl")
revenue_scaler = joblib.load("revenue_scaler.pkl")

# Load Classification Model
profit_model = joblib.load("profit_prediction_model.pkl")
classification_scaler = joblib.load("classification_scaler.pkl")

# Load MultiLabelBinarizer
mlb = joblib.load("mlb.pkl")

training_columns = joblib.load("training_columns.pkl")

# -----------------------------
# Campaign Input Section
# -----------------------------

st.title("📢 Marketing Campaign Revenue Prediction")

duration = st.number_input("Duration", min_value=1)

impressions = st.number_input("Impressions", min_value=0)

clicks = st.number_input("Clicks", min_value=0)

leads = st.number_input("Leads", min_value=0)

conversions = st.number_input("Conversions", min_value=0)

acquisition_cost = st.number_input("Acquisition Cost", min_value=0.0)

engagement_score = st.number_input("Engagement Score", min_value=0.0)


campaign_type = st.selectbox(
    "Campaign Type",
    ["Influencer", "Paid Ads", "SEO", "Email"]
)

target_audience = st.selectbox(
    "Target Audience",
    ["Men", "Women", "Youth"]
)

language = st.selectbox(
    "Language",
    ["English", "Tamil", "Hindi"]
)

customer_segment = st.selectbox(
    "Customer Segment",
    ["Premium", "Regular"]
)

brand = st.selectbox(
    "Brand",
    ["Purplle", "Tira"]
)

channel = st.selectbox(
    "Marketing Channel",
    list(mlb.classes_)
)


predict = st.button("Predict")


# -----------------------------
# Preprocessing
# -----------------------------

if predict:

    input_data = pd.DataFrame({

        "Duration": [duration],
        "Impressions": [impressions],
        "Clicks": [clicks],
        "Leads": [leads],
        "Conversions": [conversions],
        "Acquisition_Cost": [acquisition_cost],
        "Engagement_Score": [engagement_score],

        "Campaign_Type": [campaign_type],
        "Target_Audience": [target_audience],
        "Language": [language],
        "Customer_Segment": [customer_segment],
        "Brand": [brand]

    })


    # One Hot Encoding (same as training)
    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "Campaign_Type",
            "Target_Audience",
            "Language",
            "Customer_Segment",
            "Brand"
        ],
        drop_first=True
    )


    # MultiLabel Encoding for Channel

    channel_encoded = pd.DataFrame(
        mlb.transform([[channel]]),
        columns=mlb.classes_
    )


    # Combine all features

    input_processed = pd.concat(
        [
            input_encoded.reset_index(drop=True),
            channel_encoded.reset_index(drop=True)
        ],
        axis=1
    )


    st.subheader("Processed Input")
    st.dataframe(input_processed)


    # Scale

    input_processed = input_processed.reindex(
    columns=training_columns,
    fill_value=0
    )

    input_scaled = revenue_scaler.transform(input_processed)

# Revenue Prediction
    predicted_revenue = revenue_model.predict(input_scaled)

    st.subheader("💰 Predicted Revenue")

    st.success(f"₹ {predicted_revenue[0]:,.2f}")
    st.subheader("Scaled Input")
    st.write(input_scaled)

# Profit/Loss Prediction

    classification_input_scaled = classification_scaler.transform(input_processed)

    profit_prediction = profit_model.predict(classification_input_scaled)

    st.subheader("📊 Predicted Profit/Loss")

    if profit_prediction[0] == 1:
        st.success("✅ PROFIT Campaign")
    else:
        st.error("❌ LOSS Campaign")

    st.subheader("📈 Campaign Input Visualization")

    chart_data = pd.DataFrame({
        "Metric": [
            "Impressions",
            "Clicks",
            "Leads",
            "Conversions"
        ],
        "Value": [
            impressions,
            clicks,
            leads,
            conversions
        ]
    })

    st.bar_chart(
        chart_data.set_index("Metric")
    )