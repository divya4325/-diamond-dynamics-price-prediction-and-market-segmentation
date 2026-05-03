import streamlit as st
import pandas as pd
import joblib
import cluster
@st.cache_resource
def load_artifacts():
 model = joblib.load("models/xgb_final.pkl")
 scaler = joblib.load("models/scaler_feature.pkl")
 encoder = joblib.load("models/onehotencoder_diamond.pkl")
 ordinal_encoder = joblib.load("models/ordinalencoder_diamond.pkl")
 features = joblib.load("models/model_final_feature.pkl")
 return model, scaler, encoder, ordinal_encoder, features

model, scaler, encoder, ordinal_encoder, features = load_artifacts()
st.set_page_config(
    page_title="Diamond Market Analysis"
)
st.sidebar.title("Navigator")
page=st.sidebar.radio(
    "Go to",
    [
        "Price Prediction Module Diamond",
        "Market Segement Prediction(Clustering Module)"
    ]
    
)
if page=="Price Prediction Module Diamond":
    st.title("💎 Diamond Price Prediction")
    st.subheader("💎 Enter Diamond Details")
    volume=st.number_input("Volume",min_value=1.0,value=100.0)
    carat=st.number_input("Carat",min_value=0.20,value=1.0)
    depth=st.number_input("Depth",min_value=40.0,max_value=80.0,value=61.0)
    table = st.number_input("Table", min_value=43.0, max_value=93.0, value=57.0)
    cut=st.selectbox("Cut",["Fair","Good","Very Good","Premium","Ideal"])
    color=st.selectbox("Color",["D","E","F","G","H","I","J"])
    clarity=st.selectbox("Clarity",["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
    if st.button("Predict Price"):

      input_df = pd.DataFrame({
        'volume': [volume],
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity]
       })

    # ---------------- FEATURE ENGINEERING ----------------
      def get_carat_category(carat):
        if carat <= 1:
            return "Light"
        elif carat <= 2:
            return "Medium"
        else:
            return "Large"

      input_df["carat_category"] = input_df["carat"].apply(get_carat_category)

      input_df["carat_category"] = ordinal_encoder.transform(
        input_df[["carat_category"]]
      )

    # ---------------- ONE HOT ENCODING (IMPORTANT FIX) ----------------
      encoded = encoder.transform(input_df[["cut", "color", "clarity"]])

      encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["cut", "color", "clarity"])
      )

    # ---------------- NUMERIC ----------------
      numeric_df = input_df[["volume", "carat", "depth", "table", "carat_category"]].reset_index(drop=True)

    # ---------------- MERGE ----------------
      final_input = pd.concat([numeric_df, encoded_df], axis=1)

    # ---------------- ⭐ CRITICAL FIX ----------------
    # THIS is the ONLY safe alignment method
      final_input = final_input.reindex(columns=features, fill_value=0)

    # ---------------- SCALE ----------------
      final_scaled = scaler.transform(final_input)

    # ---------------- PREDICT ----------------
      prediction = model.predict(final_scaled)[0]
      USD_TO_INR = 83  # approximate

      prediction_inr = prediction * USD_TO_INR


      st.success(f"💎 Predicted Price: Rs. {prediction_inr:,.2f}")
elif page == "Market Segement Prediction(Clustering Module)":
    cluster.show()       