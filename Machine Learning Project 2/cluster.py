import streamlit as st
import pandas as pd
import joblib
@st.cache_resource
def load_artifacts():
    try:
      model = joblib.load("cluster_model/kmeans_model (1).pkl")
      scaler = joblib.load("cluster_model/scaler_cluster (1).pkl")
      ordinal_encoder = joblib.load("cluster_model/encoder_cluster (1).pkl")
      features = joblib.load("cluster_model/clustering_feature_names.pkl")
      cluster_map=joblib.load("cluster_model/cluster_map (1).pkl")
      return model, scaler, ordinal_encoder, features,cluster_map
    except Exception as e:
        st.error(f"Loading error: {e}")
        return None, None, None, None, None
model, scaler, ordinal_encoder, features,cluster_map = load_artifacts()
def show():
    st.title(" Diamond Market Segmentation")
    st.subheader("💎 Enter Diamond Details")
    volume=st.number_input("Volume",min_value=1.0,value=100.0)
    carat=st.number_input("Carat",min_value=0.20,value=1.0)
    depth=st.number_input("Depth",min_value=40.0,max_value=80.0,value=61.0)
    table = st.number_input("Table", min_value=43.0, max_value=93.0, value=57.0)
    cut = st.selectbox("Cut", ordinal_encoder.categories_[0].tolist())
    color = st.selectbox("Color", ordinal_encoder.categories_[1].tolist())
    clarity = st.selectbox("Clarity", ordinal_encoder.categories_[2].tolist())
    if st.button("Predict Cluster"):

      input_df = pd.DataFrame({
        'volume': [volume],
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity]
       })
      input_df[["cut","color","clarity"]] = ordinal_encoder.transform(
          input_df[["cut","color","clarity"]])

      input_df=input_df.reindex(columns=features,fill_value=0)
      input_scaled=scaler.transform(input_df)
      cluster=model.predict(input_scaled)[0]
      cluster_name=cluster_map.get(cluster,"Unknown Cluster")
      st.success(f"Cluster ID {cluster}")
      st.info(f"Cluster Segement {cluster_name}")
      

 