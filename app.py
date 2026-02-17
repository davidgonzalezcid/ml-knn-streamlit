import json
import pickle
import pandas as pd
import streamlit as st

TARGETS = [
    "D1%", "D2%",
    "D1 EPA", "D1 DHA",
    "D2 EPA", "D2 DHA",
    "R2 EPA", "R2 DHA"
]

@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    return model, products

st.set_page_config(page_title="Predicción KNN", layout="centered")
st.title("Predicción multi-salida (KNN)")
st.write("Input: Producto, EPA, DHA. Output: variables de D1/D2/R2.")

model, products = load_artifacts()

st.sidebar.header("Inputs")
producto = st.sidebar.selectbox("Producto", options=products)
epa = st.sidebar.number_input("EPA", value=0.0, step=1.0, format="%.4f")
dha = st.sidebar.number_input("DHA", value=0.0, step=1.0, format="%.4f")

run = st.sidebar.button("Predecir")

if run:
    X_new = pd.DataFrame([{"Producto": producto, "EPA": epa, "DHA": dha}])
    pred = model.predict(X_new)[0]
    out = pd.DataFrame([pred], columns=TARGETS)

    out["R2_calc_pred"] = 100 - out["D1%"] - out["D2%"]
    out.insert(0, "Producto", producto)
    out.insert(1, "EPA_in", epa)
    out.insert(2, "DHA_in", dha)

    st.subheader("Resultados")
    st.dataframe(out, use_container_width=True)
