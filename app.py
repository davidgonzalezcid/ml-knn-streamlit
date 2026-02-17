import os
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
    # Modelo
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Lista de productos (opcional)
    products = None
    if os.path.exists("products.json"):
        with open("products.json", "r", encoding="utf-8") as f:
            products = json.load(f)

    return model, products

st.set_page_config(page_title="Predicción KNN", layout="wide")
st.title("Predicción multi-salida (KNN)")
st.write("Input: Producto, EPA, DHA. Output: variables de D1/D2/R2.")

model, products = load_artifacts()

# ----------------------------
# Inputs
# ----------------------------
st.sidebar.header("Inputs")

# Producto: dropdown si existe products.json, si no, texto
if products and isinstance(products, list) and len(products) > 0:
    producto = st.sidebar.selectbox("Producto", options=products)
else:
    producto = st.sidebar.text_input("Producto", value="")
    st.sidebar.info("No se encontró products.json: ingresa el Producto manualmente.")

# EPA y DHA como enteros
epa = st.sidebar.number_input("EPA", min_value=0, value=0, step=1, format="%d")
dha = st.sidebar.number_input("DHA", min_value=0, value=0, step=1, format="%d")

run = st.sidebar.button("Predecir")

# ----------------------------
# Predicción
# ----------------------------
if run:
    if str(producto).strip() == "":
        st.error("Debes ingresar un Producto.")
    else:
        X_new = pd.DataFrame([{
            "Producto": str(producto).strip(),
            "EPA": int(epa),
            "DHA": int(dha)
        }])

        pred = model.predict(X_new)[0]
        out = pd.Series(pred, index=TARGETS)

        # Regla de consistencia
        r2_calc = float(100 - out["D1%"] - out["D2%"])

        # Tabla 1: Fraccionamiento
        df_frac = pd.DataFrame([{
            "Producto": str(producto).strip(),
            "D1%": float(out["D1%"]),
            "D2%": float(out["D2%"]),
            "R2%": r2_calc
        }])

        # Tabla 2: Corriente D1 (renombrado a EPA/DHA)
        df_d1 = pd.DataFrame([{
            "EPA": float(out["D1 EPA"]),
            "DHA": float(out["D1 DHA"])
        }])

        # Tabla 3: Corriente D2
        df_d2 = pd.DataFrame([{
            "EPA": float(out["D2 EPA"]),
            "DHA": float(out["D2 DHA"])
        }])

        # Tabla 4: Corriente R2
        df_r2 = pd.DataFrame([{
            "EPA": float(out["R2 EPA"]),
            "DHA": float(out["R2 DHA"])
        }])

        # ----------------------------
        # Render resultados (sin índice y en vertical)
        # ----------------------------
        st.subheader("Resultados")

        st.markdown("#### Fraccionamiento")
        st.table(df_frac.reset_index(drop=True).style.hide(axis="index"))

        st.markdown("#### Corriente D1")
        st.table(df_d1.reset_index(drop=True).style.hide(axis="index"))

        st.markdown("#### Corriente D2")
        st.table(df_d2.reset_index(drop=True).style.hide(axis="index"))

        st.markdown("#### Corriente R2")
        st.table(df_r2.reset_index(drop=True).style.hide(axis="index"))
