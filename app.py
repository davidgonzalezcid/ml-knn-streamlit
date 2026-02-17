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
def load_model():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

st.set_page_config(page_title="Predicción KNN", layout="wide")
st.title("Predicción multi-salida (KNN)")
st.write("Input: Producto, EPA, DHA. Output: variables de D1/D2/R2.")

model = load_model()

# ----------------------------
# Inputs
# ----------------------------
st.sidebar.header("Inputs")

producto = st.sidebar.text_input("Producto", value="")

epa = st.sidebar.number_input("EPA", min_value=0, value=0, step=1, format="%d")
dha = st.sidebar.number_input("DHA", min_value=0, value=0, step=1, format="%d")

run = st.sidebar.button("Predecir")

# ----------------------------
# Predicción
# ----------------------------
if run:
    if producto.strip() == "":
        st.error("Debes ingresar un Producto.")
    else:
        X_new = pd.DataFrame([{"Producto": producto.strip(), "EPA": int(epa), "DHA": int(dha)}])
        pred = model.predict(X_new)[0]
        out = pd.Series(pred, index=TARGETS)

        r2_calc = float(100 - out["D1%"] - out["D2%"])

        df_frac = pd.DataFrame([{
            "Producto": producto.strip(),
            "D1%": float(out["D1%"]),
            "D2%": float(out["D2%"]),
            "R2%": r2_calc
        }])

        df_d1 = pd.DataFrame([{
            "EPA": float(out["D1 EPA"]),
            "DHA": float(out["D1 DHA"])
        }])

        df_d2 = pd.DataFrame([{
            "EPA": float(out["D2 EPA"]),
            "DHA": float(out["D2 DHA"])
        }])

        df_r2 = pd.DataFrame([{
            "EPA": float(out["R2 EPA"]),
            "DHA": float(out["R2 DHA"])
        }])

        st.subheader("Resultados")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Tabla 1. Fraccionamiento")
            st.dataframe(df_frac, use_container_width=True)

        with c2:
            st.markdown("#### Tabla 2. Corriente D1")
            st.dataframe(df_d1, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Tabla 3. Corriente D2")
            st.dataframe(df_d2, use_container_width=True)

        with c4:
            st.markdown("#### Tabla 4. Corriente R2")
            st.dataframe(df_r2, use_container_width=True)
