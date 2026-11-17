from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción Rate - Zomato",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Predicción de Calificación de Restaurantes (Zomato)")
st.write(
    "Compara **Random Forest**, **KNN (k=10)** y **Regresión Lineal** "
    "entrenados en Google Colab sobre `dataset_zomato_numerico`."
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def cargar_modelos():
    cargados = {}
    rutas_posibles = {
        "Random Forest": [
            BASE_DIR / "modelos entrenados" / "modelo_rf.joblib",
            BASE_DIR / "modelo_rf.joblib",
        ],
        "KNN (k=10)": [
            BASE_DIR / "modelos entrenados" / "modelo_knn.joblib",
            BASE_DIR / "modelo_knn.joblib",
        ],
        "Regresión Lineal": [
            BASE_DIR / "modelos entrenados" / "modelo_lr.joblib",
            BASE_DIR / "modelo_lr.joblib",
        ],
    }
    for nombre, candidatos in rutas_posibles.items():
        for ruta in candidatos:
            if ruta.exists():
                cargados[nombre] = joblib.load(ruta)
                break
    return cargados


modelos = cargar_modelos()

if not modelos:
    st.error(
        "No se encontraron los archivos `.joblib` en `modelos entrenados/` ni junto a `app.py`."
    )
    st.stop()

st.sidebar.header("Configuración")
modelo_elegido = st.sidebar.selectbox("Modelo principal:", list(modelos.keys()))

st.sidebar.subheader("Métricas en prueba (Colab)")
metricas_df = pd.DataFrame({
    "Modelo": ["Random Forest", "KNN (k=10)", "Regresión Lineal", "Línea base"],
    "RMSE": [0.1553, 0.3121, 0.3280, 0.4372],
    "MAE": [0.0869, 0.2257, 0.2350, 0.3486],
    "R²": [0.8738, 0.4903, 0.4370, -0.0001],
})
st.sidebar.dataframe(metricas_df, hide_index=True, use_container_width=True)

st.subheader("Datos del restaurante")
c1, c2, c3 = st.columns(3)

with c1:
    online_order = st.selectbox(
        "Pedido en línea (online_order)",
        options=[1, 0],
        format_func=lambda x: "Sí (1)" if x == 1 else "No (0)",
    )
    book_table = st.selectbox(
        "Reserva de mesa (book_table)",
        options=[1, 0],
        format_func=lambda x: "Sí (1)" if x == 1 else "No (0)",
    )
    log_votes = st.number_input(
        "Votos logarítmicos (log_votes)",
        min_value=0.0,
        max_value=20.0,
        value=4.50,
        step=0.10,
    )

with c2:
    cost = st.number_input(
        "Costo (cost)",
        min_value=0.0,
        max_value=20000.0,
        value=500.0,
        step=50.0,
    )
    n_cuisines = st.number_input(
        "Número de cocinas (n_cuisines)",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )
    location = st.number_input(
        "Código ubicación (location)",
        min_value=0,
        max_value=1000,
        value=10,
        step=1,
    )

with c3:
    rest_type = st.number_input(
        "Código tipo restaurante (rest_type)",
        min_value=0,
        max_value=1000,
        value=15,
        step=1,
    )
    tipo = st.number_input(
        "Código categoría (type)",
        min_value=0,
        max_value=200,
        value=2,
        step=1,
    )
    city = st.number_input(
        "Código ciudad (city)",
        min_value=0,
        max_value=200,
        value=5,
        step=1,
    )

entrada = pd.DataFrame([{
    "online_order": online_order,
    "book_table": book_table,
    "log_votes": log_votes,
    "cost": cost,
    "n_cuisines": n_cuisines,
    "location": location,
    "rest_type": rest_type,
    "type": tipo,
    "city": city,
}])

st.subheader("Resultados en vivo")
pred_principal = float(modelos[modelo_elegido].predict(entrada)[0])
st.success(f"Predicción en vivo con **{modelo_elegido}**: **{pred_principal:.2f}**")

resumen_modelos = pd.DataFrame([
    {"Modelo": nombre, "Predicción en vivo": round(float(mod.predict(entrada)[0]), 2)}
    for nombre, mod in modelos.items()
])
st.dataframe(resumen_modelos, hide_index=True, use_container_width=True)

with st.expander("10 ejemplos del conjunto de prueba (real vs. predicción)"):
    ejemplos = pd.DataFrame({
        "Fila": list(range(10)),
        "Real": [4.10, 3.80, 4.60, 4.00, 3.60, 4.20, 3.80, 3.90, 3.10, 4.30],
        "Regresión Lineal": [4.01, 3.78, 4.33, 3.91, 3.60, 3.87, 3.71, 3.92, 3.82, 3.86],
        "KNN (k=10)": [3.83, 3.46, 4.30, 3.81, 3.41, 3.89, 3.66, 3.48, 3.98, 3.93],
        "Random Forest": [4.06, 3.71, 4.59, 4.14, 3.55, 4.06, 3.80, 3.74, 3.51, 4.20],
    })
    st.dataframe(ejemplos, hide_index=True, use_container_width=True)

with st.expander("Importancia de variables (Random Forest)"):
    importancia = pd.DataFrame({
        "Variable": [
            "log_votes", "location", "cost", "rest_type",
            "n_cuisines", "city", "book_table", "online_order", "type",
        ],
        "Importancia": [
            0.5604, 0.1219, 0.1040, 0.0783,
            0.0577, 0.0333, 0.0183, 0.0178, 0.0082,
        ],
    }).set_index("Variable")
    st.bar_chart(importancia)
