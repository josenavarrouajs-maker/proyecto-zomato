from pathlib import Path
import joblib
import numpy as np
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

ejemplos_prueba = pd.DataFrame({
    "Fila": list(range(10)),
    "Real": [4.10, 3.80, 4.60, 4.00, 3.60, 4.20, 3.80, 3.90, 3.10, 4.30],
    "Regresión Lineal": [4.01, 3.78, 4.33, 3.91, 3.60, 3.87, 3.71, 3.92, 3.82, 3.86],
    "KNN (k=10)": [3.83, 3.46, 4.30, 3.81, 3.41, 3.89, 3.66, 3.48, 3.98, 3.93],
    "Random Forest": [4.06, 3.71, 4.59, 4.14, 3.55, 4.06, 3.80, 3.74, 3.51, 4.20],
})

st.sidebar.header("Configuración")
modelo_elegido = st.sidebar.selectbox("Modelo principal:", list(modelos.keys()))

st.sidebar.subheader("Calificación real para métrica en vivo")
calificacion_real_ref = st.sidebar.number_input(
    "Calificación real de referencia (rate)",
    min_value=0.0,
    max_value=5.0,
    value=4.00,
    step=0.10,
)

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

calcular = st.button("Calcular", type="primary")

st.subheader("Resultados en vivo")
pred_principal = float(modelos[modelo_elegido].predict(entrada)[0])

if calcular:
    st.success(f"Calculado con **{modelo_elegido}**: **{pred_principal:.2f}**")
else:
    st.info(f"Predicción en vivo con **{modelo_elegido}**: **{pred_principal:.2f}**")

filas_resumen = []
filas_metricas_vivo = []

for nombre, mod in modelos.items():
    pred_val = float(mod.predict(entrada)[0])
    error_abs = abs(calificacion_real_ref - pred_val)
    error_cuad = (calificacion_real_ref - pred_val) ** 2
    filas_resumen.append({
        "Modelo": nombre,
        "Predicción en vivo": round(pred_val, 2),
        "Error abs (MAE en vivo)": round(error_abs, 4),
    })
    filas_metricas_vivo.append({
        "Modelo": nombre,
        "RMSE vivo": round(float(np.sqrt(error_cuad)), 4),
        "MAE vivo": round(error_abs, 4),
        "Predicción": round(pred_val, 4),
    })

resumen_modelos = pd.DataFrame(filas_resumen)
st.dataframe(resumen_modelos, hide_index=True, use_container_width=True)

st.sidebar.subheader("Métricas en vivo (vs. Real)")
metricas_vivo_df = pd.DataFrame(filas_metricas_vivo)
st.sidebar.data_editor(
    metricas_vivo_df,
    hide_index=True,
    use_container_width=True,
    key="metricas_en_vivo_editor",
)

with st.expander("10 ejemplos del conjunto de prueba (real vs. predicción)"):
    st.dataframe(ejemplos_prueba, hide_index=True, use_container_width=True)

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
