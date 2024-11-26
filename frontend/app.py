import streamlit as st
import requests

# Configuración de la aplicación
st.set_page_config(
    page_title="Clasificación Churn de Waze",
    layout="centered"
)

# Título de la aplicación
st.title("Clasificación Churn de Waze")

# Inputs del usuario
st.subheader("Por favor, introduce los datos:")
sessions = st.number_input("Número de sesiones", min_value=0, value=10, step=1)
drives = st.number_input("Número de viajes", min_value=0, value=5, step=1)
total_sessions = st.number_input("Total de sesiones acumuladas", min_value=0, value=50, step=1)

# Botón para realizar la predicción
if st.button("Predecir"):
    # Validar entrada
    if sessions < 0 or drives < 0 or total_sessions < 0:
        st.error("Por favor, introduce valores válidos (no negativos).")
    else:
        # Enviar la solicitud a la API
        try:
            api_url = "http://api:8000/predict"  # URL del contenedor de la API
            response = requests.post(
                api_url,
                json={
                    "sessions": sessions,
                    "drives": drives,
                    "total_sessions": total_sessions
                }
            )
            # Manejar respuesta de la API
            if response.status_code == 200:
                prediction = response.json().get("prediction", "Error en la respuesta de la API")
                if prediction == 1:
                    st.success("El comportamiento predicho es: retained")
                else:
                    st.success("El comportamiento predicho es: churned")
            else:
                st.error(f"No se pudo conectar con la API. Código de estado: {response.status_code}")
                st.error(f"Detalles del error: {response.text}")
        except Exception as e:
            st.error(f"Error al intentar conectarse con la API: {e}")
