import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import joblib
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier

# Configurar el registro de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar MLflow Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/colome8/PROYECTO_OSKU.mlflow")

# Variables globales para modelos y artefactos
model = RandomForestClassifier()
scaler = None
label_encoder = None

# Definir la ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Función para cargar el modelo Champion y artefactos
def load_models():
    global scaler, model, label_encoder
    error_messages = []

    # Cargar el vectorizador
    if scaler is None:
        try:
            scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
            logger.info("Scaler cargado exitosamente.")
        except Exception as e:
            error_messages.append(f"Error al cargar el scaler: {e}")

    # Cargar el modelo desde el Model Registry
    try:
        client = MlflowClient()
        model_name = "prefect-modelos"
        model_alias = "Champion"  # Usar el alias Champion
        alias_info = client.get_model_version_by_alias(name=model_name, alias=model_alias)
        run_id = alias_info.run_id
        # modelos
        model_uri = f"runs:/{run_id}/modelos"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print("ENTRE!!!!!!!!!!!!!!!!!!!!!!!")
        
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        model = None
        print("NOOOOOOOOOOOOOOOOOOOOO ENTRE!!!!!!!!!!!!!!!!!!!!!!!")

    # Cargar el label encoder
    if label_encoder is None:
        try:
            label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))
            logger.info("Label Encoder cargado exitosamente.")
        except Exception as e:
            error_messages.append(f"Error al cargar el label encoder: {e}")

    return error_messages


# Crear la aplicación FastAPI
app = FastAPI()


# Definir el esquema de entrada
class PredictionRequest(BaseModel):
    sessions: int
    drives: int
    total_sessions: int

'''
{
    "sessions": 30,
    "drives": 5,
    "total_sessions": 50
}
'''

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint para realizar predicciones con el modelo Champion.
    """
    # Cargar modelos y artefactos si no están cargados
    error_messages = load_models()
    if error_messages:
        logger.error(f"Errores al cargar modelos o artefactos: {error_messages}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Error al cargar modelos o artefactos",
                "details": error_messages,
            },
        )

    try:
       # Convertir los datos del request a un array
        input_data = [[request.sessions, request.drives, request.total_sessions]]

        # Asegurarse de que el preprocesador esté ajustado y transformar los datos
        processed_data = scaler.transform(input_data)  

        # Realizar la predicción
        prediction = model.predict(processed_data)

        # Decodificar la predicción
        decoded_prediction = label_encoder.inverse_transform(prediction)

        # Convertir a un tipo estándar de Python
        prediction_output = int(decoded_prediction[0])

        return {"prediction": prediction_output}
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Error durante la predicción", "details": str(e)},
        )

# uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug