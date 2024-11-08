from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
from prefect import task, Flow



# Definir tareas
@task
def read_dataframe(filename):
    df = pd.read_csv(filename)
    categorical = ['label', 'device']
    df[categorical] = df[categorical].astype(str)
    df.dropna(inplace=True)
    return df

@task
def preprocess_data(df):
    X = df[['sessions', 'drives', 'total_sessions']]
    y = df['label'].apply(lambda x: 1 if x == 'retained' else 0)
    return X, y

@task
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def train_models(X_train, y_train):
    models = [
        {"model": LogisticRegression(), "name": "LogisticRegression"},
        {"model": DecisionTreeClassifier(), "name": "DecisionTreeClassifier"},
        {"model": RandomForestClassifier(), "name": "RandomForestClassifier"},
        {"model": SVC(probability=True), "name": "SVC"},
    ]
    trained_models = []
    for m in models:
        m["model"].fit(X_train, y_train)
        trained_models.append(m)
    return trained_models

@task
def evaluate_models(models, X_test, y_test):
    results = []
    for m in models:
        y_pred = m["model"].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({"name": m["name"], "model": m["model"], "accuracy": accuracy})
    return results

@task
def select_best_model(results):
    best_model = max(results, key=lambda x: x["accuracy"])
    return best_model

@task
def log_models(results):
    mlflow.set_experiment(experiment_name="model_experiment")
    with mlflow.start_run(run_name="Nested Runs"):
        for res in results:
            with mlflow.start_run(run_name=res["name"], nested=True):
                mlflow.log_metric("accuracy", res["accuracy"])
                mlflow.sklearn.log_model(res["model"], artifact_path="model")
    return

@task
def register_best_model(best_model):
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    mlflow.set_experiment(experiment_name="model_experiment")
    # Registrar el mejor modelo
    result = mlflow.register_model(
        model_uri="runs:/{}/model".format(best_model["model"].steps[-1][1].run_id),
        name="waze-model"
    )
    return result












# Definir el modelo de datos de entrada
class InputData(BaseModel):
    sessions: float
    drives: float
    total_sessions: float

app = FastAPI()

# Cargar el modelo 'champion' desde MLflow
model_name = "waze-model"
model_alias = "champion"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}@{model_alias}"
)

@app.post("/predict")
async def predict(input_data: InputData):
    # Convertir los datos de entrada en un DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Realizar la predicción
    prediction = model.predict(df)
    
    # Retornar la predicción
    return {"prediction": int(prediction[0])}






