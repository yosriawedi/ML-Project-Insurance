from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, pandas as pd, os

# --------- Chargement des modèles (assure-toi d'avoir scikit-learn==1.6.1 dans le venv) ---------
MODELS_DIR = "models"
ocsvm_a = joblib.load(os.path.join(MODELS_DIR, "ocsvm_FS_A.joblib"))
ocsvm_b = joblib.load(os.path.join(MODELS_DIR, "ocsvm_FS_B.joblib"))
iforest_a = joblib.load(os.path.join(MODELS_DIR, "isoforest_FS_A.joblib"))
iforest_b = joblib.load(os.path.join(MODELS_DIR, "isoforest_FS_B.joblib"))

# --------- App ---------
app = FastAPI(title="Anomaly Detection API")

# (utile si tu ouvres la page depuis un autre hôte que celui de l'API ; sinon inoffensif)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreins en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Page d'accueil -> sert l'interface
@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))

# --------- Schéma d'entrée ---------
class InsuranceData(BaseModel):
    age: float
    bmi: float
    children: int
    sex: str
    smoker: str
    region: str
    charges: float

# --------- Endpoint prédiction ---------
@app.post("/predict")
def predict(data: InsuranceData):
    X = pd.DataFrame([data.dict()])

    results = {}
    # OCSVM A
    Z = ocsvm_a.named_steps["prep"].transform(X)
    score = -ocsvm_a.named_steps["model"].decision_function(Z)
    results["ocsvm_A_score"] = float(score[0])
    results["ocsvm_A_anomaly"] = int(score[0] > 0.5)  # ajuste le seuil

    # OCSVM B
    Z = ocsvm_b.named_steps["prep"].transform(X)
    score = -ocsvm_b.named_steps["model"].decision_function(Z)
    results["ocsvm_B_score"] = float(score[0])
    results["ocsvm_B_anomaly"] = int(score[0] > 0.5)

    # IF A
    Z = iforest_a.named_steps["prep"].transform(X)
    score = -iforest_a.named_steps["model"].score_samples(Z)
    results["iforest_A_score"] = float(score[0])
    results["iforest_A_anomaly"] = int(score[0] > 0.5)

    # IF B
    Z = iforest_b.named_steps["prep"].transform(X)
    score = -iforest_b.named_steps["model"].score_samples(Z)
    results["iforest_B_score"] = float(score[0])
    results["iforest_B_anomaly"] = int(score[0] > 0.5)

    return JSONResponse(results)

@app.get("/health")
def health():
    return {"status": "ok"}
