import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "quejas_limpias.csv")

TARGET = "CATEGORIA"
