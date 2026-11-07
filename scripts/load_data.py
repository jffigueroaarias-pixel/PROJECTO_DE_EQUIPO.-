import pandas as pd
from pathlib import Path

def cargar_datos_churn() -> pd.DataFrame:
    """Carga el dataset de Churn desde la ruta establecida y valida columnas."""
    # Ruta relativa al dataset según la estructura real del proyecto (subiendo 1 nivel desde scripts)
    ruta_csv = Path(__file__).resolve().parents[1] / "data" / "Customer-Churn-Records.csv"
    
    if not ruta_csv.exists():
        # Intentar ruta alternativa común en la raíz de actividades
        ruta_csv = Path("activities/data/Customer-Churn-Records.csv")
        
    if not ruta_csv.exists():
        # Intentar ruta directa desde la raíz del workspace
        ruta_csv = Path("data/Customer-Churn-Records.csv")
        
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo Customer-Churn-Records.csv en las rutas buscadas.")
        
    df = pd.read_csv(ruta_csv)
    return df