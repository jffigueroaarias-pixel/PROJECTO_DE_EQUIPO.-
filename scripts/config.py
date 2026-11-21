"""Lectura de los archivos YAML de configuración."""
from pathlib import Path

import yaml

# Raíz del repositorio: las rutas del YAML se resuelven desde aquí, así los
# scripts funcionan sin importar la carpeta desde la que se ejecuten.
RAIZ = Path(__file__).resolve().parents[1]


def cargar_config(nombre: str = "config.yaml") -> dict:
    with open(RAIZ / "configs" / nombre, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(relativa: str) -> Path:
    """Convierte una ruta del YAML (relativa a la raíz) en absoluta."""
    return RAIZ / relativa
