"""Tests de datos: esquema, rangos y nulos del dataset de churn."""
import pandas as pd
import pytest

from scripts.load_data import COLUMNAS_ESPERADAS, cargar_datos

RANGOS = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 10),
    "Balance": (0, None),
    "NumOfProducts": (1, 4),
    "EstimatedSalary": (0, None),
    "Satisfaction Score": (1, 5),
    "Point Earned": (0, None),
}
BINARIAS = ["HasCrCard", "IsActiveMember", "Exited", "Complain"]
CATEGORIAS = {
    "Geography": {"France", "Spain", "Germany"},
    "Gender": {"Male", "Female"},
    "Card Type": {"SILVER", "GOLD", "PLATINUM", "DIAMOND"},
}


def test_esquema_tiene_todas_las_columnas(df):
    assert set(COLUMNAS_ESPERADAS) <= set(df.columns)


def test_tipos_numericos_y_categoricos(df):
    for columna in [*RANGOS, *BINARIAS]:
        assert pd.api.types.is_numeric_dtype(df[columna]), columna
    for columna in CATEGORIAS:
        assert not pd.api.types.is_numeric_dtype(df[columna]), columna


def test_sin_nulos(df):
    assert df.isna().sum().sum() == 0


def test_sin_clientes_duplicados(df):
    assert df["CustomerId"].is_unique


@pytest.mark.parametrize("columna,rango", RANGOS.items())
def test_rangos_numericos(df, columna, rango):
    minimo, maximo = rango
    assert df[columna].min() >= minimo
    if maximo is not None:
        assert df[columna].max() <= maximo


@pytest.mark.parametrize("columna", BINARIAS)
def test_columnas_binarias(df, columna):
    assert set(df[columna].unique()) <= {0, 1}


@pytest.mark.parametrize("columna,valores", CATEGORIAS.items())
def test_categorias_conocidas(df, columna, valores):
    assert set(df[columna].unique()) <= valores


def test_carga_falla_si_falta_una_columna(tmp_path, df):
    archivo = tmp_path / "incompleto.csv"
    df.drop(columns=["Age"]).to_csv(archivo, index=False)
    with pytest.raises(ValueError, match="Age"):
        cargar_datos(archivo)


def test_carga_falla_si_el_objetivo_tiene_nulos(tmp_path, df):
    archivo = tmp_path / "nulos.csv"
    datos = df.head(20).copy()
    datos.loc[datos.index[0], "Exited"] = None
    datos.to_csv(archivo, index=False)
    with pytest.raises(ValueError, match="nulos"):
        cargar_datos(archivo)
