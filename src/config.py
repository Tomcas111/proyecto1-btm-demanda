# -*- coding: utf-8 -*-
"""
Configuración común del proyecto:
rutas, paleta de colores validada y parámetros del análisis BtM.

Proyecto 1 - ILE4100 Planeamiento de Sistemas de Potencia
Integración del efecto Behind-the-Meter (BtM) en la estimación
de demanda de largo plazo del sistema eléctrico colombiano.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"
SALIDAS = RAIZ / "output"
FIGURAS = SALIDAS / "figuras"
TABLAS = SALIDAS / "tablas"
for _d in (SALIDAS, FIGURAS, TABLAS):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------- paleta (superficie clara)
# Paleta categórica validada (orden fijo, no ciclar) + tinta/rejilla.
PALETA = {
    "serie1": "#2a78d6",   # azul     - demanda bruta / principal
    "serie2": "#eb6834",   # naranja  - escenario CREG 4%
    "serie3": "#1baf7a",   # aqua     - escenario tipo España
    "serie4": "#eda100",   # amarillo - escenario PNUMA
    "tinta": "#0b0b0b",
    "tinta2": "#52514e",
    "silenciado": "#898781",
    "rejilla": "#e1e0d9",
    "eje": "#c3c2b7",
    "superficie": "#fcfcfb",
}

ESTILO_MPL = {
    "figure.facecolor": PALETA["superficie"],
    "axes.facecolor": PALETA["superficie"],
    "savefig.facecolor": PALETA["superficie"],
    "axes.edgecolor": PALETA["eje"],
    "axes.labelcolor": PALETA["tinta2"],
    "axes.grid": True,
    "grid.color": PALETA["rejilla"],
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": PALETA["silenciado"],
    "ytick.color": PALETA["silenciado"],
    "text.color": PALETA["tinta"],
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "lines.linewidth": 2.0,
    "legend.frameon": False,
}

# ------------------------------------------------------------ parámetros
ANIO_BASE = 2025          # último año observado (ancla de la proyección)
HORIZONTE = range(2026, 2041)   # 2026-2040, alineado con UPME rev. jul-2026

# Supuestos de crecimiento del PIB real (UPME-Fedesarrollo, rev. ene-2026)
PIB_CRECIMIENTO = {2026: 0.029}
PIB_CRECIMIENTO.update({a: 0.030 for a in range(2027, 2041)})

# Factor de planta FV Colombia (≈1.450 kWh/kWp-año) con sensibilidad
FACTOR_PLANTA_FV = 0.165
FACTOR_PLANTA_FV_RANGO = (0.15, 0.18)

# Participación del mercado regulado en la demanda comercial (2023, UPME)
PART_REGULADO = 0.68

# Umbral CREG (Res. 101 072 de 2025, art. 12): 4 % de la demanda
# comercial regulada anual cubierta con energía BtM entregada a la red.
UMBRAL_CREG = 0.04
# Fracción de la generación BtM que se entrega a la red como excedente
# (autoconsumo instantáneo típico FV residencial/comercial sin baterías ~50-70 %)
FRACCION_EXCEDENTES = 0.40


def cargar_series():
    """Serie anual demanda SIN (GWh) y PIB real (mmm COP-2015), 2000-2025."""
    df = pd.read_csv(DATOS / "demanda_pib_anual.csv").set_index("anio")
    return df


def cargar_upme():
    """Proyección oficial UPME rev. jul-2024 (escenario medio)."""
    return pd.read_csv(DATOS / "upme_proyeccion_jul2024.csv").set_index("anio")
