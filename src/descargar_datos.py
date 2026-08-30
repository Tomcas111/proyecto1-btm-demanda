# -*- coding: utf-8 -*-
"""
Script OPCIONAL de actualización de datos desde fuentes primarias.

No es necesario para reproducir el informe (los CSV de data/ ya traen las
series consolidadas y citadas), pero permite regenerarlas o refinarlas:

  1. Demanda comercial mensual del SIN desde la API pública de XM
     (paquete pydataxm) -> agregación anual.
  2. PIB real de Colombia (moneda constante) desde la API del Banco Mundial.
  3. Anexo oficial de la proyección UPME 2026-2040 (Excel, rev. jul-2026).

Uso:  pip install pydataxm requests openpyxl
      python descargar_datos.py
"""
import sys
from pathlib import Path

DATOS = Path(__file__).resolve().parents[1] / "data"

URL_UPME_XLSX = ("https://docs.upme.gov.co/DemandayEficiencia/Documents/"
                 "Anexo_proyeccion_demanda_2026_2040_ver_Ago2026.xlsx")
URL_WB = ("https://api.worldbank.org/v2/country/COL/indicator/"
          "NY.GDP.MKTP.KN?format=json&per_page=60&date=1995:2025")


def demanda_xm():
    """Demanda comercial del SIN, mensual -> anual, vía pydataxm."""
    try:
        from pydataxm.pydataxm import ReadDB
    except ImportError:
        print("pydataxm no instalado: pip install pydataxm"); return
    import datetime as dt
    import pandas as pd
    api = ReadDB()
    df = api.request_data("DemaCome", "Sistema",
                          dt.date(2000, 1, 1), dt.date.today())
    cols = [c for c in df.columns if c.startswith("Values_Hour")]
    df["gwh_dia"] = df[cols].sum(axis=1) / 1e6      # kWh -> GWh
    df["anio"] = pd.to_datetime(df["Date"]).dt.year
    anual = df.groupby("anio")["gwh_dia"].sum().round(0)
    anual.to_csv(DATOS / "demanda_xm_api.csv", header=["demanda_gwh"])
    print("OK -> data/demanda_xm_api.csv\n", anual.tail())


def pib_banco_mundial():
    try:
        import requests
    except ImportError:
        print("requests no instalado"); return
    import pandas as pd
    r = requests.get(URL_WB, timeout=60).json()[1]
    serie = (pd.DataFrame(r)[["date", "value"]].dropna().astype(
        {"date": int, "value": float}).sort_values("date"))
    serie["pib_mmm_cop2015"] = (serie["value"] / 1e9).round(0)
    serie.rename(columns={"date": "anio"})[["anio", "pib_mmm_cop2015"]].to_csv(
        DATOS / "pib_wb_api.csv", index=False)
    print("OK -> data/pib_wb_api.csv")


def anexo_upme():
    try:
        import requests
    except ImportError:
        print("requests no instalado"); return
    destino = DATOS / "Anexo_proyeccion_demanda_2026_2040.xlsx"
    r = requests.get(URL_UPME_XLSX, timeout=120)
    r.raise_for_status()
    destino.write_bytes(r.content)
    print(f"OK -> {destino} ({len(r.content)/1e6:.1f} MB)")


if __name__ == "__main__":
    for fn in (demanda_xm, pib_banco_mundial, anexo_upme):
        try:
            fn()
        except Exception as exc:            # red corporativa, VPN, etc.
            print(f"[aviso] {fn.__name__} falló: {exc}", file=sys.stderr)
