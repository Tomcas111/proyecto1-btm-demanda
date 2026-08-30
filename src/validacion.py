# -*- coding: utf-8 -*-
"""
Etapa 4 del pipeline: MARCO DE VALIDACIÓN Y COMPARACIÓN.

Prueba empírica de la metodología propuesta con datos históricos, mediante
tres ejercicios complementarios:

V1  Backtest retrospectivo en Colombia (modelo autorregresivo multivariado).
    Se reestimó el modelo de combinación VAR/VEC con información hasta 2019
    (y hasta 2021) y se pronosticó fuera de muestra condicional al PIB
    observado; el desempeño (MAPE, RMSE) se comparó contra dos referentes
    univariados: caminata aleatoria con deriva y AR(1).

V2  Experimento natural en España (validación del Paso 1: reconstrucción).
    Con las series de APPA (producción de autoconsumo 2018-2025 y demanda
    nacional b.c. 2020-2025) se contrastó la demanda MEDIDA (neta) contra la
    demanda RECONSTRUIDA (bruta = neta + autoconsumo aprovechado): tasas de
    crecimiento, elasticidades aparentes al PIB y fracción del crecimiento
    real "enmascarada". Además se validó el estimador de reconstrucción
    G = P_prom x horas (Paso 1) contra la producción real reportada.

V3  Panel Colombia-España (Gujarati cap. 16, LSDV con efectos fijos).
    ln D_it = a_i + b_i ln Y_it + c s_it + e_it, con s = participación BtM
    sobre demanda bruta. Hipótesis H0: c = -1 (enmascaramiento uno a uno,
    pues ln(1-s) ~ -s). No rechazar H0 valida la corrección aditiva.

Salidas: output/tablas/t7_backtest.csv, t8_espana.csv, t9_panel.csv
         y output/validacion_series.csv (para figuras).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from config import DATOS, SALIDAS, TABLAS, cargar_series

HORAS_APROVECHADAS_ES = None   # se calibra con 2025 (10.550 GWh / P media)


# ================================================== V1: backtest Colombia
def _pronostico_condicional(entren, anios_futuro, pib_obs):
    """Combinación VEC+VAR condicional a la senda de PIB observada."""
    entren = entren.copy()
    entren["d2020"] = (entren.index == 2020).astype(float)
    usa_dummy = entren["d2020"].sum() > 0     # solo si 2020 esta en la muestra
    n = len(anios_futuro)
    # --- VECM
    kw = dict(exog=entren[["d2020"]]) if usa_dummy else {}
    res_v = VECM(entren[["ln_dee", "ln_pib"]], k_ar_diff=1, coint_rank=1,
                 deterministic="ci", **kw).fit()
    kw_fc = dict(exog_fc=np.zeros((n, 1))) if usa_dummy else {}
    f_vecm = res_v.predict(steps=n, **kw_fc)[:, 0]
    # --- VAR(1) en diferencias con PIB impuesto
    d = entren[["ln_dee", "ln_pib"]].diff().dropna()
    res_a = VAR(d).fit(1)
    ult, ln = d.values[-1:], [entren["ln_dee"].iloc[-1]]
    for a in anios_futuro:
        d_dee = res_a.forecast(ult, 1)[0][0]
        ln.append(ln[-1] + d_dee)
        ult = np.array([[d_dee, pib_obs[a]]])
    f_var = np.array(ln[1:])
    return 0.5 * f_vecm + 0.5 * f_var, f_vecm, f_var


def backtest_colombia():
    df = cargar_series()
    df["ln_dee"] = np.log(df["demanda_gwh"])
    df["ln_pib"] = np.log(df["pib_mmm_cop2015"])
    pib_obs = df["ln_pib"].diff().to_dict()

    filas = []
    for corte in (2019, 2021):
        anios = [a for a in df.index if a > corte]
        entren = df[df.index <= corte]
        real = df.loc[anios, "demanda_gwh"].values
        comb, f_vecm, f_var = _pronostico_condicional(entren, anios, pib_obs)
        pred = {"Combinado VEC+VAR": np.exp(comb)}
        # referente 1: caminata aleatoria con deriva histórica
        deriva = entren["ln_dee"].diff().mean()
        pred["Caminata aleatoria c/deriva"] = np.exp(
            entren["ln_dee"].iloc[-1] + deriva * np.arange(1, len(anios) + 1))
        # referente 2: AR(1) univariado sobre crecimientos
        g = entren["ln_dee"].diff().dropna()
        rho = sm.OLS(g[1:].values, sm.add_constant(g[:-1].values)).fit().params
        gs, ln_ar = g.iloc[-1], [entren["ln_dee"].iloc[-1]]
        for _ in anios:
            gs = rho[0] + rho[1] * gs
            ln_ar.append(ln_ar[-1] + gs)
        pred["AR(1) univariado"] = np.exp(ln_ar[1:])
        for nombre, p in pred.items():
            filas.append({
                "corte_entrenamiento": corte,
                "horizonte": f"{anios[0]}-{anios[-1]}",
                "modelo": nombre,
                "MAPE_pct": round(100 * np.mean(np.abs(p - real) / real), 2),
                "RMSE_gwh": int(np.sqrt(np.mean((p - real) ** 2))),
            })
        if corte == 2019:   # serie para la figura F7a
            pd.DataFrame({"anio": anios, "real": real,
                          "combinado": pred["Combinado VEC+VAR"].round(0),
                          "rw_deriva": pred["Caminata aleatoria c/deriva"].round(0)}
                         ).to_csv(SALIDAS / "backtest_2019.csv", index=False)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLAS / "t7_backtest.csv", index=False)
    return tabla


# ================================================== V2: experimento España
def experimento_espana():
    es = pd.read_csv(DATOS / "espana_anual.csv").set_index("anio")
    es["bruta_gwh"] = es["demanda_bc_gwh"] + es["autoconsumo_aprovechado_gwh"]

    # --- (a) crecimiento y enmascaramiento 2020-2025
    neta_ini, neta_fin = es.loc[2020, "demanda_bc_gwh"], es.loc[2025, "demanda_bc_gwh"]
    bruta_ini, bruta_fin = es.loc[2020, "bruta_gwh"], es.loc[2025, "bruta_gwh"]
    d_neta, d_bruta = neta_fin - neta_ini, bruta_fin - bruta_ini
    pib_ini, pib_fin = es.loc[2020, "pib_meur2015"], es.loc[2025, "pib_meur2015"]
    g_pib = np.log(pib_fin / pib_ini)
    elast_neta = np.log(neta_fin / neta_ini) / g_pib
    elast_bruta = np.log(bruta_fin / bruta_ini) / g_pib

    # --- (b) validación del estimador de reconstrucción del Paso 1
    #     G_est(t) = P_promedio(t) x horas, horas calibradas SOLO con 2025
    p_prom = (es["capacidad_btm_mw"] + es["capacidad_btm_mw"].shift(1)) / 2
    horas = es.loc[2025, "autoconsumo_aprovechado_gwh"] * 1000 / p_prom.loc[2025]
    g_est = (p_prom * horas / 1000).dropna()
    err = 100 * (g_est - es["autoconsumo_aprovechado_gwh"]) / \
        es["autoconsumo_aprovechado_gwh"]
    mape_rec = np.abs(err.loc[2021:2025]).mean()   # 2019-2020: base pequeña

    filas = [
        ("Crecimiento demanda medida (neta) 2020-2025, %", round(100 * (neta_fin / neta_ini - 1), 2)),
        ("Crecimiento demanda reconstruida (bruta) 2020-2025, %", round(100 * (bruta_fin / bruta_ini - 1), 2)),
        ("Crecimiento real del consumo enmascarado, %", round(100 * (1 - d_neta / d_bruta), 1)),
        ("Elasticidad aparente al PIB, demanda neta", round(elast_neta, 2)),
        ("Elasticidad aparente al PIB, demanda bruta", round(elast_bruta, 2)),
        ("Horas equivalentes calibradas 2025, h/anio", int(horas)),
        ("MAPE del estimador de reconstruccion 2021-2025, %", round(mape_rec, 1)),
        ("Demanda+autoconsumo publicada por REE 2025, GWh", 269753),
    ]
    tabla = pd.DataFrame(filas, columns=["indicador", "valor"])
    tabla.to_csv(TABLAS / "t8_espana.csv", index=False)

    es[["demanda_bc_gwh", "bruta_gwh", "autoconsumo_aprovechado_gwh"]].to_csv(
        SALIDAS / "espana_series.csv")
    detalle = pd.DataFrame({"g_estimada": g_est.round(0),
                            "g_real": es["autoconsumo_aprovechado_gwh"],
                            "error_pct": err.round(1)}).dropna()
    detalle.to_csv(SALIDAS / "espana_reconstruccion.csv")
    return tabla, detalle


# ================================================== V3: panel CO-ES (LSDV)
def panel_lsdv():
    co = cargar_series().copy()
    co["pais"], co["dem"], co["pib"] = "CO", co["demanda_gwh"], co["pib_mmm_cop2015"]
    # participación BtM sobre demanda bruta en Colombia (registro AGPE/GD):
    # despreciable hasta 2017; 2018-2025 según capacidad acumulada reportada
    cap_co = {2018: 30, 2019: 60, 2020: 100, 2021: 150, 2022: 200, 2023: 236,
              2024: 339, 2025: 645}          # MW (UPME, nota 24 rev jul-2024)
    gen_co = {a: c * 1.4454 for a, c in cap_co.items()}     # GWh (FC 16,5 %)
    co["s"] = [gen_co.get(a, 0.0) / (d + gen_co.get(a, 0.0))
               for a, d in zip(co.index, co["dem"])]

    es = pd.read_csv(DATOS / "espana_anual.csv").set_index("anio").dropna(
        subset=["demanda_bc_gwh"])
    es["pais"], es["dem"], es["pib"] = "ES", es["demanda_bc_gwh"], es["pib_meur2015"]
    es["s"] = es["autoconsumo_aprovechado_gwh"] / \
        (es["demanda_bc_gwh"] + es["autoconsumo_aprovechado_gwh"])

    panel = pd.concat([co[["pais", "dem", "pib", "s"]],
                       es[["pais", "dem", "pib", "s"]]])
    panel["ln_d"], panel["ln_y"] = np.log(panel["dem"]), np.log(panel["pib"])

    # Panel en TASAS DE CRECIMIENTO (LSDV en primeras diferencias).
    # En niveles, s_t y ln Y_t son casi colineales en la corta muestra
    # espanola y el coeficiente c no queda identificado; en diferencias la
    # variacion de instalacion anual (delta s) se separa del ciclo del PIB.
    #   dln D_it = a_i + b_i dln Y_it + c ds_it + phi d2020 + e_it,  H0: c=-1
    partes = []
    for _, x in panel.groupby("pais"):
        partes.append(x.assign(dln_d=x["ln_d"].diff(), dln_y=x["ln_y"].diff(),
                               ds=x["s"].diff()))
    g = pd.concat(partes).dropna(subset=["dln_d"])
    g["d2020"] = (g.index == 2020).astype(float)
    X = pd.DataFrame({
        "const_CO": (g["pais"] == "CO").astype(float),
        "const_ES": (g["pais"] == "ES").astype(float),
        "dlnY_CO": g["dln_y"] * (g["pais"] == "CO"),
        "dlnY_ES": g["dln_y"] * (g["pais"] == "ES"),
        "d2020": g["d2020"],
        "ds_btm": g["ds"],
    })
    modelo = sm.OLS(g["dln_d"], X).fit(cov_type="HC1")
    c, se = modelo.params["ds_btm"], modelo.bse["ds_btm"]
    t_h0 = (c - (-1.0)) / se          # H0: c = -1 (enmascaramiento 1 a 1)
    from scipy import stats
    p_h0 = 2 * (1 - stats.t.cdf(abs(t_h0), df=int(modelo.df_resid)))
    tabla = pd.DataFrame({
        "parametro": ["c (coef. de Delta s BtM)", "e.e. robusto (HC1)",
                      "IC 95 %", "t de H0: c=-1", "p-valor H0: c=-1",
                      "elasticidad PIB Colombia (corto plazo)",
                      "elasticidad PIB Espana (corto plazo)",
                      "R2", "n (obs.)"],
        "valor": [round(c, 3), round(se, 3),
                  f"[{c-1.96*se:.2f}; {c+1.96*se:.2f}]",
                  round(t_h0, 2), round(p_h0, 3),
                  round(modelo.params["dlnY_CO"], 3),
                  round(modelo.params["dlnY_ES"], 3),
                  round(modelo.rsquared, 3), int(modelo.nobs)],
    })
    tabla.to_csv(TABLAS / "t9_panel.csv", index=False)
    return tabla, modelo


if __name__ == "__main__":
    print("== V1. Backtest Colombia ==\n", backtest_colombia(), "\n")
    t8, det = experimento_espana()
    print("== V2. Experimento España ==\n", t8, "\n")
    print(det, "\n")
    t9, _ = panel_lsdv()
    print("== V3. Panel LSDV Colombia-España ==\n", t9)
