# -*- coding: utf-8 -*-
"""
Etapa 1 del pipeline: análisis econométrico de la relación de largo plazo
entre la demanda de energía eléctrica del SIN y el PIB real (2000-2025).

Replica, a escala didáctica, la práctica de la UPME (modelo de combinación
de pronósticos con VAR y VEC; cfr. UPME 2016, 2024, 2026 y Gujarati caps.
21-22):

  1. Pruebas de raíz unitaria ADF y KPSS sobre ln(DEE) y ln(PIB).
  2. Prueba de cointegración de Johansen.
  3. Estimación del VEC (elasticidad ingreso de largo plazo y velocidad
     de ajuste) con dummy COVID-2020 como variable exógena.
  4. Estimación de un VAR en diferencias (modelo alternativo).
  5. Combinación de pronósticos con ponderaciones por inverso del ECM
     en evaluación pseudo-fuera-de-muestra.
  6. Proyección de la demanda "orgánica" del SIN 2026-2040 bajo la senda
     de PIB UPME-Fedesarrollo.

Salidas: output/tablas/*.csv y output/series_proyeccion.csv
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order
from statsmodels.tsa.api import VAR

from config import (ANIO_BASE, HORIZONTE, PIB_CRECIMIENTO, TABLAS, SALIDAS,
                    cargar_series)

np.random.seed(4100)


# ------------------------------------------------------------------ datos
def preparar_datos():
    df = cargar_series()
    df["ln_dee"] = np.log(df["demanda_gwh"])
    df["ln_pib"] = np.log(df["pib_mmm_cop2015"])
    df["d2020"] = (df.index == 2020).astype(float)
    return df


# ------------------------------------------------- 1. raíces unitarias
def pruebas_raiz_unitaria(df):
    filas = []
    for nombre, serie in [("ln(DEE)", df["ln_dee"]), ("ln(PIB)", df["ln_pib"]),
                          ("Δln(DEE)", df["ln_dee"].diff().dropna()),
                          ("Δln(PIB)", df["ln_pib"].diff().dropna())]:
        adf_est, adf_p, _, _, _, _ = adfuller(serie, regression="c", autolag="AIC")
        try:
            kpss_est, kpss_p, _, _ = kpss(serie, regression="c", nlags="auto")
        except Exception:
            kpss_est, kpss_p = np.nan, np.nan
        filas.append({"serie": nombre, "ADF_estadistico": round(adf_est, 3),
                      "ADF_pvalor": round(adf_p, 3),
                      "KPSS_estadistico": round(kpss_est, 3),
                      "KPSS_pvalor": round(kpss_p, 3)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TABLAS / "t1_raices_unitarias.csv", index=False)
    return tabla


# ------------------------------------------------------- 2. cointegración
def prueba_johansen(df):
    """Johansen en dos muestras: completa y pre-COVID (2000-2019).

    El quiebre estructural de 2020 (la demanda cayó -2,3 % y el PIB -7,3 %,
    con recuperaciones asimétricas) debilita la evidencia de cointegración
    en la muestra completa; la relación de largo plazo se aprecia con
    claridad en la muestra estable 2000-2019. Este es, en pequeño, el mismo
    fenómeno que el BtM produciría de forma permanente sobre la demanda
    medida (ver informe, sección de metodología).
    """
    filas = []
    for det, det_lbl in [(0, "constante"), (1, "constante+tendencia")]:
        for etiqueta, sub in [("2000-2025", df),
                              ("2000-2019", df[df.index <= 2019])]:
            jo = coint_johansen(sub[["ln_dee", "ln_pib"]].values,
                                det_order=det, k_ar_diff=1)
            for i, hip in enumerate(["r = 0", "r <= 1"]):
                filas.append({
                    "deterministica": det_lbl, "muestra": etiqueta,
                    "hipotesis": hip,
                    "traza": round(jo.lr1[i], 3),
                    "vc_traza_90": round(jo.cvt[i, 0], 3),
                    "vc_traza_95": round(jo.cvt[i, 1], 3),
                    "max_autovalor": round(jo.lr2[i], 3),
                    "vc_max_95": round(jo.cvm[i, 1], 3),
                })
    tabla = pd.DataFrame(filas)
    tabla["rechaza_95_traza"] = tabla["traza"] > tabla["vc_traza_95"]
    tabla.to_csv(TABLAS / "t2_johansen.csv", index=False)
    return tabla


def prueba_engle_granger(df):
    """Engle-Granger: ADF sobre los residuos de ln_dee ~ ln_pib (Gujarati c.21)."""
    import statsmodels.api as sm
    ols = sm.OLS(df["ln_dee"], sm.add_constant(df["ln_pib"])).fit()
    adf = adfuller(ols.resid, regression="n", autolag="AIC")
    tabla = pd.DataFrame({
        "estadistico": ["beta (elasticidad estatica)", "constante",
                        "ADF residuos", "p-valor ADF residuos", "R2"],
        "valor": [round(ols.params["ln_pib"], 3), round(ols.params["const"], 3),
                  round(adf[0], 3), round(adf[1], 4), round(ols.rsquared, 4)],
    })
    tabla.to_csv(TABLAS / "t2b_engle_granger.csv", index=False)
    return tabla


# ------------------------------------------------------------- 3. VECM
def estimar_vecm(df):
    """VECM(1), rango 1, constante en la CE, en dos muestras.

    En 2000-2019 el mecanismo de corrección de error opera como predice la
    teoría (alpha < 0 y significativo). En la muestra completa, la sucesión
    de choques 2020-2025 (COVID, El Niño 2023-24, entrada de grandes cargas)
    distorsiona el término de corrección: alpha cambia de signo y la "CE"
    absorbe la tendencia común. Es la ilustración empírica del argumento
    central del informe: los quiebres en la demanda MEDIDA -como el que el
    BtM inducirá de forma creciente- degradan la inferencia del VEC si no se
    reconstruye la demanda bruta antes de estimar.
    """
    filas = []
    for etiqueta, sub, exog in [
            ("2000-2019", df[df.index <= 2019], None),
            ("2000-2025", df, df[["d2020"]])]:
        res = VECM(sub[["ln_dee", "ln_pib"]], exog=exog, k_ar_diff=1,
                   coint_rank=1, deterministic="ci").fit()
        beta_n = res.beta[:, 0] / res.beta[0, 0]
        filas.append({
            "muestra": etiqueta,
            "elasticidad_LP": round(float(-beta_n[1]), 3),
            "alpha_demanda": f"{res.alpha[0,0]:.3f} ({res.tvalues_alpha[0,0]:.2f})",
            "alpha_pib": f"{res.alpha[1,0]:.3f} ({res.tvalues_alpha[1,0]:.2f})",
        })
    resumen = pd.DataFrame(filas)
    resumen.to_csv(TABLAS / "t3_vecm.csv", index=False)
    res_full = VECM(df[["ln_dee", "ln_pib"]], exog=df[["d2020"]], k_ar_diff=1,
                    coint_rank=1, deterministic="ci").fit()
    elasticidad_lp = float(-(res_full.beta[:, 0] / res_full.beta[0, 0])[1])
    return res_full, elasticidad_lp, float(res_full.alpha[0, 0])


def pronostico_vecm(df, pasos):
    """Proyección del VECM condicionada a la senda de PIB (exógena=0 futuro)."""
    endog = df[["ln_dee", "ln_pib"]]
    exog = df[["d2020"]]
    res = VECM(endog, exog=exog, k_ar_diff=1, coint_rank=1,
               deterministic="ci").fit()
    exog_fut = np.zeros((pasos, 1))
    prono = res.predict(steps=pasos, exog_fc=exog_fut)
    return pd.DataFrame(prono, columns=["ln_dee", "ln_pib"],
                        index=list(HORIZONTE)[:pasos])


# ------------------------------------------------------------- 4. VAR(Δ)
def pronostico_var(df, pasos):
    """VAR en primeras diferencias con la senda de PIB impuesta (condicional).

    Se estima Δln_dee ~ rezagos de (Δln_dee, Δln_pib) y se itera hacia
    adelante sustituyendo Δln_pib por la senda exógena UPME-Fedesarrollo,
    de modo análogo al 'VAR exógeno' de la UPME.
    """
    d = df[["ln_dee", "ln_pib"]].diff().dropna()
    sel = VAR(d).select_order(2)
    p = max(1, sel.aic if isinstance(sel.aic, (int, np.integer)) else 1)
    res = VAR(d).fit(p)
    ult = d.values[-p:].copy()
    ln_dee = [df["ln_dee"].iloc[-1]]
    for anio in list(HORIZONTE)[:pasos]:
        pred = res.forecast(ult, 1)[0]
        d_pib = PIB_CRECIMIENTO[anio]           # se impone la senda de PIB
        d_dee = pred[0]
        ln_dee.append(ln_dee[-1] + d_dee)
        ult = np.vstack([ult[1:], [d_dee, np.log(1 + d_pib)]]) if p > 1 \
            else np.array([[d_dee, np.log(1 + d_pib)]])
    idx = list(HORIZONTE)[:pasos]
    return pd.Series(ln_dee[1:], index=idx, name="ln_dee"), p


# ------------------------------------- 5. combinación de pronósticos
def ponderaciones_pseudo_oos(df, inicio_eval=2015):
    """Ponderaciones por inverso del ECM de pronósticos a un paso 2015-2025."""
    errores = {"vecm": [], "var": []}
    anios = [a for a in df.index if a >= inicio_eval]
    for a in anios:
        entren = df[df.index < a]
        if len(entren) < 12:
            continue
        real = df.loc[a, "ln_dee"]
        try:
            e_v = VECM(entren[["ln_dee", "ln_pib"]], exog=entren[["d2020"]],
                       k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
            p_v = e_v.predict(steps=1, exog_fc=np.zeros((1, 1)))[0, 0]
            errores["vecm"].append((real - p_v) ** 2)
        except Exception:
            pass
        try:
            dtr = entren[["ln_dee", "ln_pib"]].diff().dropna()
            r = VAR(dtr).fit(1)
            pr = r.forecast(dtr.values[-1:], 1)[0][0]
            errores["var"].append((real - (entren["ln_dee"].iloc[-1] + pr)) ** 2)
        except Exception:
            pass
    ecm = {k: np.mean(v) for k, v in errores.items() if v}
    inv = {k: 1.0 / v for k, v in ecm.items()}
    s = sum(inv.values())
    w = {k: v / s for k, v in inv.items()}
    pd.DataFrame({"modelo": list(ecm), "ECM_1paso": [ecm[k] for k in ecm],
                  "ponderacion": [round(w[k], 3) for k in ecm]}
                 ).to_csv(TABLAS / "t4_ponderaciones.csv", index=False)
    return w


# ------------------------------------------------------------------ main
def main():
    df = preparar_datos()
    print("== 1. Raíces unitarias ==\n", pruebas_raiz_unitaria(df), "\n")
    print("== 2. Johansen ==\n", prueba_johansen(df), "\n")
    print("== 2b. Engle-Granger ==\n", prueba_engle_granger(df), "\n")
    res, elast, alpha = estimar_vecm(df)
    print(f"== 3. VECM: elasticidad LP = {elast:.3f}, alpha = {alpha:.3f} ==\n")

    pasos = len(list(HORIZONTE))
    f_vecm = pronostico_vecm(df, pasos)["ln_dee"]
    f_var, p_var = pronostico_var(df, pasos)
    w = ponderaciones_pseudo_oos(df)
    print(f"== 5. Ponderaciones (inverso ECM): {w} | VAR(p={p_var}) ==\n")

    ln_comb = w.get("vecm", .5) * f_vecm + w.get("var", .5) * f_var
    demanda = np.exp(ln_comb)

    # Ancla en el nivel oficial 2025 (XM) para eliminar sesgo de nivel:
    base = cargar_series_nivel_2025()
    factor = base / np.exp(df.loc[ANIO_BASE, "ln_dee"])
    demanda_anclada = demanda * factor

    salida = pd.DataFrame({
        "sin_organico_gwh": demanda_anclada.round(0),
        "ln_vecm": f_vecm.round(4), "ln_var": f_var.round(4),
    })
    salida.index.name = "anio"
    salida.to_csv(SALIDAS / "series_proyeccion.csv")
    tcac = (demanda_anclada.iloc[-1] / base) ** (1 / pasos) - 1
    print(f"Proyección SIN orgánico 2040: {demanda_anclada.iloc[-1]:,.0f} GWh "
          f"(TCAC 2025-2040 = {tcac*100:.2f} %)")
    return salida


def cargar_series_nivel_2025():
    return float(cargar_series().loc[ANIO_BASE, "demanda_gwh"])


if __name__ == "__main__":
    main()
