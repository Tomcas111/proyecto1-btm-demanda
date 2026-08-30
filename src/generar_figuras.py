# -*- coding: utf-8 -*-
"""
Etapa 3 del pipeline: figuras del informe (matplotlib, superficie clara).

F1  Demanda del SIN y PIB real, índice 2000=100 (un solo eje).
F2  Proyección 2026-2040: demanda bruta vs. netas por escenario BtM,
    con la senda oficial UPME jul-2024 como referencia.
F3  Capacidad BtM por escenario (MW) y referencia España 2025.
F4  (Anexo) Evolución del autoconsumo en España 2018-2025 (APPA).
F5  (Anexo) Curva de carga diaria ilustrativa 2035 con y sin BtM.
F6  (Anexo) Ajuste del VECM dentro de muestra.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ESTILO_MPL, FIGURAS, PALETA, SALIDAS, cargar_series,
                    cargar_upme, FACTOR_PLANTA_FV)

plt.rcParams.update(ESTILO_MPL)
C = PALETA


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIGURAS / f"{nombre}.pdf")
    fig.savefig(FIGURAS / f"{nombre}.png", dpi=200)
    plt.close(fig)
    print("figura:", nombre)


# ------------------------------------------------------------------- F1
def f1_historia():
    df = cargar_series()
    idx_dee = 100 * df["demanda_gwh"] / df.loc[2000, "demanda_gwh"]
    idx_pib = 100 * df["pib_mmm_cop2015"] / df.loc[2000, "pib_mmm_cop2015"]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.plot(df.index, idx_dee, color=C["serie1"], label="Demanda SIN")
    ax.plot(df.index, idx_pib, color=C["serie2"], label="PIB real")
    ax.annotate("Demanda SIN", (2025, idx_dee.loc[2025]), xytext=(-2, 8),
                textcoords="offset points", ha="right", color=C["tinta2"],
                fontsize=9)
    ax.annotate("PIB real", (2025, idx_pib.loc[2025]), xytext=(-2, -14),
                textcoords="offset points", ha="right", color=C["tinta2"],
                fontsize=9)
    ax.axvspan(2019.5, 2020.5, color=C["rejilla"], alpha=.6, lw=0)
    ax.text(2020, 96, "COVID-19", ha="center", fontsize=8, color=C["silenciado"])
    ax.set_title("Demanda de energía eléctrica del SIN y PIB real (índice 2000 = 100)")
    ax.set_ylabel("Índice 2000 = 100")
    ax.legend(loc="upper left")
    guardar(fig, "f1_demanda_pib")


# ------------------------------------------------------------------- F2
def f2_proyeccion():
    res = pd.read_csv(SALIDAS / "escenarios_btm.csv", index_col="anio")
    upme = cargar_upme()
    df = cargar_series()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    # histórico
    ax.plot(df.index[df.index >= 2015], df.loc[2015:, "demanda_gwh"] / 1000,
            color=C["tinta2"], lw=1.6)
    # proyecciones
    ax.plot(res.index, res["demanda_bruta_gwh"] / 1000, color=C["serie1"],
            label="Bruta (modelo propio, sin BtM)")
    ax.plot(res.index, res["E0_upme_neta_gwh"] / 1000, color=C["serie2"],
            label="Neta E0 (BtM tendencial UPME)")
    ax.plot(res.index, res["E2_espana_neta_gwh"] / 1000, color=C["serie3"],
            label="Neta E2 (trayectoria España)")
    ax.plot(res.index, res["E3_pnuma_neta_gwh"] / 1000, color=C["serie4"],
            label="Neta E3 (potencial PNUMA)")
    ax.plot(upme.index[upme.index >= 2026],
            upme.loc[2026:, "sin_gce_me_gd_gwh_medio"] / 1000,
            color=C["silenciado"], ls=(0, (4, 3)), lw=1.6,
            label="UPME jul-2024, esc. medio (ref.)")
    ax.set_title("Proyección de demanda 2026-2040, bruta vs. neta por escenario BtM")
    ax.set_ylabel("TWh-año")
    ax.legend(loc="upper left", fontsize=8.2)
    guardar(fig, "f2_proyeccion_escenarios")


# ------------------------------------------------------------------- F3
def f3_capacidad():
    res = pd.read_csv(SALIDAS / "escenarios_btm.csv", index_col="anio")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(res.index, res["E0_upme_mw"] / 1000, color=C["serie1"],
            label="E0 Tendencial UPME")
    ax.plot(res.index, res["E1_creg4_mw"] / 1000, color=C["serie2"],
            ls=(0, (4, 3)), label="E1 Envolvente CREG 4 %")
    ax.plot(res.index, res["E2_espana_mw"] / 1000, color=C["serie3"],
            label="E2 Trayectoria España")
    ax.plot(res.index, res["E3_pnuma_mw"] / 1000, color=C["serie4"],
            label="E3 Potencial PNUMA")
    ax.axhline(9.59, color=C["silenciado"], lw=1.0, ls=(0, (1, 2)))
    ax.text(2026.2, 9.75, "España 2025: 9,59 GW BtM (APPA)", fontsize=8,
            color=C["silenciado"])
    ax.set_title("Capacidad BtM instalada por escenario (GW)")
    ax.set_ylabel("GW")
    ax.legend(loc="center left", fontsize=8.2)
    guardar(fig, "f3_capacidad_btm")


# ------------------------------------------------------------------- F4
def f4_espana():
    anios = ["≤2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    resid = np.array([17, 38, 131, 384, 1408, 1935, 2281, 2649])
    ci = np.array([285, 603, 1092, 1969, 3594, 5010, 6095, 6941])
    x = np.arange(len(anios))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, ci, width=.62, color=C["serie1"], label="Comercial / industrial")
    ax.bar(x, resid, width=.62, bottom=ci, color=C["serie2"], label="Residencial",
           edgecolor=C["superficie"], linewidth=2)
    for i, tot in enumerate(resid + ci):
        ax.text(i, tot + 150, f"{tot:,}".replace(",", "."), ha="center",
                fontsize=8, color=C["tinta2"])
    ax.set_xticks(x, anios)
    ax.set_title("España: potencia BtM acumulada (MW), APPA 2025")
    ax.set_ylabel("MW")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    guardar(fig, "f4_espana_btm")


# ------------------------------------------------------------------- F5
def f5_curva_carga():
    """Curva de carga diaria ilustrativa (forma típica del SIN, XM) para 2035
    bajo E2: la punta nocturna (~19-20 h) no cambia; el mediodía se deprime."""
    horas = np.arange(24)
    # forma típica normalizada de un día hábil del SIN (punta 19-20 h = 1,0)
    forma = np.array([.68, .64, .62, .61, .62, .66, .72, .78, .83, .86, .88,
                      .89, .88, .88, .88, .87, .86, .88, .96, 1.0, .98, .92,
                      .83, .74])
    res = pd.read_csv(SALIDAS / "escenarios_btm.csv", index_col="anio")
    pmax_2035 = 15.0  # GW aprox. (UPME jul-2024: 14,7 GW en 2035 con GCE+ME+GD)
    carga = forma * pmax_2035
    btm_mw = res.loc[2035, "E2_espana_mw"] / 1000.0
    # perfil solar normalizado (salida 6 h, cenit 12 h, puesta 18 h)
    solar = np.clip(np.sin(np.pi * (horas - 6) / 12), 0, None) ** 1.5
    neta = carga - btm_mw * 0.75 * solar          # 75 % de potencia al cenit
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(horas, carga, color=C["serie1"], label="Carga bruta")
    ax.plot(horas, neta, color=C["serie2"], label="Carga neta con BtM (E2)")
    ax.fill_between(horas, neta, carga, color=C["serie2"], alpha=.14, lw=0)
    ax.annotate("punta nocturna sin cambio", xy=(19, carga[19]),
                xytext=(21.8, 13.6), fontsize=8, ha="center",
                color=C["tinta2"],
                arrowprops=dict(arrowstyle="-", color=C["silenciado"], lw=.8))
    ax.annotate("«hueco» solar\nal mediodía", (12.2, neta[12] + 0.55),
                fontsize=8, ha="center", color=C["tinta2"])
    ax.set_title("Curva de carga diaria ilustrativa, 2035 (GW), escenario E2")
    ax.set_xlabel("hora del día")
    ax.set_ylabel("GW")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.legend(loc="lower right", fontsize=8.4)
    guardar(fig, "f5_curva_carga")


# ------------------------------------------------------------------- F6
def f6_ajuste_vecm():
    import statsmodels.api as sm
    from analisis_econometrico import preparar_datos
    df = preparar_datos()
    ols = sm.OLS(df["ln_dee"], sm.add_constant(df["ln_pib"])).fit()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(df.index, df["ln_dee"], color=C["serie1"], label="ln(DEE) observado")
    ax.plot(df.index, ols.fittedvalues, color=C["serie2"], ls=(0, (4, 3)),
            label="Relación de largo plazo estimada")
    ax.set_title("Relación de cointegración demanda-PIB (estática, 2000-2025)")
    ax.set_ylabel("ln(GWh)")
    ax.legend(loc="upper left", fontsize=8.4)
    guardar(fig, "f6_cointegracion")


# ------------------------------------------------------------------- F7
def f7_validacion():
    """Panel doble: (a) backtest Colombia 2020-2025; (b) experimento España:
    demanda medida vs. reconstruida (bruta = neta + autoconsumo APPA)."""
    bt = pd.read_csv(SALIDAS / "backtest_2019.csv").set_index("anio")
    es = pd.read_csv(SALIDAS / "espana_series.csv").set_index("anio")
    df = cargar_series()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 3.0))

    hist = df.loc[2015:2025, "demanda_gwh"] / 1000
    a.plot(hist.index, hist, color=C["tinta2"], lw=1.6, label="Observada")
    a.plot(bt.index, bt["combinado"] / 1000, color=C["serie1"],
           label="Combinado VEC+VAR")
    a.plot(bt.index, bt["rw_deriva"] / 1000, color=C["silenciado"],
           ls=(0, (4, 3)), lw=1.5, label="Caminata aleatoria")
    a.set_title("(a) Backtest Colombia: entrenado\ncon 2000-2019", fontsize=9.5)
    a.set_ylabel("TWh-año")
    a.legend(fontsize=7.6, loc="upper left")

    b.plot(es.index, es["demanda_bc_gwh"] / 1000, color=C["serie1"],
           label="Medida (neta, b.c.)")
    b.plot(es.index, es["bruta_gwh"] / 1000, color=C["serie2"],
           label="Reconstruida (neta + BtM)")
    b.scatter([2025], [269.753], color=C["tinta"], zorder=5, s=22)
    b.annotate("REE 2025:\n«demanda +\nautoconsumo»", (2025, 269.753),
               xytext=(-8, -37), textcoords="offset points", fontsize=7.4,
               ha="right", color=C["tinta2"])
    b.set_title("(b) España: demanda medida vs.\nreconstruida (APPA/REE)",
                fontsize=9.5)
    b.set_ylabel("TWh-año")
    b.set_xticks([2020, 2021, 2022, 2023, 2024, 2025])
    b.legend(fontsize=7.6, loc="upper left")
    guardar(fig, "f7_validacion")


if __name__ == "__main__":
    f1_historia()
    f2_proyeccion()
    f3_capacidad()
    f4_espana()
    f5_curva_carga()
    f6_ajuste_vecm()
    f7_validacion()
