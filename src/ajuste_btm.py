# -*- coding: utf-8 -*-
"""
Etapa 2 del pipeline: módulo de ajuste Behind-the-Meter (BtM).

Construye la demanda bruta esperada 2026-2040 (demanda orgánica del SIN
proyectada en analisis_econometrico.py + cuña GCE+ME de la UPME) y la
ajusta con cuatro escenarios de penetración BtM:

  E0  Tendencial UPME     (rev. jul-2026: 3.331 MW a 2030; 7.182 MW a 2040)
  E1  Envolvente CREG 4%  (capacidad cuya energía anual equivale al 4 % de
                           la demanda comercial regulada; art. 12,
                           Res. CREG 101 072 de 2025)
  E2  Trayectoria España  (la participación BtM sobre demanda replica la
                           senda española: 4,1 % a los 7 años del despegue,
                           saturación logística en 9 %)
  E3  Potencial PNUMA     (7.424 MWp alcanzados en 2035 y estables)

Salidas: output/escenarios_btm.csv, output/tablas/t5*.csv, t6*.csv
"""
import numpy as np
import pandas as pd

from config import (FACTOR_PLANTA_FV, FRACCION_EXCEDENTES, HORIZONTE,
                    PART_REGULADO, SALIDAS, TABLAS, UMBRAL_CREG,
                    cargar_upme)

H = list(HORIZONTE)
GWH_POR_MW = 8760 * FACTOR_PLANTA_FV / 1000.0   # ≈1,45 GWh-año por MW


# ------------------------------------------------------ demanda bruta
def demanda_bruta():
    """SIN orgánico (modelo propio) + cuña incremental GCE+ME (UPME jul-2024).

    La serie histórica de XM sobre la que se estima el modelo ya contiene la
    demanda de los GCE existentes; por eso solo se agrega el COMPONENTE
    INCREMENTAL de la cuña GCE+ME de la UPME respecto de su nivel de 2025
    (nuevas grandes cargas y movilidad eléctrica), extrapolada a 2039-2040
    con el incremento medio de los últimos tres años.
    """
    organico = pd.read_csv(SALIDAS / "series_proyeccion.csv",
                           index_col="anio")["sin_organico_gwh"]
    upme = cargar_upme()
    cuna_total = upme["sin_gce_me_gwh_medio"] - upme["sin_gwh_medio"]
    cuna = (cuna_total - cuna_total.loc[2025]).loc[2026:]
    incremento = cuna.diff().tail(3).mean()
    for anio in (2039, 2040):
        cuna.loc[anio] = cuna.loc[anio - 1] + incremento
    bruta = organico + cuna.reindex(H).values
    return pd.DataFrame({"sin_organico_gwh": organico.round(0),
                         "cuna_gce_me_gwh": cuna.reindex(H).round(0),
                         "demanda_bruta_gwh": bruta.round(0)})


# -------------------------------------------------- escenarios BtM (MW)
def interp_geometrica(anclas):
    """Interpola geométricamente entre anclas {año: MW}."""
    anios = sorted(anclas)
    serie = {}
    for a0, a1 in zip(anios[:-1], anios[1:]):
        tasa = (anclas[a1] / anclas[a0]) ** (1 / (a1 - a0))
        for t in range(a0, a1):
            serie[t] = anclas[a0] * tasa ** (t - a0)
    serie[anios[-1]] = anclas[anios[-1]]
    return pd.Series(serie)


def esc_tendencial_upme():
    """E0: anclas UPME rev. jul-2026 (645 MW en 2025 según rev. ene-2026)."""
    return interp_geometrica({2025: 645, 2030: 3331, 2040: 7182}).loc[H]


def esc_creg_4pct(bruta):
    """E1: capacidad cuya generación anual equivale al 4 % de la demanda
    comercial regulada (interpretación conservadora del umbral del art. 12:
    toda la generación BtM se contabiliza contra el umbral)."""
    mr = PART_REGULADO * bruta["demanda_bruta_gwh"]
    return (UMBRAL_CREG * mr / GWH_POR_MW)


def esc_espana(bruta):
    """E2: participación BtM sobre demanda replica la senda de España.

    España: despegue 2019 → 4,1 % de la demanda nacional en 2025 (APPA).
    Colombia: despegue efectivo 2025 (1,1 % ya alcanzado) → 4,1 % en 2032,
    saturación logística s_max = 9 % (España seguiría esa senda con el
    ritmo actual de ~1,2 GW/año hacia 2032-2035).
    """
    s_max, s_2025, s_2032 = 0.09, 0.011, 0.041
    # logística s(t) = s_max / (1 + exp(-k (t - t0))) calibrada en 2 puntos
    def inv(s):
        return np.log(s_max / s - 1.0)
    k = (inv(s_2025) - inv(s_2032)) / (2032 - 2025)
    t0 = 2025 + inv(s_2025) / k
    t = np.array(H, dtype=float)
    s = s_max / (1.0 + np.exp(-k * (t - t0)))
    return pd.Series(s * bruta["demanda_bruta_gwh"].values / GWH_POR_MW,
                     index=H)


def esc_pnuma():
    """E3: el potencial de mercado PNUMA (7.424 MWp) se alcanza en 2035."""
    return interp_geometrica({2025: 645, 2035: 7424, 2040: 7424}).loc[H]


# ------------------------------------------------------------- síntesis
def main():
    bruta = demanda_bruta()
    esc = pd.DataFrame({
        "E0_upme_mw": esc_tendencial_upme(),
        "E1_creg4_mw": esc_creg_4pct(bruta),
        "E2_espana_mw": esc_espana(bruta),
        "E3_pnuma_mw": esc_pnuma(),
    }).round(0)

    res = bruta.copy()
    mr = PART_REGULADO * bruta["demanda_bruta_gwh"]
    for e in ["E0_upme", "E1_creg4", "E2_espana", "E3_pnuma"]:
        gen = esc[f"{e}_mw"] * GWH_POR_MW
        res[f"{e}_mw"] = esc[f"{e}_mw"]
        res[f"{e}_gen_gwh"] = gen.round(0)
        res[f"{e}_neta_gwh"] = (bruta["demanda_bruta_gwh"] - gen).round(0)
        # % del umbral CREG: energía entregada al SDL sobre demanda regulada
        res[f"{e}_pct_umbral_total"] = (100 * gen / mr).round(2)          # excedentes = 100 %
        res[f"{e}_pct_umbral_fe"] = (100 * FRACCION_EXCEDENTES * gen / mr).round(2)
    res.to_csv(SALIDAS / "escenarios_btm.csv")

    # --- tabla 5: capacidad, generación y umbral CREG por hitos
    hitos = [2026, 2030, 2035, 2040]
    t5 = []
    for e, nombre in [("E0_upme", "E0 Tendencial UPME"),
                      ("E1_creg4", "E1 Envolvente CREG 4%"),
                      ("E2_espana", "E2 Trayectoria España"),
                      ("E3_pnuma", "E3 Potencial PNUMA")]:
        fila = {"escenario": nombre}
        for h in hitos:
            fila[f"MW_{h}"] = int(res.loc[h, f"{e}_mw"])
            fila[f"pctMR_{h}"] = res.loc[h, f"{e}_pct_umbral_total"]
        cruces = res.index[res[f"{e}_pct_umbral_total"] > 4.0]
        fila["cruce_umbral_4pct"] = int(cruces[0]) if len(cruces) else "no cruza"
        cruces_fe = res.index[res[f"{e}_pct_umbral_fe"] > 4.0]
        fila["cruce_umbral_fe40"] = int(cruces_fe[0]) if len(cruces_fe) else "no cruza"
        t5.append(fila)
    pd.DataFrame(t5).to_csv(TABLAS / "t5_escenarios_umbral.csv", index=False)

    # --- tabla 6: demanda neta vs bruta y TCAC
    base25 = 84235.0
    t6 = []
    for e, nombre in [("bruta", "Demanda bruta (sin BtM)"),
                      ("E0_upme", "Neta E0 UPME"), ("E1_creg4", "Neta E1 CREG 4%"),
                      ("E2_espana", "Neta E2 España"), ("E3_pnuma", "Neta E3 PNUMA")]:
        col = "demanda_bruta_gwh" if e == "bruta" else f"{e}_neta_gwh"
        serie = res[col]
        t6.append({
            "trayectoria": nombre,
            "gwh_2030": int(serie.loc[2030]), "gwh_2035": int(serie.loc[2035]),
            "gwh_2040": int(serie.loc[2040]),
            "tcac_2025_2040_pct": round(100 * ((serie.loc[2040] / base25) ** (1 / 15) - 1), 2),
            "enmascarado_2040_gwh": int(res["demanda_bruta_gwh"].loc[2040] - serie.loc[2040]),
        })
    pd.DataFrame(t6).to_csv(TABLAS / "t6_demanda_neta.csv", index=False)

    print(res[["demanda_bruta_gwh", "E0_upme_neta_gwh", "E2_espana_neta_gwh",
               "E3_pnuma_neta_gwh"]].loc[[2026, 2030, 2035, 2040]])
    print("\nUmbral CREG (energía total BtM / demanda MR), % :")
    print(res[[c for c in res.columns if "pct_umbral_total" in c]].loc[[2026, 2030, 2035, 2040]])
    return res


if __name__ == "__main__":
    main()
