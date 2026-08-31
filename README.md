# Integración del efecto Behind-the-Meter (BtM) en la estimación de demanda de largo plazo, Colombia

**Proyecto 1, ILE4100 Planeamiento de Sistemas de Potencia**

Código fuente, datos y salidas del informe *"Integración del efecto de las
soluciones Behind-the-Meter en las prácticas de estimación de demanda a
largo plazo del sistema eléctrico colombiano"*.

## Autores

- Tomas Castro Roa
- William David Velasquez Baron
- Carlos Fernando Diaz Vargas

## Estructura

```
├── data/                       # series consolidadas y parámetros (con fuentes)
│   ├── demanda_pib_anual.csv   # demanda SIN (GWh) y PIB real, 2000-2025
│   ├── upme_proyeccion_jul2024.csv  # proyección oficial UPME (esc. medio)
│   └── parametros_btm.csv      # parámetros BtM (CREG, APPA, PNUMA, REE...)
├── src/
│   ├── config.py               # rutas, paleta y supuestos
│   ├── analisis_econometrico.py# ADF/KPSS, Johansen, Engle-Granger, VECM, VAR,
│   │                           # combinación de pronósticos (estilo UPME)
│   ├── ajuste_btm.py           # demanda bruta y escenarios BtM E0-E3
│   ├── validacion.py           # V1 backtest CO, V2 experimento España,
│   │                           # V3 panel LSDV Colombia-España (H0: c=-1)
│   ├── generar_figuras.py      # figuras F1-F7 del informe
│   └── descargar_datos.py      # (opcional) refresco desde XM/Banco Mundial/UPME
├── output/                     # tablas CSV y figuras generadas
└── informe/                    # fuentes LaTeX del informe
```

## Reproducción

```bash
pip install -r requirements.txt
cd src
python analisis_econometrico.py   # 1) econometría y proyección orgánica
python ajuste_btm.py              # 2) escenarios BtM y demanda neta
python validacion.py              # 3) V1 backtest, V2 España, V3 panel
python generar_figuras.py         # 4) figuras
cd ../informe && latexmk -pdf informe_btm.tex   # 5) informe (opcional)
```

Los cuatro scripts corren en menos de un minuto con los CSV incluidos.
`descargar_datos.py` permite regenerar las series desde las fuentes
primarias (API de XM vía `pydataxm`, API del Banco Mundial y anexo Excel
oficial de la UPME rev. jul-2026); requiere conexión a internet.

## Datos y fuentes principales

| Serie / parámetro | Fuente |
|---|---|
| Demanda SIN 2017-2025 | XM (informes anuales y comunicados; 2025 = 2024×1,0262) |
| Demanda SIN 2000-2016 | Compilación de informes anuales XM (regenerable con `descargar_datos.py`) |
| PIB real (COP constantes 2015) | DANE / Banco Mundial `NY.GDP.MKTP.KN` |
| Proyección oficial 2024-2038 | UPME, *Proyección de demanda EE y potencia máxima*, rev. jul-2024 |
| Tasas y GD 2026-2040 | UPME, rev. jul-2026 (2,78 % SIN; GD 3.331 MW-2030 / 7.182 MW-2040) |
| Umbral 4 % | Res. CREG 101 072 de 2025 (art. 12) y Doc. CREG 901 099 de 2024 |
| España BtM | APPA, *Informe anual del autoconsumo fotovoltaico y almacenamiento 2025* |
| Punta España | REE (máximo 2025: 40.070 MW, 15-ene 20:57) |

## Publicar en GitHub
El enlace del repositorio se referencia en el Anexo A del informe.
