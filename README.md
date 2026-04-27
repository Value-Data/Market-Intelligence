# Market-Intelligence — Análisis Causal de Brokers (China)
.
App de Streamlit que visualiza el efecto causal (CATE) de cada broker
sobre el retorno (USD/KG) frente al broker de referencia
**Forever Fresh (Shanghai) Fruit**, usando modelos `CausalForestDML`.

## Estructura

- `app_china_cate.py` — aplicación Streamlit.
- `df_cate_*.csv` — resultados pre-computados desde el notebook (fuente de datos del app).
- `EDA_China.ipynb` — análisis exploratorio y entrenamiento de modelos.
- `informe_analisis_causal_china.pdf` — informe con hallazgos.
- `ranking_brokers_general.csv`, `variedad_semana_broker_vs_ff.csv` — tablas auxiliares.

La app lee los CSVs pre-computados desde el notebook (`EDA_China.ipynb`).
Los modelos entrenados (`*.pkl`) no están versionados ni se necesitan para
ejecutar la app: pesan >1 GB cada uno (límite de GitHub: 100 MB por archivo).
Si quieres recomputar los CSVs desde cero, corre el notebook.

## Correr en local

```bash
pip install -r requirements.txt
streamlit run app_china_cate.py
```

## Tabs disponibles

1. **Broker × Semana** — modelos binarios (uno por broker vs referencia).
2. **Broker × Variedad** — heterogeneidad por variedad comercial.
3. **Broker × Serie** — heterogeneidad por calibre (serie).
4. **Broker × Etiqueta** — heterogeneidad por marca.
5. **Broker × Calibre** — heterogeneidad por tamaño de fruta.
6. **Broker × Codenvase** — heterogeneidad por tipo de envase.

## Filtros

- **Ocultar valores inciertos** — oculta celdas donde el IC 95 % cruza 0.
- **Solo MEJOR / PEOR** — filtra dirección del efecto.
- **Registros mínimos por broker (N)** — excluye brokers con baja muestra.
- **Temporadas mínimas (N_TEMP)** — excluye brokers con pocas temporadas.
- **Soporte mínimo por celda (N_OBS)** — oculta celdas con pocas observaciones reales.
