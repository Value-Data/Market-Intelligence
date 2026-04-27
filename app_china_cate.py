"""
Streamlit App — Efecto Causal (CATE) de Brokers en Mercado China
Carga los modelos .pkl de CausalForestDML y genera visualizaciones interactivas.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# ── Configuracion de pagina ───────────────────────────────────────────
st.set_page_config(
    page_title="CATE China - Analisis Causal de Brokers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BROKER_REF_DEFAULT = "Forever Fresh (Shanghai) Fruit"

# ── Colores ───────────────────────────────────────────────────────────
COLOR_MEJOR = "#2ecc71"
COLOR_PEOR = "#e74c3c"
COLOR_INCIERTO = "#95a5a6"


# =====================================================================
# CARGA DE CATE DESDE CSV
# =====================================================================
@st.cache_data(show_spinner="Cargando CATE Broker x Semana...")
def load_cate_brokers_semana():
    csv_path = os.path.join(BASE_DIR, "df_cate_brokers.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["SEMANA"] = df["SEMANA"].astype(int)
    for col in ("N", "N_TEMP"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


_FIXED_COLS = {
    "BROKER", "SEMANA", "CATE", "CI_LO", "CI_HI",
    "N", "N_TEMP", "DIRECCION", "N_OBS", "IC_WIDTH",
}


def _load_cate_dim(dimension_col, csv_name):
    csv_path = os.path.join(BASE_DIR, csv_name)
    if not os.path.exists(csv_path):
        return pd.DataFrame(), [], []
    df = pd.read_csv(csv_path)
    if dimension_col not in df.columns:
        candidatos = [c for c in df.columns if c not in _FIXED_COLS]
        if not candidatos:
            return pd.DataFrame(), [], []
        df = df.rename(columns={candidatos[0]: dimension_col})
    df["SEMANA"] = df["SEMANA"].astype(int)
    dim_values_all = sorted(df[dimension_col].dropna().unique().tolist())
    brokers_alt = sorted(df["BROKER"].dropna().unique().tolist())
    return df, dim_values_all, brokers_alt


@st.cache_data(show_spinner="Cargando CATE Broker x Variedad x Semana...")
def load_cate_variedad():
    return _load_cate_dim("VARIEDAD", "df_cate_variedad.csv")


@st.cache_data(show_spinner="Cargando CATE Broker x Serie x Semana...")
def load_cate_serie():
    return _load_cate_dim("SERIE", "df_cate_serie.csv")


@st.cache_data(show_spinner="Cargando CATE Broker x Etiqueta x Semana...")
def load_cate_etiqueta():
    return _load_cate_dim("ETIQUETA", "df_cate_etiqueta.csv")


@st.cache_data(show_spinner="Cargando CATE Broker x Calibre x Semana...")
def load_cate_calibre():
    return _load_cate_dim("CALIBRE", "df_cate_calibre.csv")


@st.cache_data(show_spinner="Cargando CATE Broker x Codenvase x Semana...")
def load_cate_codenvase():
    return _load_cate_dim("CODENVASE", "df_cate_codenvase.csv")


# =====================================================================
# FILTROS DE SIGNIFICANCIA
# =====================================================================
def apply_significance_filter(df, ocultar_inciertos):
    """Pone CATE=NaN donde DIRECCION es INCIERTO si el filtro esta activo."""
    if not ocultar_inciertos:
        return df
    df = df.copy()
    df.loc[df["DIRECCION"] == "INCIERTO", "CATE"] = np.nan
    return df


def filter_by_min_n(df, min_n):
    """Filtra brokers con menos de min_n registros (solo si la col N existe)."""
    if "N" not in df.columns or min_n <= 0:
        return df
    return df[df["N"] >= min_n]


def filter_by_min_nobs(df, min_nobs):
    """Oculta celdas con N_OBS < min_nobs (CATE=NaN, DIRECCION='SIN SOPORTE')."""
    if "N_OBS" not in df.columns or min_nobs <= 0:
        return df
    df = df.copy()
    mask = df["N_OBS"] < min_nobs
    df.loc[mask, "CATE"] = np.nan
    df.loc[mask, "DIRECCION"] = "SIN SOPORTE"
    return df


# =====================================================================
# FUNCIONES DE VISUALIZACION
# =====================================================================
LOW_SUPPORT_THRESHOLD = 50


def make_heatmap(
    pivot,
    pivot_dir,
    title,
    xlabel="Semana vs CNY",
    ylabel="",
    pivot_nobs=None,
):
    """
    Heatmap interactivo con escala divergente RdYlGn.
    Las celdas NaN (inciertos filtrados) quedan en blanco.
    Las celdas INCIERTO no filtradas se anotan con '?'.
    Las celdas con N_OBS bajo (10-50) se anotan con '~'.
    """
    vals = pivot.values.copy().astype(float)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        max_abs = 1.0
    else:
        max_abs = max(abs(valid.min()), abs(valid.max()), 0.01)

    def _nobs_at(i, j):
        if pivot_nobs is None:
            return None
        if i >= pivot_nobs.shape[0] or j >= pivot_nobs.shape[1]:
            return None
        v = pivot_nobs.iloc[i, j]
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    text_vals = []
    for i in range(pivot.shape[0]):
        row = []
        for j in range(pivot.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                row.append("")
                continue

            markers = []
            if (
                pivot_dir is not None
                and i < pivot_dir.shape[0]
                and j < pivot_dir.shape[1]
                and pivot_dir.iloc[i, j] == "INCIERTO"
            ):
                markers.append("?")

            n = _nobs_at(i, j)
            if n is not None and not np.isnan(n) and n < LOW_SUPPORT_THRESHOLD:
                markers.append("~")

            row.append(f"{v:.2f}" + (" " + " ".join(markers) if markers else ""))
        text_vals.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=vals,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{y}</b><br>"
                + xlabel
                + ": %{x}<br>"
                + "CATE: %{z:.3f} USD/KG<extra></extra>"
            ),
            colorbar=dict(title="CATE<br>(USD/KG)"),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=max(400, len(pivot) * 45 + 120),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="white",
    )
    return fig


def make_line_chart(df_broker, broker_name, ref_name):
    """Linea CATE + banda de confianza para un broker."""
    df_plot = df_broker.dropna(subset=["CATE"]).sort_values("SEMANA")
    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos significativos", showarrow=False)
        return fig

    fig = go.Figure()

    # Banda de confianza
    fig.add_trace(
        go.Scatter(
            x=list(df_plot["SEMANA"]) + list(df_plot["SEMANA"][::-1]),
            y=list(df_plot["CI_HI"]) + list(df_plot["CI_LO"][::-1]),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 95%",
            hoverinfo="skip",
        )
    )

    # Linea CATE
    colors = [
        COLOR_MEJOR if d == "MEJOR" else COLOR_PEOR if d == "PEOR" else COLOR_INCIERTO
        for d in df_plot["DIRECCION"]
    ]
    fig.add_trace(
        go.Scatter(
            x=df_plot["SEMANA"],
            y=df_plot["CATE"],
            mode="lines+markers",
            name="CATE",
            line=dict(color="#2980b9", width=2),
            marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
            hovertemplate=(
                "Semana %{x}<br>"
                "CATE: %{y:.3f} USD/KG<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"{broker_name} vs {ref_name}",
        xaxis_title="Semanas vs CNY",
        yaxis_title="CATE (USD/KG)",
        height=400,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def make_bar_chart(df_semana, semana, title_extra=""):
    """Barras CATE por broker para una semana fija. Filtra NaN."""
    df_sorted = df_semana.dropna(subset=["CATE"]).sort_values("CATE", ascending=True)
    if df_sorted.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos significativos para esta semana", showarrow=False)
        return fig

    colors = [
        COLOR_MEJOR if d == "MEJOR" else COLOR_PEOR if d == "PEOR" else COLOR_INCIERTO
        for d in df_sorted["DIRECCION"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["CATE"],
            y=df_sorted["BROKER"],
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data",
                symmetric=False,
                array=(df_sorted["CI_HI"] - df_sorted["CATE"]).tolist(),
                arrayminus=(df_sorted["CATE"] - df_sorted["CI_LO"]).tolist(),
                color="rgba(0,0,0,0.3)",
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "CATE: %{x:.3f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"CATE por Broker - Semana {semana} {title_extra}",
        xaxis_title="CATE (USD/KG vs Forever Fresh)",
        height=max(400, len(df_sorted) * 30 + 100),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


# =====================================================================
# APP PRINCIPAL
# =====================================================================
def main():
    # ── Sidebar ───────────────────────────────────────────────────────
    st.sidebar.title("Analisis Causal China")
    st.sidebar.markdown(
        """
        **Modelos CausalForestDML**

        Referencia: **Forever Fresh (Shanghai) Fruit**

        Los valores CATE representan el efecto causal
        de usar un broker alternativo vs la referencia,
        medido en **USD/KG** sobre el retorno.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros de significancia")

    min_nobs = st.sidebar.number_input(
        "Soporte minimo por celda (N_OBS):",
        min_value=0,
        max_value=2000,
        value=10,
        step=10,
        help=(
            "Oculta celdas con menos observaciones reales que este umbral. "
            "10 = default del notebook. 50+ = solo celdas con magnitud "
            "confiable (IC estrecho)."
        ),
    )

    ocultar_inciertos = True
    solo_significativos = "Todos (significativos)"
    min_n = 0
    min_temporadas = 0

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **Interpretacion de colores:**
        - :green[**Verde**] = MEJOR que Forever Fresh (IC > 0)
        - :red[**Rojo**] = PEOR que Forever Fresh (IC < 0)
        - **Blanco/vacio** = Incierto (filtrado) o sin soporte (N_OBS<10)

        **Marcadores en celdas:**
        - `?` = IC cruza 0 (no significativo)
        - `~` = Soporte bajo (N_OBS < {LOW_SUPPORT_THRESHOLD}) — dirección
          confiable pero magnitud ruidosa
        """
    )

    # Guardar filtros en dict para pasar a las funciones
    filtros = {
        "ocultar_inciertos": ocultar_inciertos,
        "solo_significativos": solo_significativos,
        "min_n": min_n,
        "min_temporadas": min_temporadas,
        "min_nobs": min_nobs,
    }

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab4, tab5, tab6 = st.tabs(
        [
            "Broker x Semana",
            "Broker x Variedad",
            "Broker x Etiqueta",
            "Broker x Calibre",
            "Broker x Codenvase",
        ]
    )

    # ==================================================================
    # TAB 1: BROKER x SEMANA (modelos binarios)
    # ==================================================================
    with tab1:
        st.header("Efecto Causal por Broker y Semana vs CNY")
        st.caption(
            "Modelos binarios (1 por broker). Cada broker se compara "
            "individualmente contra Forever Fresh."
        )

        df_bs = load_cate_brokers_semana()

        # Aplicar filtros globales
        df_bs = filter_by_min_n(df_bs, filtros["min_n"])
        if filtros["min_temporadas"] > 0 and "N_TEMP" in df_bs.columns:
            df_bs = df_bs[df_bs["N_TEMP"] >= filtros["min_temporadas"]]
        df_bs = filter_by_min_nobs(df_bs, filtros["min_nobs"])

        if filtros["solo_significativos"] == "Solo MEJOR":
            df_bs = df_bs[df_bs["DIRECCION"].isin(["MEJOR", "INCIERTO"])]
        elif filtros["solo_significativos"] == "Solo PEOR":
            df_bs = df_bs[df_bs["DIRECCION"].isin(["PEOR", "INCIERTO"])]

        df_bs_display = apply_significance_filter(df_bs, filtros["ocultar_inciertos"])

        # ── Filtros de tab ────────────────────────────────────────────
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            brokers_disponibles = sorted(df_bs["BROKER"].unique())
            brokers_sel = st.multiselect(
                "Filtrar brokers:",
                brokers_disponibles,
                default=brokers_disponibles,
                key="bs_brokers",
            )
        with col_f2:
            if df_bs["SEMANA"].nunique() > 0:
                semanas_range = st.slider(
                    "Rango de semanas:",
                    int(df_bs["SEMANA"].min()),
                    int(df_bs["SEMANA"].max()),
                    (int(df_bs["SEMANA"].min()), int(df_bs["SEMANA"].max())),
                    key="bs_semanas",
                )
            else:
                semanas_range = (0, 0)

        df_filt = df_bs_display[
            (df_bs_display["BROKER"].isin(brokers_sel))
            & (df_bs_display["SEMANA"] >= semanas_range[0])
            & (df_bs_display["SEMANA"] <= semanas_range[1])
        ]
        # Version sin filtro de inciertos para DIRECCION en heatmap
        df_filt_raw = df_bs[
            (df_bs["BROKER"].isin(brokers_sel))
            & (df_bs["SEMANA"] >= semanas_range[0])
            & (df_bs["SEMANA"] <= semanas_range[1])
        ]

        if df_filt.empty:
            st.warning("No hay datos para los filtros seleccionados.")
        elif df_filt["CATE"].dropna().empty:
            st.warning(
                "No hay datos significativos con los filtros actuales. "
                "Reduce el soporte mínimo (N_OBS) o permite valores inciertos."
            )
        else:
            # Metricas resumen
            n_sig = (df_filt_raw["DIRECCION"] != "INCIERTO").sum()
            n_total = len(df_filt_raw)
            n_mejor = (df_filt_raw["DIRECCION"] == "MEJOR").sum()
            n_peor = (df_filt_raw["DIRECCION"] == "PEOR").sum()
            n_incierto = (df_filt_raw["DIRECCION"] == "INCIERTO").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Celdas significativas", f"{n_sig}/{n_total}")
            c2.metric("MEJOR", n_mejor)
            c3.metric("PEOR", n_peor)
            c4.metric("INCIERTO (filtrado)", n_incierto)

            # ── Heatmap ───────────────────────────────────────────────
            pivot = df_filt.pivot_table(
                index="BROKER", columns="SEMANA", values="CATE"
            )
            pivot_dir = df_filt_raw.pivot_table(
                index="BROKER", columns="SEMANA",
                values="DIRECCION", aggfunc="first"
            )
            pivot_nobs = None
            if "N_OBS" in df_filt_raw.columns:
                pivot_nobs = df_filt_raw.pivot_table(
                    index="BROKER", columns="SEMANA",
                    values="N_OBS", aggfunc="first"
                )

            # Ordenar por CATE promedio (ignorando NaN)
            orden = pivot.apply(lambda r: r.dropna().mean() if r.notna().any() else -999, axis=1)
            orden = orden.sort_values(ascending=False).index
            pivot = pivot.reindex(orden)
            pivot_dir = pivot_dir.reindex(orden)
            if pivot_nobs is not None:
                pivot_nobs = pivot_nobs.reindex(orden)

            st.plotly_chart(
                make_heatmap(
                    pivot,
                    pivot_dir,
                    "CATE por Broker x Semana vs CNY (Modelos Binarios)",
                    pivot_nobs=pivot_nobs,
                ),
                use_container_width=True,
            )

            # ── Detalle por broker ────────────────────────────────────
            st.subheader("Detalle por Broker")
            broker_detalle = st.selectbox(
                "Seleccionar broker para ver detalle:",
                sorted(df_filt_raw["BROKER"].unique()),
                key="bs_detalle",
            )

            df_det = df_filt_raw[df_filt_raw["BROKER"] == broker_detalle].sort_values(
                "SEMANA"
            )
            if not df_det.empty:
                col_chart, col_info = st.columns([3, 1])
                with col_chart:
                    st.plotly_chart(
                        make_line_chart(df_det, broker_detalle, BROKER_REF_DEFAULT),
                        use_container_width=True,
                    )
                with col_info:
                    if "N" in df_det.columns and not df_det.empty:
                        st.metric("Registros", f"{int(df_det['N'].iloc[0]):,}")
                    if "N_TEMP" in df_det.columns and not df_det.empty:
                        st.metric("Temporadas", int(df_det["N_TEMP"].iloc[0]))
                    n_sig_b = (df_det["DIRECCION"] != "INCIERTO").sum()
                    st.metric("Semanas significativas", f"{n_sig_b}/{len(df_det)}")
                    df_sig = df_det[df_det["DIRECCION"] != "INCIERTO"]
                    if not df_sig.empty:
                        ate_prom = df_sig["CATE"].mean()
                        st.metric(
                            "CATE Prom. (sig.)",
                            f"{ate_prom:+.3f}",
                        )
                        mejor_sem = df_sig.loc[df_sig["CATE"].idxmax()]
                        st.metric(
                            "Mejor Semana",
                            f"Sem {int(mejor_sem['SEMANA'])}",
                            f"{mejor_sem['CATE']:+.3f}",
                        )

            # ── Barras por semana seleccionada ────────────────────────
            st.subheader("Comparativa por Semana")
            semana_sel = st.select_slider(
                "Seleccionar semana:",
                options=sorted(df_filt["SEMANA"].unique()),
                key="bs_sem_bar",
            )
            df_sem = df_filt[df_filt["SEMANA"] == semana_sel]
            st.plotly_chart(
                make_bar_chart(df_sem, semana_sel, "(Modelo Binario)"),
                use_container_width=True,
            )

    # ==================================================================
    # TAB 2: BROKER x VARIEDAD x SEMANA (multiclass)
    # ==================================================================
    with tab2:
        st.header("Efecto Causal por Broker, Variedad y Semana")
        st.caption(
            "Modelo multiclass: todos los brokers en un solo modelo. "
            "Heterogeneidad por VARIEDAD COMERCIAL."
        )

        df_var, variedades, brokers_alt_v = load_cate_variedad()
        if df_var.empty:
            st.warning("No se encontró df_cate_variedad.csv.")
        else:
            _render_multiclass_tab(
                df_var,
                dim_col="VARIEDAD",
                dim_values=variedades,
                brokers_alt=brokers_alt_v,
                broker_ref=BROKER_REF_DEFAULT,
                tab_key="var",
                filtros=filtros,
            )

    # ==================================================================
    # TAB 4: BROKER x ETIQUETA x SEMANA (multiclass)
    # ==================================================================
    with tab4:
        st.header("Efecto Causal por Broker, Etiqueta y Semana")
        st.caption(
            "Modelo multiclass: heterogeneidad por ETIQUETA (marca del producto)."
        )

        df_etq, etiquetas, brokers_alt_e = load_cate_etiqueta()
        if df_etq.empty:
            st.warning("No se encontró df_cate_etiqueta.csv.")
        else:
            _render_multiclass_tab(
                df_etq,
                dim_col="ETIQUETA",
                dim_values=etiquetas,
                brokers_alt=brokers_alt_e,
                broker_ref=BROKER_REF_DEFAULT,
                tab_key="etq",
                filtros=filtros,
            )

    # ==================================================================
    # TAB 5: BROKER x CALIBRE x SEMANA (multiclass)
    # ==================================================================
    with tab5:
        st.header("Efecto Causal por Broker, Calibre y Semana")
        st.caption(
            "Modelo multiclass: heterogeneidad por CALIBRE (tamaño de la fruta)."
        )

        df_cal, calibres, brokers_alt_c = load_cate_calibre()
        if df_cal.empty:
            st.warning("No se encontró df_cate_calibre.csv.")
        else:
            _render_multiclass_tab(
                df_cal,
                dim_col="CALIBRE",
                dim_values=calibres,
                brokers_alt=brokers_alt_c,
                broker_ref=BROKER_REF_DEFAULT,
                tab_key="cal",
                filtros=filtros,
            )

    # ==================================================================
    # TAB 6: BROKER x CODENVASE x SEMANA (multiclass)
    # ==================================================================
    with tab6:
        st.header("Efecto Causal por Broker, Codenvase y Semana")
        st.caption(
            "Modelo multiclass: heterogeneidad por CODENVASE (tipo de envase)."
        )

        df_cod, codenvases, brokers_alt_cv = load_cate_codenvase()
        if df_cod.empty:
            st.warning("No se encontró df_cate_codenvase.csv.")
        else:
            _render_multiclass_tab(
                df_cod,
                dim_col="CODENVASE",
                dim_values=codenvases,
                brokers_alt=brokers_alt_cv,
                broker_ref=BROKER_REF_DEFAULT,
                tab_key="cod",
                filtros=filtros,
            )


def _render_multiclass_tab(
    df, dim_col, dim_values, brokers_alt, broker_ref, tab_key, filtros
):
    """Renderiza un tab multiclass generico (variedad, serie o etiqueta)."""

    ocultar = filtros["ocultar_inciertos"]
    solo = filtros["solo_significativos"]

    # Filtro de soporte minimo por celda
    df = filter_by_min_nobs(df, filtros.get("min_nobs", 0))

    # Aplicar filtro de direccion global
    df_raw = df.copy()
    if solo == "Solo MEJOR":
        # Mantener MEJOR y blanquear todo lo demas
        df_raw = df_raw[df_raw["DIRECCION"].isin(["MEJOR", "INCIERTO"])]
    elif solo == "Solo PEOR":
        df_raw = df_raw[df_raw["DIRECCION"].isin(["PEOR", "INCIERTO"])]

    df_display = apply_significance_filter(df_raw, ocultar)

    if df_raw.empty:
        st.warning(
            "No hay datos para los filtros seleccionados. "
            "Reduce el soporte mínimo (N_OBS) o ajusta los filtros de la barra lateral."
        )
        return

    vista = st.radio(
        "Tipo de vista:",
        ["Por " + dim_col, "Por Broker", "Consolidado"],
        horizontal=True,
        key=f"{tab_key}_vista",
    )

    if vista == f"Por {dim_col}":
        # ── Seleccionar dimension → heatmap brokers x semanas ─────
        dim_sel = st.selectbox(
            f"Seleccionar {dim_col}:",
            sorted(dim_values),
            key=f"{tab_key}_dim",
        )

        df_dim = df_display[df_display[dim_col] == dim_sel]
        df_dim_raw = df_raw[df_raw[dim_col] == dim_sel]

        # Filtro de semanas
        sem_min = int(df_dim_raw["SEMANA"].min()) if not df_dim_raw.empty else 3
        sem_max = int(df_dim_raw["SEMANA"].max()) if not df_dim_raw.empty else 15
        sem_range = st.slider(
            "Rango de semanas:",
            sem_min,
            sem_max,
            (sem_min, sem_max),
            key=f"{tab_key}_sem_range1",
        )
        df_dim = df_dim[
            (df_dim["SEMANA"] >= sem_range[0])
            & (df_dim["SEMANA"] <= sem_range[1])
        ]
        df_dim_raw = df_dim_raw[
            (df_dim_raw["SEMANA"] >= sem_range[0])
            & (df_dim_raw["SEMANA"] <= sem_range[1])
        ]

        if df_dim_raw.empty:
            st.warning("No hay datos para la seleccion.")
            return

        if df_dim["CATE"].dropna().empty:
            st.warning(
                "No hay datos significativos con los filtros actuales. "
                "Reduce el soporte mínimo (N_OBS) o permite valores inciertos."
            )
            return

        # Metricas
        n_total = len(df_dim_raw)
        n_sig = (df_dim_raw["DIRECCION"] != "INCIERTO").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Celdas significativas", f"{n_sig}/{n_total}")
        c2.metric("MEJOR", (df_dim_raw["DIRECCION"] == "MEJOR").sum())
        c3.metric("PEOR", (df_dim_raw["DIRECCION"] == "PEOR").sum())

        # Heatmap
        pivot = df_dim.pivot_table(
            index="BROKER", columns="SEMANA", values="CATE"
        )
        pivot_dir = df_dim_raw.pivot_table(
            index="BROKER", columns="SEMANA",
            values="DIRECCION", aggfunc="first"
        )
        pivot_nobs = None
        if "N_OBS" in df_dim_raw.columns:
            pivot_nobs = df_dim_raw.pivot_table(
                index="BROKER", columns="SEMANA",
                values="N_OBS", aggfunc="first"
            )
        # Agregar referencia como fila de ceros
        pivot.loc[broker_ref] = 0
        orden = pivot.apply(
            lambda r: r.dropna().mean() if r.notna().any() else -999, axis=1
        ).sort_values(ascending=False).index
        pivot = pivot.reindex(orden)
        pivot_dir = pivot_dir.reindex(orden).fillna("")
        if pivot_nobs is not None:
            pivot_nobs = pivot_nobs.reindex(orden)

        st.plotly_chart(
            make_heatmap(
                pivot,
                pivot_dir,
                f"CATE por Broker x Semana - {dim_col}: {dim_sel}",
                pivot_nobs=pivot_nobs,
            ),
            use_container_width=True,
        )

        # Barras por semana
        st.subheader("Comparativa por Semana")
        semana_sel = st.select_slider(
            "Semana:",
            options=sorted(df_dim["SEMANA"].unique()),
            key=f"{tab_key}_sem_bar1",
        )
        df_sem = df_dim[df_dim["SEMANA"] == semana_sel]
        st.plotly_chart(
            make_bar_chart(
                df_sem, semana_sel, f"| {dim_col}: {dim_sel}"
            ),
            use_container_width=True,
        )

        # ── Global: CATE promedio por broker (todas las semanas filtradas) ─
        st.subheader(f"Global - CATE promedio por Broker | {dim_col}: {dim_sel}")
        st.caption(
            f"Promedio sobre semanas {sem_range[0]}-{sem_range[1]} "
            "(solo celdas significativas con soporte)."
        )
        df_global = df_dim.dropna(subset=["CATE"])
        if df_global.empty:
            st.info("Sin datos significativos para el consolidado.")
        else:
            ranking_global = (
                df_global.groupby("BROKER")
                .agg(
                    CATE=("CATE", "mean"),
                    CI_LO=("CI_LO", "mean"),
                    CI_HI=("CI_HI", "mean"),
                    N_SEM=("CATE", "count"),
                )
                .reset_index()
                .sort_values("CATE", ascending=True)
            )
            colors_g = [
                COLOR_MEJOR if v > 0 else COLOR_PEOR
                for v in ranking_global["CATE"]
            ]
            fig_global = go.Figure(
                go.Bar(
                    x=ranking_global["CATE"],
                    y=ranking_global["BROKER"],
                    orientation="h",
                    marker_color=colors_g,
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(ranking_global["CI_HI"] - ranking_global["CATE"]).tolist(),
                        arrayminus=(ranking_global["CATE"] - ranking_global["CI_LO"]).tolist(),
                        color="rgba(0,0,0,0.3)",
                    ),
                    customdata=ranking_global["N_SEM"],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "CATE prom: %{x:.3f}<br>"
                        "Semanas significativas: %{customdata}<extra></extra>"
                    ),
                )
            )
            fig_global.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_global.update_layout(
                title=f"CATE Promedio por Broker - {dim_col}: {dim_sel}",
                xaxis_title="CATE promedio (USD/KG vs Forever Fresh)",
                height=max(400, len(ranking_global) * 30 + 100),
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_global, use_container_width=True)

        # Detalle linea por broker
        st.subheader("Detalle por Broker")
        broker_det = st.selectbox(
            "Broker:",
            sorted(brokers_alt),
            key=f"{tab_key}_broker_det1",
        )
        df_det = df_dim_raw[df_dim_raw["BROKER"] == broker_det].sort_values("SEMANA")
        if not df_det.empty:
            st.plotly_chart(
                make_line_chart(df_det, broker_det, broker_ref),
                use_container_width=True,
            )

    elif vista == "Por Broker":
        # ── Seleccionar broker → heatmap dims x semanas ───────────
        broker_sel = st.selectbox(
            "Seleccionar Broker:",
            sorted(brokers_alt),
            key=f"{tab_key}_broker",
        )

        df_bk = df_display[df_display["BROKER"] == broker_sel]
        df_bk_raw = df_raw[df_raw["BROKER"] == broker_sel]

        # Filtro de semanas
        sem_min = int(df_bk_raw["SEMANA"].min()) if not df_bk_raw.empty else 3
        sem_max = int(df_bk_raw["SEMANA"].max()) if not df_bk_raw.empty else 15
        sem_range = st.slider(
            "Rango de semanas:",
            sem_min,
            sem_max,
            (sem_min, sem_max),
            key=f"{tab_key}_sem_range2",
        )
        df_bk = df_bk[
            (df_bk["SEMANA"] >= sem_range[0])
            & (df_bk["SEMANA"] <= sem_range[1])
        ]
        df_bk_raw = df_bk_raw[
            (df_bk_raw["SEMANA"] >= sem_range[0])
            & (df_bk_raw["SEMANA"] <= sem_range[1])
        ]

        if df_bk_raw.empty:
            st.warning("No hay datos para la seleccion.")
            return

        if df_bk["CATE"].dropna().empty:
            st.warning(
                "No hay datos significativos con los filtros actuales. "
                "Reduce el soporte mínimo (N_OBS) o permite valores inciertos."
            )
            return

        # Metricas
        n_total = len(df_bk_raw)
        n_sig = (df_bk_raw["DIRECCION"] != "INCIERTO").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Celdas significativas", f"{n_sig}/{n_total}")
        c2.metric("MEJOR", (df_bk_raw["DIRECCION"] == "MEJOR").sum())
        c3.metric("PEOR", (df_bk_raw["DIRECCION"] == "PEOR").sum())

        # Heatmap dim x semana
        pivot = df_bk.pivot_table(
            index=dim_col, columns="SEMANA", values="CATE"
        )
        pivot_dir = df_bk_raw.pivot_table(
            index=dim_col, columns="SEMANA",
            values="DIRECCION", aggfunc="first"
        )
        pivot_nobs = None
        if "N_OBS" in df_bk_raw.columns:
            pivot_nobs = df_bk_raw.pivot_table(
                index=dim_col, columns="SEMANA",
                values="N_OBS", aggfunc="first"
            )
        orden = pivot.apply(
            lambda r: r.dropna().mean() if r.notna().any() else -999, axis=1
        ).sort_values(ascending=False).index
        pivot = pivot.reindex(orden)
        pivot_dir = pivot_dir.reindex(orden).fillna("")
        if pivot_nobs is not None:
            pivot_nobs = pivot_nobs.reindex(orden)

        st.plotly_chart(
            make_heatmap(
                pivot,
                pivot_dir,
                f"{broker_sel} - CATE por {dim_col} x Semana",
                ylabel=dim_col,
                pivot_nobs=pivot_nobs,
            ),
            use_container_width=True,
        )

        # Detalle linea por dimension
        st.subheader(f"Detalle por {dim_col}")
        dim_det = st.selectbox(
            f"{dim_col}:",
            sorted(df_bk_raw[dim_col].unique()),
            key=f"{tab_key}_dim_det2",
        )
        df_det = df_bk_raw[df_bk_raw[dim_col] == dim_det].sort_values("SEMANA")
        if not df_det.empty:
            st.plotly_chart(
                make_line_chart(
                    df_det,
                    f"{broker_sel} | {dim_col}: {dim_det}",
                    broker_ref,
                ),
                use_container_width=True,
            )

    else:
        # ── Consolidado: CATE promedio broker x dimension ─────────
        st.subheader(f"CATE Promedio por Broker x {dim_col} (todas las semanas)")

        df_cons = df_display.dropna(subset=["CATE"])
        if df_cons.empty:
            st.warning(
                "No hay datos significativos con los filtros actuales. "
                "Reduce el soporte mínimo (N_OBS) o permite valores inciertos."
            )
            return

        # Usar datos con filtro de significancia aplicado
        pivot_global = (
            df_cons.groupby(["BROKER", dim_col])["CATE"]
            .mean()
            .unstack(dim_col)
        )
        if pivot_global.empty or pivot_global.shape[1] == 0:
            st.warning("No hay datos para mostrar en el consolidado.")
            return
        pivot_global.loc[broker_ref] = 0
        orden = pivot_global.apply(
            lambda r: r.dropna().mean() if r.notna().any() else -999, axis=1
        ).sort_values(ascending=False).index
        pivot_global = pivot_global.reindex(orden)

        st.plotly_chart(
            make_heatmap(
                pivot_global,
                None,
                f"CATE Promedio por Broker x {dim_col}",
                xlabel=dim_col,
            ),
            use_container_width=True,
        )

        # Ranking de brokers (solo sobre datos significativos)
        st.subheader("Ranking de Brokers (CATE promedio - solo significativos)")
        ranking = (
            df_display.dropna(subset=["CATE"])
            .groupby("BROKER")["CATE"]
            .mean()
            .sort_values(ascending=False)
        )
        if ranking.empty:
            st.info("No hay resultados significativos con los filtros actuales.")
        else:
            ranking_df = ranking.reset_index()
            ranking_df.columns = ["BROKER", "CATE_PROMEDIO"]

            fig_rank = go.Figure(
                go.Bar(
                    x=ranking_df["CATE_PROMEDIO"],
                    y=ranking_df["BROKER"],
                    orientation="h",
                    marker_color=[
                        COLOR_MEJOR if v > 0 else COLOR_PEOR
                        for v in ranking_df["CATE_PROMEDIO"]
                    ],
                    hovertemplate="<b>%{y}</b><br>CATE prom: %{x:.3f}<extra></extra>",
                )
            )
            fig_rank.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_rank.update_layout(
                title=f"Ranking Global - {dim_col}",
                xaxis_title="CATE promedio (USD/KG vs Forever Fresh)",
                height=max(400, len(ranking_df) * 30 + 100),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        # Tabla resumen
        with st.expander("Ver tabla de datos (solo significativos)"):
            df_tabla = df_display.dropna(subset=["CATE"])
            if not df_tabla.empty:
                st.dataframe(
                    df_tabla.style.format(
                        {"CATE": "{:.3f}", "CI_LO": "{:.3f}", "CI_HI": "{:.3f}"}
                    ),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("Sin datos significativos.")


if __name__ == "__main__":
    main()