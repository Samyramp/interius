
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Interius",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

def get_password():
    # En Streamlit Cloud se puede definir APP_PASSWORD en secrets.toml
    try:
        return st.secrets["APP_PASSWORD"]
    except Exception:
        # Para pruebas en Colab / local
        return "interius2025"


def login_screen():
    """Pantalla de login con logo grande y barra de contraseña naranja."""

    login_css = """
    <style>
    /* Ocultar sidebar mientras estamos en login */
    [data-testid="stSidebar"] {
        display: none;
    }

    .login-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        margin-top: 4rem;
    }
    .login-logo img {
        max-width: 420px;
    }

    /* Input de contraseña naranja con transparencia */
    input[type="password"] {
        background: rgba(255, 140, 0, 0.12);
        border: 1px solid rgba(255, 140, 0, 0.9);
        border-radius: 999px;
        padding: 0.75rem 1rem;
    }
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)

    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="login-logo">', unsafe_allow_html=True)
    st.image("/content/logo_interius.png", width=420)
    st.markdown("</div>", unsafe_allow_html=True)

    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        if pwd == get_password():
            st.session_state["logged_in"] = True
        else:
            st.error("Contraseña incorrecta")

    st.markdown("</div>", unsafe_allow_html=True)


# Estado de login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Si NO está logueado -> muestro login y corto la app
if not st.session_state["logged_in"]:
    login_screen()
    st.stop()

# ---------------------------------------------------------
# A PARTIR DE AQUÍ: APP NORMAL (YA LOGUEADO)
# ---------------------------------------------------------

# Vuelvo a asegurar que la sidebar se vea normal
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: flex !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Leer CSS base
with open("/content/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Logo normal del dashboard
st.image("/content/logo_interius.png", width=300)


# ---------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_data():
    df_meta = pd.read_csv("/content/df_meta_17_10.csv")
    df_google = pd.read_csv("/content/df_google_17_10.csv")


    # Parseo de fechas
    df_meta["fecha"] = pd.to_datetime(df_meta["fecha"], errors="coerce")
    df_google["day"] = pd.to_datetime(df_google["day"], errors="coerce")

    return df_meta, df_google

df_meta, df_google = load_data()

# GOOGLE
google_tipo_campana_all = sorted(df_google["categoria_campana"].dropna().unique().tolist())
google_tipo_anuncio_all = sorted(df_google["ad_type"].dropna().unique().tolist())
google_min_date = df_google["day"].min().date()
google_max_date = df_google["day"].max().date()

# META
meta_tipo_campana_all = sorted(df_meta["categoria_campana"].dropna().unique().tolist())
meta_tipo_anuncio_all = sorted(df_meta["nombre_del_conjunto_de_anuncios"].dropna().unique().tolist())
meta_min_date = df_meta["fecha"].min().date()
meta_max_date = df_meta["fecha"].max().date()


# ---------------------------------------------------------
# INICIALIZAR SESSION STATE PARA FILTROS
# ---------------------------------------------------------

if "g_tipo_campana" not in st.session_state:
    st.session_state["g_tipo_campana"] = ["Todos"]
if "g_tipo_anuncio" not in st.session_state:
    st.session_state["g_tipo_anuncio"] = ["Todos"]
if "g_fecha" not in st.session_state:
    st.session_state["g_fecha"] = (google_min_date, google_max_date)

if "m_tipo_campana" not in st.session_state:
    st.session_state["m_tipo_campana"] = ["Todos"]
if "m_tipo_anuncio" not in st.session_state:
    st.session_state["m_tipo_anuncio"] = ["Todos"]
if "m_fecha" not in st.session_state:
    st.session_state["m_fecha"] = (meta_min_date, meta_max_date)

# estado para el filtro de ad_type en el modelo CVR Google
if "g_model_ad_type" not in st.session_state:
    st.session_state["g_model_ad_type"] = ["Todos"]


def handle_multiselect_all(key: str):
    selected = st.session_state.get(key, [])
    if "Todos" in selected and len(selected) > 1:
        selected = [v for v in selected if v != "Todos"]
    if len(selected) == 0:
        selected = ["Todos"]
    st.session_state[key] = selected


# ---------------------------------------------------------
# FUNCIONES AUXILIARES DE KPIs
# ---------------------------------------------------------

def calcular_kpis_google(df):
    alcance = float(df["impr"].sum())
    clicks = float(df["clicks"].sum())
    costo = float(df["cost"].sum())
    conv = float(df["conversions"].sum())

    ctr = float(df["ctr"].mean()) if "ctr" in df.columns else 0.0
    cvr = float(df["cvr"].mean()) if "cvr" in df.columns else 0.0

    # CPC
    if "avg_cpc" in df.columns:
        cpc = float(df["avg_cpc"].mean())
    elif "cpc" in df.columns:
        cpc = float(df["cpc"].mean())
    else:
        cpc = float(costo / clicks) if clicks > 0 else 0.0

    # CPM
    if "avg_cpm" in df.columns:
        cpm = float(df["avg_cpm"].mean())
    elif "cpm" in df.columns:
        cpm = float(df["cpm"].mean())
    else:
        cpm = float(costo / alcance * 1000) if alcance > 0 else 0.0

    return {
        "alcance": alcance,
        "clicks": clicks,
        "costo": costo,
        "conversiones": conv,
        "ctr": ctr,
        "cvr": cvr,
        "cpc": cpc,
        "cpm": cpm,
    }


def calcular_kpis_meta(df):
    alcance = float(df["impresiones"].sum())
    clicks = float(df["clics_en_el_enlace"].sum())
    costo = float(df["importe_gastado_mxn"].sum())
    conv = float(df["resultados"].sum())

    # CTR
    if "ctr_porcentaje_de_clics_en_el_enlace" in df.columns:
        ctr = float(df["ctr_porcentaje_de_clics_en_el_enlace"].mean())
    elif "ctr" in df.columns:
        ctr = float(df["ctr"].mean())
    else:
        ctr = 0.0

    # CVR
    if "CVR" in df.columns:
        cvr = float(df["CVR"].mean())
    elif "cvr" in df.columns:
        cvr = float(df["cvr"].mean())
    else:
        cvr = 0.0

    # CPC
    if "cpc_costo_por_clic_en_el_enlace" in df.columns:
        cpc = float(df["cpc_costo_por_clic_en_el_enlace"].mean())
    elif "cpc" in df.columns:
        cpc = float(df["cpc"].mean())
    else:
        cpc = float(costo / clicks) if clicks > 0 else 0.0

    # CPM
    if "cpm_costo_por_mil_impresiones" in df.columns:
        cpm = float(df["cpm_costo_por_mil_impresiones"].mean())
    elif "cpm" in df.columns:
        cpm = float(df["cpm"].mean())
    else:
        cpm = float(costo / alcance * 1000) if alcance > 0 else 0.0

    return {
        "alcance": alcance,
        "clicks": clicks,
        "costo": costo,
        "conversiones": conv,
        "ctr": ctr,
        "cvr": cvr,
        "cpc": cpc,
        "cpm": cpm,
    }


# ---------------------------------------------------------
# CALLBACKS RESET FILTROS
# ---------------------------------------------------------

def reset_google_filters():
    st.session_state["g_tipo_campana"] = ["Todos"]
    st.session_state["g_tipo_anuncio"] = ["Todos"]
    st.session_state["g_fecha"] = (google_min_date, google_max_date)


def reset_meta_filters():
    st.session_state["m_tipo_campana"] = ["Todos"]
    st.session_state["m_tipo_anuncio"] = ["Todos"]
    st.session_state["m_fecha"] = (meta_min_date, meta_max_date)


# ---------------------------------------------------------
# SIDEBAR: SELECCIÓN DE PLATAFORMA + FILTROS
# ---------------------------------------------------------

st.sidebar.title("Configuración")

plataforma = st.sidebar.radio(
    "Plataforma",
    options=["Google", "Meta"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Nota:** Los filtros de esta barra solo afectan las pestañas "
    "**DF** y **Dashboard**. Los modelos utilizan siempre la base completa "
    "de datos (no están afectados por estos filtros)."
)
st.sidebar.markdown("---")

df_filtrado = None
kpis = None

# ================= GOOGLE =================
if plataforma == "Google":
    st.sidebar.subheader("Filtros Google")

    opciones_campana_g = ["Todos"] + google_tipo_campana_all
    opciones_anuncio_g = ["Todos"] + google_tipo_anuncio_all

    st.sidebar.multiselect(
        "Tipo de campaña",
        options=opciones_campana_g,
        key="g_tipo_campana",
        on_change=handle_multiselect_all,
        args=("g_tipo_campana",)
    )
    tipo_campana = st.session_state["g_tipo_campana"]

    st.sidebar.multiselect(
        "Tipo de anuncio",
        options=opciones_anuncio_g,
        key="g_tipo_anuncio",
        on_change=handle_multiselect_all,
        args=("g_tipo_anuncio",)
    )
    tipo_anuncio = st.session_state["g_tipo_anuncio"]

    rango_fecha = st.sidebar.slider(
        "Rango de fechas",
        min_value=google_min_date,
        max_value=google_max_date,
        key="g_fecha"
    )

    st.sidebar.button("Reset filtros", on_click=reset_google_filters)

    df_filtrado = df_google.copy()

    if tipo_campana and "Todos" not in tipo_campana:
        df_filtrado = df_filtrado[df_filtrado["categoria_campana"].isin(tipo_campana)]

    if tipo_anuncio and "Todos" not in tipo_anuncio:
        df_filtrado = df_filtrado[df_filtrado["ad_type"].isin(tipo_anuncio)]

    if isinstance(rango_fecha, (list, tuple)) and len(rango_fecha) == 2:
        fecha_desde, fecha_hasta = rango_fecha
    else:
        fecha_desde = fecha_hasta = rango_fecha

    fecha_desde = pd.to_datetime(fecha_desde)
    fecha_hasta = pd.to_datetime(fecha_hasta)

    df_filtrado = df_filtrado[
        (df_filtrado["day"] >= fecha_desde)
        & (df_filtrado["day"] <= fecha_hasta)
    ]

    kpis = calcular_kpis_google(df_filtrado) if not df_filtrado.empty else calcular_kpis_google(df_google.iloc[0:0])

# ================= META =================
else:
    st.sidebar.subheader("Filtros Meta")

    opciones_campana_m = ["Todos"] + meta_tipo_campana_all
    opciones_anuncio_m = ["Todos"] + meta_tipo_anuncio_all

    st.sidebar.multiselect(
        "Tipo de campaña",
        options=opciones_campana_m,
        key="m_tipo_campana",
        on_change=handle_multiselect_all,
        args=("m_tipo_campana",)
    )
    tipo_campana = st.session_state["m_tipo_campana"]

    st.sidebar.multiselect(
        "Tipo de anuncio",
        options=opciones_anuncio_m,
        key="m_tipo_anuncio",
        on_change=handle_multiselect_all,
        args=("m_tipo_anuncio",)
    )
    tipo_anuncio = st.session_state["m_tipo_anuncio"]

    rango_fecha = st.sidebar.slider(
        "Rango de fechas",
        min_value=meta_min_date,
        max_value=meta_max_date,
        key="m_fecha"
    )

    st.sidebar.button("Reset filtros", on_click=reset_meta_filters)

    df_filtrado = df_meta.copy()

    if tipo_campana and "Todos" not in tipo_campana:
        df_filtrado = df_filtrado[df_filtrado["categoria_campana"].isin(tipo_campana)]

    if tipo_anuncio and "Todos" not in tipo_anuncio:
        df_filtrado = df_filtrado[df_filtrado["nombre_del_conjunto_de_anuncios"].isin(tipo_anuncio)]

    if isinstance(rango_fecha, (list, tuple)) and len(rango_fecha) == 2:
        fecha_desde, fecha_hasta = rango_fecha
    else:
        fecha_desde = fecha_hasta = rango_fecha

    fecha_desde = pd.to_datetime(fecha_desde)
    fecha_hasta = pd.to_datetime(fecha_hasta)

    df_filtrado = df_filtrado[
        (df_filtrado["fecha"] >= fecha_desde)
        & (df_filtrado["fecha"] <= fecha_hasta)
    ]

    kpis = calcular_kpis_meta(df_filtrado) if not df_filtrado.empty else calcular_kpis_meta(df_meta.iloc[0:0])


# ---------------------------------------------------------
# TABS: DF, DASHBOARD, MODELOS
# ---------------------------------------------------------

tab_eda, tab_dashboard, tab_modelos = st.tabs(["DF", "Dashboard", "Modelos"])

# ---------------- TAB DF ----------------
with tab_eda:
    st.subheader(f"DF - {plataforma}")
    st.markdown("---")

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        num_entradas = len(df_filtrado)

        if plataforma == "Google":
            alcance_prom = df_filtrado["impr"].mean()
        else:
            alcance_prom = df_filtrado["impresiones"].mean()

        clicks_tot = kpis["clicks"]
        conv_tot = kpis["conversiones"]
        costo_tot = kpis["costo"]

        cvr_prom = kpis["cvr"]
        cpc_prom = kpis["cpc"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Número de entradas", f"{num_entradas:,}")
        c2.metric("Alcance promedio", f"{alcance_prom:,.0f}")
        c3.metric("CVR promedio", f"{cvr_prom:.2f}%")
        c4.metric("CPC promedio", f"${cpc_prom:.2f}")

        st.markdown("---")

        st.markdown("### Vista de datos filtrados")
        st.dataframe(df_filtrado)

        csv_data = df_filtrado.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar CSV filtrado",
            data=csv_data,
            file_name=f"interius_{plataforma.lower()}_filtrado.csv",
            mime="text/csv"
        )

# ---------------- TAB DASHBOARD ----------------
with tab_dashboard:
    st.subheader(f"KPIs - {plataforma}")
    st.markdown("---")

    if df_filtrado is None or df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alcance", f"{kpis['alcance']:,.0f}")
        c2.metric("Clicks", f"{kpis['clicks']:,.0f}")
        c3.metric("CTR", f"{kpis['ctr']:.2f}%")
        c4.metric("Costo", f"${kpis['costo']:,.2f}")

        st.markdown("---")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Conversiones", f"{kpis['conversiones']:,.0f}")
        c6.metric("CVR", f"{kpis['cvr']:,.2f}%")
        c7.metric("CPC", f"${kpis['cpc']:,.2f}")
        c8.metric("CPM", f"{kpis['cpm']:,.2f}")

        st.markdown("---")
        st.markdown("### Visualizaciones")

        g1, g2 = st.columns(2)

        # g1 Mejores campañas en base a CPC
        with g1:
            st.markdown("#### Mejores campañas en base al CPC")

            if plataforma == "Google":
                if "avg_cpc" in df_filtrado.columns:
                    data_cpc = (
                        df_filtrado
                        .groupby("campaign", as_index=False)
                        .agg(CPC=("avg_cpc", "mean"))
                        .dropna(subset=["CPC"])
                    )
                else:
                    data_cpc = (
                        df_filtrado
                        .groupby("campaign", as_index=False)
                        .agg(
                            costo=("cost", "sum"),
                            clicks=("clicks", "sum")
                        )
                    )
                    data_cpc["CPC"] = np.where(
                        data_cpc["clicks"] > 0,
                        data_cpc["costo"] / data_cpc["clicks"],
                        np.nan
                    )
                    data_cpc = data_cpc.dropna(subset=["CPC"])
                nombre_col = "campaign"
            else:
                if "cpc_costo_por_clic_en_el_enlace" in df_filtrado.columns:
                    data_cpc = (
                        df_filtrado
                        .groupby("nombre_de_la_campaaa", as_index=False)
                        .agg(CPC=("cpc_costo_por_clic_en_el_enlace", "mean"))
                        .dropna(subset=["CPC"])
                    )
                else:
                    data_cpc = (
                        df_filtrado
                        .groupby("nombre_de_la_campaaa", as_index=False)
                        .agg(
                            costo=("importe_gastado_mxn", "sum"),
                            clicks=("clics_en_el_enlace", "sum")
                        )
                    )
                    data_cpc["CPC"] = np.where(
                        data_cpc["clicks"] > 0,
                        data_cpc["costo"] / data_cpc["clicks"],
                        np.nan
                    )
                    data_cpc = data_cpc.dropna(subset=["CPC"])
                nombre_col = "nombre_de_la_campaaa"

            if not data_cpc.empty:

                if plataforma == "Meta":
                    max_camps = int(data_cpc.shape[0])
                    default_n = min(10, max_camps)
                    n_camps = st.slider(
                        "Número de campañas a mostrar",
                        min_value=1,
                        max_value=max_camps,
                        value=default_n,
                        step=1,
                        key="meta_n_cpc_campaigns"
                    )
                else:
                    n_camps = min(10, int(data_cpc.shape[0]))

                data_plot = (
                    data_cpc
                    .sort_values("CPC", ascending=True)
                    .head(n_camps)
                )

                fig1 = px.bar(
                    data_plot,
                    x="CPC",
                    y=nombre_col,
                    orientation="h",
                    text=data_plot["CPC"].round(2),
                )
                fig1.update_layout(
                    xaxis_title="CPC promedio",
                    yaxis_title="Campaña",
                    showlegend=False,
                    height=400,
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No hay datos suficientes para calcular CPC por campaña.")

        # g2 CPC/CPM por tipo de anuncio
        with g2:
            st.markdown("#### Mejores anuncios en base a CPC y CPM")

            if plataforma == "Google":
                if {"avg_cpc", "avg_cpm"}.issubset(df_filtrado.columns):
                    data_tipo = (
                        df_filtrado
                        .groupby("ad_type", as_index=False)
                        .agg(
                            CPC=("avg_cpc", "mean"),
                            CPM=("avg_cpm", "mean")
                        )
                        .dropna(subset=["CPC", "CPM"])
                    )
                    id_col = "ad_type"
                else:
                    data_tipo = pd.DataFrame()
                    id_col = None

                if id_col is not None and not data_tipo.empty:
                    data_melt = data_tipo.melt(
                        id_vars=id_col,
                        value_vars=["CPC", "CPM"],
                        var_name="Métrica",
                        value_name="Valor"
                    )
                    fig2 = px.bar(
                        data_melt,
                        x=id_col,
                        y="Valor",
                        color="Métrica",
                        barmode="group"
                    )
                    fig2.update_layout(
                        xaxis_title="Tipo de anuncio",
                        yaxis_title="Costo promedio",
                        height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para mostrar CPC y CPM por tipo de anuncio.")
            else:
                metric_choice = st.radio(
                    "Métrica",
                    options=["CPC", "CPM"],
                    horizontal=True,
                    key="meta_top_metric"
                )

                required_cols = {
                    "nombre_del_conjunto_de_anuncios",
                    "cpc_costo_por_clic_en_el_enlace",
                    "cpm_costo_por_mil_impresiones",
                }

                if required_cols.issubset(df_filtrado.columns):
                    data_tipo = (
                        df_filtrado
                        .groupby("nombre_del_conjunto_de_anuncios", as_index=False)
                        .agg(
                            CPC=("cpc_costo_por_clic_en_el_enlace", "mean"),
                            CPM=("cpm_costo_por_mil_impresiones", "mean"),
                        )
                        .dropna(subset=["CPC", "CPM"])
                    )

                    if data_tipo.empty:
                        st.info("No hay datos suficientes para calcular CPC/CPM por tipo de anuncio.")
                    else:
                        metric_col = metric_choice

                        max_cats = int(data_tipo.shape[0])
                        default_n = min(10, max_cats)
                        n_cats = st.slider(
                            "Número de tipos de anuncio a mostrar",
                            min_value=1,
                            max_value=max_cats,
                            value=default_n,
                            step=1,
                            key="meta_n_adtypes"
                        )

                        data_plot = (
                            data_tipo
                            .sort_values(metric_col, ascending=True)
                            .head(n_cats)
                        )

                        fig2 = px.bar(
                            data_plot,
                            x="nombre_del_conjunto_de_anuncios",
                            y=metric_col,
                            text=data_plot[metric_col].round(2)
                        )
                        fig2.update_layout(
                            xaxis_title="Tipo de anuncio",
                            yaxis_title=f"{metric_col} promedio",
                            height=420,
                        )
                        fig2.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No hay columnas suficientes para mostrar CPC/CPM por tipo de anuncio en Meta.")

        st.markdown("---")
        g3, g4 = st.columns(2)

        # g3 KPIs mes a mes
        with g3:
            st.markdown("#### KPIs mes a mes")

            if "mes" not in df_filtrado.columns:
                st.info("No existe la columna 'mes' en los datos filtrados.")
            else:
                if plataforma == "Google":
                    agg_specs = {}
                    if "avg_cpc" in df_filtrado.columns:
                        agg_specs["CPC"] = ("avg_cpc", "mean")
                    if "avg_cpm" in df_filtrado.columns:
                        agg_specs["CPM"] = ("avg_cpm", "mean")
                    if "ctr" in df_filtrado.columns:
                        agg_specs["CTR"] = ("ctr", "mean")
                    if "cvr" in df_filtrado.columns:
                        agg_specs["CVR"] = ("cvr", "mean")
                else:
                    agg_specs = {}
                    if "cpc_costo_por_clic_en_el_enlace" in df_filtrado.columns:
                        agg_specs["CPC"] = ("cpc_costo_por_clic_en_el_enlace", "mean")
                    if "cpm_costo_por_mil_impresiones" in df_filtrado.columns:
                        agg_specs["CPM"] = ("cpm_costo_por_mil_impresiones", "mean")
                    if "ctr_porcentaje_de_clics_en_el_enlace" in df_filtrado.columns:
                        agg_specs["CTR"] = ("ctr_porcentaje_de_clics_en_el_enlace", "mean")
                    if "CVR" in df_filtrado.columns:
                        agg_specs["CVR"] = ("CVR", "mean")

                if not agg_specs:
                    st.info("No hay columnas de KPIs suficientes para agrupar por mes.")
                else:
                    kpi_mes = (
                        df_filtrado
                        .groupby("mes")
                        .agg(**agg_specs)
                        .reset_index()
                    )
                    kpi_mes_long = kpi_mes.melt(
                        id_vars="mes",
                        var_name="KPI",
                        value_name="Valor"
                    )
                    fig3 = px.line(
                        kpi_mes_long,
                        x="mes",
                        y="Valor",
                        color="KPI",
                        markers=True
                    )
                    fig3.update_layout(
                        xaxis_title="Mes",
                        yaxis_title="Valor promedio",
                        height=380,
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        # g4 Costo por tipo de campaña
        with g4:
            st.markdown("#### Costo por tipo de campaña")

            if plataforma == "Google":
                cost_col = "cost_con_iva" if "cost_con_iva" in df_filtrado.columns else "cost"
            else:
                cost_col = "importe_gastado_mxn_con_iva" if "importe_gastado_mxn_con_iva" in df_filtrado.columns else "importe_gastado_mxn"

            if "categoria_campana" in df_filtrado.columns:
                costo_cat = (
                    df_filtrado
                    .groupby("categoria_campana", as_index=False)[cost_col]
                    .sum()
                )
                costo_cat = costo_cat[costo_cat[cost_col] > 0]

                if not costo_cat.empty:
                    fig4 = px.pie(
                        costo_cat,
                        names="categoria_campana",
                        values=cost_col,
                        hole=0.5
                    )
                    fig4.update_layout(height=380)
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No hay costos positivos para mostrar por tipo de campaña.")
            else:
                st.info("La columna 'categoria_campana' no existe en los datos filtrados.")


# ---------------- TAB MODELOS ----------------
with tab_modelos:
    st.subheader("Modelos")

    def _norm_camp(x):
        if not isinstance(x, str):
            return None
        s = x.strip().lower()
        if "awareness" in s or "alcance" in s:
            return "Awareness"
        if "conversion" in s or "conversión" in s or "conversions" in s:
            return "Conversiones"
        return x

    def _hinges(clicks, n=9):
        clicks = pd.Series(clicks).astype(float)
        qs = clicks.quantile(np.linspace(0.1, 0.9, n))
        data = {}
        for i, k in enumerate(qs):
            data[f"hinge_{i+1}"] = np.maximum(0.0, clicks - k)
        return pd.DataFrame(data, index=clicks.index)

    def _onehot(df, cols):
        df_copy = df.copy()
        for c in cols:
            if c in df_copy.columns:
                df_copy[c] = df_copy[c].astype(str)
        return pd.get_dummies(df_copy[cols], columns=cols, dtype=float)

    # Safety
    if (plataforma == "Meta" and (df_meta is None or df_meta.empty)) or (plataforma == "Google" and (df_google is None or df_google.empty)):
        st.warning("Datos de backend ausentes. Asegúrate de que df_meta y df_google estén cargados.")
        st.stop()

    left_col, right_col = st.columns(2)

    # --------- MODELO CVR (IZQUIERDA) ---------
    with left_col:
        if plataforma == "Meta":
            st.markdown("### Predicción CVR (Modelo Meta)")
            df_all_meta = df_meta.copy()
            if 'categoria_campana' in df_all_meta.columns:
                df_conv = df_all_meta[df_all_meta['categoria_campana'] == "Conversiones"].copy()
            else:
                df_conv = df_all_meta.copy()

            if df_conv is None or df_conv.empty:
                st.warning("No hay campañas Conversiones en la base completa de Meta.")
                st.stop()

            if 'CVR' not in df_conv.columns or df_conv['CVR'].dropna().empty:
                st.warning("La columna 'CVR' no existe o está vacía en la base completa de Meta.")
                st.stop()

            s = pd.to_numeric(df_conv['CVR'], errors='coerce').dropna()
            if s.empty:
                st.warning("La columna 'CVR' en Meta no contiene valores válidos.")
                st.stop()
            median = s.median()
            df_conv['CVR_scaled'] = s * 100.0 if median <= 1.0 else s

            invest = st.number_input("Inversión (MXN)", min_value=0.0, value=100.0, step=50.0, key="meta_only_invest_cvr")

            df_conv = df_conv.sort_values("importe_gastado_mxn")
            lowess = sm.nonparametric.lowess(df_conv['CVR_scaled'], df_conv['importe_gastado_mxn'], frac=0.4, return_sorted=True)
            x_smooth, y_smooth = lowess[:, 0], lowess[:, 1]

            if invest <= x_smooth.min():
                x0, x1 = x_smooth[0], x_smooth[1]; y0, y1 = y_smooth[0], y_smooth[1]
                slope = (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else 0
                predicted = float(y0 + slope * (invest - x0))
            elif invest >= x_smooth.max():
                x0, x1 = x_smooth[-2], x_smooth[-1]; y0, y1 = y_smooth[-2], y_smooth[-1]
                slope = (y1 - y0) / (x1 - x0) if (x1 - x0) != 0 else 0
                predicted = float(y1 + slope * (invest - x1))
            else:
                predicted = float(np.interp(invest, x_smooth, y_smooth))

            residuals = df_conv['CVR_scaled'] - np.interp(df_conv['importe_gastado_mxn'], x_smooth, y_smooth)
            low = predicted + residuals.quantile(0.10)
            high = predicted + residuals.quantile(0.90)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted CVR", f"{predicted:.2f}%")
            c2.metric("Lower Bound", f"{max(low,0):.2f}%")
            c3.metric("Upper Bound", f"{max(high,0):.2f}%")

            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(df_conv['importe_gastado_mxn'], df_conv['CVR_scaled'], s=30, edgecolor='w', alpha=0.8)
            ax.plot(x_smooth, y_smooth, label="LOWESS")
            ax.fill_between(x_smooth, y_smooth + residuals.quantile(0.10), y_smooth + residuals.quantile(0.90), alpha=0.2)
            ax.scatter([invest],[predicted], color='red', zorder=5, s=80)
            ax.set_xlabel("Inversión (MXN)")
            ax.set_ylabel("CVR (%)")
            st.pyplot(fig)

        else:
            st.markdown("### Predicción CVR (Árbol Google)")

            df_all_google = df_google.copy()

            if "ad_type" in df_all_google.columns:
                ad_types_all = sorted(df_all_google["ad_type"].dropna().unique().tolist())
                opciones_model_ad = ["Todos"] + ad_types_all

                st.multiselect(
                    "Filtrar tipo de anuncio (modelo CVR Google)",
                    options=opciones_model_ad,
                    key="g_model_ad_type",
                    on_change=handle_multiselect_all,
                    args=("g_model_ad_type",)
                )
                selected_ad_types = st.session_state["g_model_ad_type"]

                if "Todos" not in selected_ad_types:
                    df_all_google = df_all_google[df_all_google["ad_type"].isin(selected_ad_types)]

            if df_all_google.empty:
                st.warning("No hay datos disponibles para los tipos de anuncio seleccionados en el modelo de CVR Google.")
                st.stop()

            if 'categoria_campana' in df_all_google.columns:
                df_conv = df_all_google[df_all_google['categoria_campana'] == "Conversiones"].copy()
            else:
                df_conv = df_all_google.copy()

            if 'cvr' not in df_conv.columns and 'CVR' in df_conv.columns:
                df_conv['cvr'] = df_conv['CVR']
            if 'cvr' not in df_conv.columns:
                if 'conversions' in df_conv.columns and 'clicks' in df_conv.columns:
                    df_conv['cvr'] = np.where(df_conv['clicks']>0, df_conv['conversions'] / df_conv['clicks'], np.nan)
                elif 'resultados' in df_conv.columns and 'clics_en_el_enlace' in df_conv.columns:
                    df_conv['cvr'] = np.where(df_conv['clics_en_el_enlace']>0, df_conv['resultados'] / df_conv['clics_en_el_enlace'], np.nan)

            if 'outlier_flag' in df_conv.columns:
                df_conv = df_conv[~df_conv['outlier_flag']]

            if 'cvr' not in df_conv.columns or df_conv['cvr'].dropna().empty:
                st.warning("No se encontró columna de CVR ni forma de calcularla en la base completa de Google.")
                st.stop()

            cvr_series_raw = pd.to_numeric(df_conv['cvr'], errors='coerce').dropna()
            if cvr_series_raw.empty:
                st.warning("La columna 'cvr' en Google no contiene valores válidos.")
                st.stop()

            median_cvr = cvr_series_raw.median()
            df_conv = df_conv.loc[cvr_series_raw.index].copy()
            df_conv['cvr_scaled'] = cvr_series_raw * 100.0 if median_cvr <= 1.0 else cvr_series_raw

            cost_col = None
            for cand in ['cost_con_iva', 'cost', 'importe_gastado_mxn', 'cost_with_vat']:
                if cand in df_conv.columns:
                    cost_col = cand
                    break
            if cost_col is None:
                st.warning("No se encontró columna de costo adecuada en la base completa de Google.")
                st.stop()

            candidate_cats = ['campaign_bid_strategy_type', 'campaign_type', 'ad_type', 'dia_semana', 'cuartil']
            numeric_feats = [cost_col]
            available_cats = [c for c in candidate_cats if c in df_conv.columns]
            df_model = df_conv.dropna(subset=['cvr_scaled', cost_col]).copy()
            if df_model.shape[0] < 10:
                st.warning("No hay suficientes filas con CVR y features en la base completa de Google para entrenar el modelo.")
                st.stop()

            bins = [-float('inf'), 1.5, 3.0, 5.0, float('inf')]
            labels_cvr = ['Low', 'Average', 'High', 'Very High']
            y = pd.cut(df_model['cvr_scaled'], bins=bins, labels=labels_cvr, right=False)

            X = df_model[available_cats + numeric_feats].copy()
            if available_cats:
                X = pd.get_dummies(X, columns=available_cats, drop_first=True)
            X = X.select_dtypes(include=[np.number]).fillna(0)
            if X.shape[0] < 10 or X.shape[1] == 0:
                st.warning("No hay suficientes datos o features numéricas en la base completa de Google.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(X_train, y_train)

            default_invest = float(np.nanmedian(df_model[cost_col].values))
            if np.isnan(default_invest) or default_invest == 0:
                default_invest = 100.0
            invest = st.number_input("Inversión (MXN)", min_value=0.0, value=default_invest, step=50.0, key="google_only_invest_cvr")

            pred_row = {}
            for c in available_cats:
                mode = df_model[c].mode()
                pred_row[c] = mode.iloc[0] if not mode.empty else df_model[c].iloc[0]
            pred_row[cost_col] = invest

            feat_df = pd.DataFrame([pred_row])
            if available_cats:
                feat_df = pd.get_dummies(feat_df, columns=available_cats, drop_first=True)
            for col in X.columns:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            feat_df = feat_df[X.columns]

            probs = clf.predict_proba(feat_df)[0]
            class_order = clf.classes_

            grouped = df_model.groupby(pd.cut(df_model['cvr_scaled'], bins=bins, labels=labels_cvr, right=False))['cvr_scaled']
            means = grouped.mean().reindex(labels_cvr)
            p10s = grouped.quantile(0.10).reindex(labels_cvr)
            p90s = grouped.quantile(0.90).reindex(labels_cvr)

            predicted = float(np.nansum([p * (means.loc[cls] if not pd.isna(means.loc[cls]) else 0.0) for p, cls in zip(probs, class_order)]))
            low = float(np.nansum([p * (p10s.loc[cls] if not pd.isna(p10s.loc[cls]) else 0.0) for p, cls in zip(probs, class_order)]))
            high = float(np.nansum([p * (p90s.loc[cls] if not pd.isna(p90s.loc[cls]) else 0.0) for p, cls in zip(probs, class_order)]))

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted CVR", f"{predicted:.2f}%")
            c2.metric("Lower Bound", f"{low:.2f}%")
            c3.metric("Upper Bound", f"{high:.2f}%")

            acc = accuracy_score(y_test, clf.predict(X_test))
            pct = (np.sum(df_model[cost_col].values <= invest) / len(df_model[cost_col].values)) * 100 if len(df_model) > 0 else 0
            st.caption(f"Percentil inversión ~{pct:.0f} — Modelo accuracy (test): {acc:.2f}.")

            try:
                frac = 0.3 if len(df_model) > 30 else 0.6
                lowess2 = sm.nonparametric.lowess(df_model['cvr_scaled'].values, df_model[cost_col].values, frac=frac, return_sorted=True)
                xs2, ys2 = lowess2[:,0], lowess2[:,1]
                ys_interp = np.interp(df_model[cost_col].values, xs2, ys2)
                resid = df_model['cvr_scaled'].values - ys_interp
                resid_std = np.nanstd(resid)
                band_l = ys2 - 1.96*resid_std
                band_u = ys2 + 1.96*resid_std
            except Exception:
                xs2, ys2, band_l, band_u = df_model[cost_col].values, df_model['cvr_scaled'].values, df_model['cvr_scaled'].values*0, df_model['cvr_scaled'].values*0

            fig, ax = plt.subplots(figsize=(9,5))
            sns.scatterplot(x=df_model[cost_col].values, y=df_model['cvr_scaled'].values, ax=ax, s=30, edgecolor='w')
            ax.plot(xs2, ys2, linewidth=2)
            ax.fill_between(xs2, band_l, band_u, alpha=0.25)
            ax.scatter([invest], [predicted], color='red', s=80, zorder=5)
            ax.set_xlabel("Inversión (MXN)")
            ax.set_ylabel("CVR (%)")
            ax.set_title("Curva de respuesta — CVR esperado + banda")
            st.pyplot(fig)

    # --------- MODELOS AWARENESS / ALCANCE (DERECHA) ---------
    with right_col:
        if plataforma == "Google":
            st.markdown("### Predicción Awareness (Impresiones)")
            df_all_google2 = df_google.copy()
            df_all_google2["__camp__"] = df_all_google2.get("categoria_campana", pd.Series(index=df_all_google2.index)).map(_norm_camp)
            dsub = df_all_google2[df_all_google2["__camp__"]=="Awareness"].copy()
            if dsub.shape[0] < 10:
                st.warning("Awareness (Google): se requieren ≥10 filas en la base completa de Google para entrenar este modelo.")
            else:
                req = ["cost","clicks","mes","dia_semana","ad_type","categoria_campana"]
                missing = [c for c in req if c not in dsub.columns]
                if missing:
                    st.warning(f"Awareness (Google): faltan columnas: {missing}")
                else:
                    dsub["cost"] = pd.to_numeric(dsub["cost"], errors="coerce")
                    dsub["clicks"] = pd.to_numeric(dsub["clicks"], errors="coerce")
                    dsub = dsub.dropna(subset=["cost","clicks","mes","dia_semana","ad_type","categoria_campana"])
                    imp_candidates = ["impr", "impresiones", "impressions", "imps"]
                    imp_col = None
                    for c in imp_candidates:
                        if c in dsub.columns:
                            imp_col = c
                            break
                    if imp_col is None or dsub[imp_col].dropna().empty:
                        st.warning(f"No hay columna de impresiones válida en Awareness Google. Buscar columnas: {imp_candidates}")
                    else:
                        dsub[imp_col] = pd.to_numeric(dsub[imp_col], errors="coerce")
                        dsub = dsub.dropna(subset=[imp_col])
                        if dsub.shape[0] < 10:
                            st.warning("Awareness (Google): no hay suficientes filas con impresiones válidas para entrenar.")
                        else:
                            y_aw = dsub[imp_col].astype(float).values
                            clicks_aw = dsub['clicks'].astype(float)
                            c_center = clicks_aw - clicks_aw.mean()
                            X_poly = pd.DataFrame({f"c^{p}": c_center**p for p in range(1,7)}, index=dsub.index)
                            X_hinge = _hinges(clicks_aw)
                            D_aw = _onehot(dsub, ["mes","dia_semana","ad_type","categoria_campana"])
                            X_int = pd.DataFrame(index=dsub.index)
                            for col_d in D_aw.columns:
                                for col_n in list(X_poly.columns) + list(X_hinge.columns):
                                    X_int[f"{col_d}__x__{col_n}"] = D_aw[col_d] * (X_poly[col_n] if col_n in X_poly.columns else X_hinge[col_n])
                            X_aw = pd.concat([X_poly, X_hinge, D_aw, X_int], axis=1).fillna(0)
                            vt = VarianceThreshold(0.0)
                            Xr_aw = vt.fit_transform(X_aw)
                            cols_aw = X_aw.columns[vt.get_support(indices=True)]
                            Xr_aw = pd.DataFrame(Xr_aw, columns=cols_aw, index=X_aw.index)
                            scaler_aw = StandardScaler(with_mean=True, with_std=True)
                            Xs_aw = pd.DataFrame(scaler_aw.fit_transform(Xr_aw), columns=Xr_aw.columns, index=Xr_aw.index)
                            reg_aw = LinearRegression(fit_intercept=True)
                            reg_aw.fit(Xs_aw, y_aw)
                            yhat_aw = reg_aw.predict(Xs_aw)
                            residuals_reg = y_aw - yhat_aw
                            low_r = np.percentile(residuals_reg, 10)
                            high_r = np.percentile(residuals_reg, 90)
                            RMSE = np.sqrt(np.mean((y_aw - yhat_aw)**2))
                            R2 = r2_score(y_aw, yhat_aw)

                            default_invest = float(np.nanmedian(dsub['cost'].values)) if not np.isnan(np.nanmedian(dsub['cost'].values)) else 100.0
                            invest_aw = st.number_input("Inversión (MXN)", min_value=0.0, value=default_invest, step=50.0, key="google_aw_only_invest")

                            median_clicks = np.nanmedian(dsub['clicks'].values)
                            median_cost = np.nanmedian(dsub['cost'].values)
                            ratio = (median_clicks / median_cost) if (median_cost and not np.isnan(median_cost) and median_cost>0) else (np.nanmedian(dsub['clicks']) / 100.0)
                            est_clicks = invest_aw * ratio
                            c_est = est_clicks - clicks_aw.mean()
                            Xp_poly = pd.DataFrame({f"c^{p}": [c_est**p] for p in range(1,7)}, index=[0])
                            hinge_vals = clicks_aw.quantile(np.linspace(0.1,0.9,9)).values
                            Xp_hinge = pd.DataFrame({f"hinge_{i+1}": [max(0.0, est_clicks - k)] for i,k in enumerate(hinge_vals)}, index=[0])

                            D_modal = {}
                            for col in ["mes","dia_semana","ad_type","categoria_campana"]:
                                if col in dsub.columns:
                                    mode = dsub[col].mode()
                                    D_modal[col] = mode.iloc[0] if not mode.empty else dsub[col].iloc[0]
                                else:
                                    D_modal[col] = ""
                            Dp = _onehot(pd.DataFrame([D_modal]), ["mes","dia_semana","ad_type","categoria_campana"])
                            for col in D_aw.columns:
                                if col not in Dp.columns:
                                    Dp[col] = 0.0
                            Dp = Dp[D_aw.columns]

                            Xp_int = pd.DataFrame(index=[0])
                            for col_d in D_aw.columns:
                                for col_n in list(Xp_poly.columns) + list(Xp_hinge.columns):
                                    val_num = Xp_poly[col_n].iloc[0] if col_n in Xp_poly.columns else Xp_hinge[col_n].iloc[0]
                                    Xp_int[f"{col_d}__x__{col_n}"] = Dp[col_d].iloc[0] * val_num
                            Xp = pd.concat([Xp_poly, Xp_hinge, Dp, Xp_int], axis=1).fillna(0)
                            Xp_r = Xp.reindex(columns=cols_aw, fill_value=0)
                            Xp_s = pd.DataFrame(scaler_aw.transform(Xp_r), columns=cols_aw, index=[0])
                            pred_imp = reg_aw.predict(Xp_s)[0]
                            pred_low = pred_imp + low_r
                            pred_high = pred_imp + high_r

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Predicted Impressions", f"{pred_imp:,.0f}")
                            c2.metric("Lower Bound", f"{max(pred_low,0):,.0f}")
                            c3.metric("Upper Bound", f"{max(pred_high,0):,.0f}")
                            st.caption(f"Awareness regression metrics (Google) — RMSE: {RMSE:.3f} | R2: {R2:.3f}")

                            fig, ax = plt.subplots(figsize=(8,5))
                            ax.scatter(dsub['cost'], dsub[imp_col], alpha=0.6, s=30, edgecolor='w')
                            try:
                                lowess_aw = sm.nonparametric.lowess(dsub[imp_col].values, dsub['cost'].values, frac=0.4, return_sorted=True)
                                xs_aw, ys_aw = lowess_aw[:,0], lowess_aw[:,1]
                                ys_interp_aw = np.interp(dsub['cost'].values, xs_aw, ys_aw)
                                resid_aw = dsub[imp_col].values - ys_interp_aw
                                ax.plot(xs_aw, ys_aw, linewidth=2)
                                ax.fill_between(xs_aw, ys_aw + np.percentile(resid_aw,10), ys_aw + np.percentile(resid_aw,90), alpha=0.2)
                            except Exception:
                                pass
                            ax.scatter([invest_aw], [pred_imp], color='red', s=80, zorder=5)
                            ax.set_xlabel("Inversión (MXN)")
                            ax.set_ylabel("Impresiones")
                            ax.set_title("Awareness — Predicción Impr (Google)")
                            st.pyplot(fig)

        else:
            st.markdown("### Predicción Alcance (Awareness Meta)")
            df_all_meta2 = df_meta.copy()
            df_all_meta2["__camp__"] = df_all_meta2.get("categoria_campana", pd.Series(index=df_all_meta2.index)).map(lambda x: _norm_camp(x) if pd.notna(x) else None)
            dsub = df_all_meta2[df_all_meta2["__camp__"]=="Awareness"].copy()
            if dsub.shape[0] < 10:
                st.warning("Awareness (Meta): se requieren ≥10 filas en la base completa para entrenar este modelo.")
            else:
                dsub["impresiones"] = pd.to_numeric(dsub["impresiones"], errors="coerce")
                dsub["frecuencia"] = pd.to_numeric(dsub["frecuencia"], errors="coerce")
                dsub["importe_gastado_mxn"] = pd.to_numeric(dsub["importe_gastado_mxn"], errors="coerce")
                dsub = dsub.dropna(subset=["impresiones","frecuencia","importe_gastado_mxn","mes","dia_semana","cuartil"])
                dsub["alcance"] = dsub["impresiones"] / dsub["frecuencia"]
                mask_valid = (dsub["importe_gastado_mxn"] > 0) & (dsub["impresiones"] > 0) & (dsub["frecuencia"] > 0)
                dsub = dsub.loc[mask_valid].copy()
                if dsub.shape[0] < 10:
                    st.warning("Awareness (Meta): no hay suficientes filas válidas tras limpieza.")
                else:
                    def cap_at_p(df_local, col, p=0.99):
                        hi = df_local[col].quantile(p)
                        lo = df_local[col].quantile(1-p) if (df_local[col] >= 0).all() is False else df_local[col].min()
                        df_local[col + "_clipped"] = np.clip(df_local[col], lo, hi)
                        return hi
                    _ = cap_at_p(dsub, "importe_gastado_mxn", p=0.99)
                    _ = cap_at_p(dsub, "frecuencia", p=0.99)
                    dsub["log_importe_gastado_mxn"] = np.log1p(dsub["importe_gastado_mxn_clipped"])
                    dsub["frecuencia_clipped"] = dsub["frecuencia_clipped"] if "frecuencia_clipped" in dsub.columns else dsub["frecuencia"]
                    dsub["mes"] = dsub["mes"].astype(str)
                    dsub["dia_semana"] = dsub["dia_semana"].astype(str)
                    if dsub["cuartil"].dtype == "O":
                        mapa_cuartil = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
                        dsub["cuartil_ord"] = dsub["cuartil"].map(mapa_cuartil)
                    else:
                        dsub["cuartil_ord"] = pd.to_numeric(dsub["cuartil"], errors="coerce")
                    if dsub["cuartil_ord"].isna().any():
                        moda = dsub["cuartil_ord"].mode(dropna=True)
                        dsub["cuartil_ord"] = dsub["cuartil_ord"].fillna(moda.iloc[0] if len(moda) else 2)

                    dsub_sub = dsub.copy()
                    feature_cols_base = ["frecuencia_clipped", "log_importe_gastado_mxn"]
                    dummies_mes = pd.get_dummies(dsub_sub["mes"], prefix="mes", dtype=int)
                    dummies_dia = pd.get_dummies(dsub_sub["dia_semana"], prefix="dia", dtype=int)
                    X_df = pd.concat(
                        [
                            dsub_sub[feature_cols_base].reset_index(drop=True),
                            dummies_mes.reset_index(drop=True),
                            dummies_dia.reset_index(drop=True),
                            dsub_sub[["cuartil_ord"]].reset_index(drop=True)
                        ],
                        axis=1
                    )
                    y_aw = dsub_sub["alcance"].reset_index(drop=True)

                    if X_df.shape[0] < 10:
                        st.warning("Awareness (Meta): no hay suficientes filas para entrenar tras selección.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X_df, y_aw, test_size=0.2, random_state=42)

                        rf = RandomForestRegressor(
                            n_estimators=500,
                            max_depth=None,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=42
                        )
                        rf.fit(X_train.fillna(0), y_train.fillna(0))

                        pred_train = rf.predict(X_train.fillna(0))
                        pred_test = rf.predict(X_test.fillna(0))

                        def rmse_fn(y_true, y_pred):
                            return np.sqrt(mean_squared_error(y_true, y_pred))

                        def mape(y_true, y_pred, eps=1e-9):
                            y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
                            mask = np.abs(y_true) > eps
                            return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

                        metrics = {
                            "R2_train": r2_score(y_train, pred_train) if len(y_train)>0 else np.nan,
                            "R2_test":  r2_score(y_test, pred_test) if len(y_test)>0 else np.nan,
                            "RMSE_test": rmse_fn(y_test, pred_test) if len(y_test)>0 else np.nan,
                            "MAE_test":  mean_absolute_error(y_test, pred_test) if len(y_test)>0 else np.nan,
                            "MAPE_test": mape(y_test, pred_test) if len(y_test)>0 else np.nan
                        }

                        st.caption(f"Awareness RF metrics — R2_test: {metrics['R2_test']:.3f} | RMSE_test: {metrics['RMSE_test']:.1f}")

                        default_invest = float(np.nanmedian(dsub["importe_gastado_mxn"].values)) if not np.isnan(np.nanmedian(dsub["importe_gastado_mxn"].values)) else 100.0
                        invest_meta = st.number_input("Inversión (MXN)", min_value=0.0, value=default_invest, step=50.0, key="meta_aw_only_invest")

                        freq_med = float(np.nanmedian(dsub["frecuencia"].values)) if not np.isnan(np.nanmedian(dsub["frecuencia"].values)) else 1.0
                        sel_mes_modal = dsub_sub["mes"].mode().iloc[0] if not dsub_sub["mes"].mode().empty else dsub_sub["mes"].iloc[0]
                        sel_day_modal = dsub_sub["dia_semana"].mode().iloc[0] if not dsub_sub["dia_semana"].mode().empty else dsub_sub["dia_semana"].iloc[0]
                        sel_cuartil_modal = dsub_sub["cuartil_ord"].mode().iloc[0] if not dsub_sub["cuartil_ord"].mode().empty else dsub_sub["cuartil_ord"].iloc[0]

                        imp_clipped = np.clip(invest_meta, dsub["importe_gastado_mxn_clipped"].min(), dsub["importe_gastado_mxn_clipped"].quantile(0.99))
                        freq_clipped = np.clip(freq_med, dsub["frecuencia_clipped"].min(), dsub["frecuencia_clipped"].quantile(0.99))
                        log_imp = np.log1p(imp_clipped)

                        Xp = pd.DataFrame(
                            {
                                "frecuencia_clipped": [freq_clipped],
                                "log_importe_gastado_mxn": [log_imp],
                                "cuartil_ord": [sel_cuartil_modal]
                            }
                        )

                        dmes_cols = pd.get_dummies(dsub_sub["mes"], prefix="mes", dtype=int).columns
                        ddia_cols = pd.get_dummies(dsub_sub["dia_semana"], prefix="dia", dtype=int).columns
                        for m in dmes_cols:
                            Xp[m] = 1 if m == f"mes_{sel_mes_modal}" else 0
                        for d in ddia_cols:
                            Xp[d] = 1 if d == f"dia_{sel_day_modal}" else 0

                        for col in X_train.columns:
                            if col not in Xp.columns:
                                Xp[col] = 0
                        Xp = Xp[X_train.columns].fillna(0)

                        pred_alc = rf.predict(Xp)[0]
                        resid_train = y_train.values - rf.predict(X_train.fillna(0))
                        low_r = np.percentile(resid_train, 10)
                        high_r = np.percentile(resid_train, 90)
                        pred_low = pred_alc + low_r
                        pred_high = pred_alc + high_r

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicted Alcance", f"{pred_alc:,.0f}")
                        c2.metric("Lower Bound", f"{max(pred_low,0):,.0f}")
                        c3.metric("Upper Bound", f"{max(pred_high,0):,.0f}")

                        fig, ax = plt.subplots(figsize=(8,5))
                        ax.scatter(dsub_sub["importe_gastado_mxn"], dsub_sub["alcance"], alpha=0.6, s=30, edgecolor='w')
                        try:
                            lowess_aw = sm.nonparametric.lowess(dsub_sub["alcance"].values, dsub_sub["importe_gastado_mxn"].values, frac=0.4, return_sorted=True)
                            xs_aw, ys_aw = lowess_aw[:,0], lowess_aw[:,1]
                            ys_interp_aw = np.interp(dsub_sub["importe_gastado_mxn"].values, xs_aw, ys_aw)
                            resid_aw = dsub_sub["alcance"].values - ys_interp_aw
                            ax.plot(xs_aw, ys_aw, linewidth=2)
                            ax.fill_between(xs_aw, ys_aw + np.percentile(resid_aw,10), ys_aw + np.percentile(resid_aw,90), alpha=0.2)
                        except Exception:
                            pass
                        ax.scatter([invest_meta], [pred_alc], color='red', s=80, zorder=5)
                        ax.set_xlabel("Inversión (MXN)")
                        ax.set_ylabel("Alcance (impresiones / frecuencia)")
                        ax.set_title("Awareness (Meta) — Predicción Alcance vs Inversión")
                        st.pyplot(fig)
