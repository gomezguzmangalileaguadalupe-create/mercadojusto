# -*- coding: utf-8 -*-
"""
Mercado Justo: Dashboard analítico para el mercado de ganado bovino.
Despliegue en Render mediante Docker.

Archivo de datos esperado:
    base_carne_14000_datos.xlsx

El Excel debe estar en la misma carpeta que este archivo mercado.py,
o se puede indicar otra ruta mediante la variable de entorno DATA_FILE.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
import networkx as nx

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# =====================================================================
# 1. LOCALIZAR Y VALIDAR EL ARCHIVO EXCEL
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent

path_xlsx = Path(
    os.environ.get(
        "DATA_FILE",
        str(BASE_DIR / "base_carne_14000_datos.xlsx")
    )
)

if not path_xlsx.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de datos: {path_xlsx}. "
        "Sube base_carne_14000_datos.xlsx a GitHub en la misma "
        "carpeta que mercado.py, o configura DATA_FILE."
    )

print(f"Leyendo archivo de datos: {path_xlsx}")

pdf_raw = pd.read_excel(path_xlsx)

columnas_requeridas = [
    "fecha",
    "estado",
    "producto",
    "precio_kg",
    "mercado",
    "tipo_alimentacion",
    "inflacion",
    "temperatura",
]

faltantes = [
    columna
    for columna in columnas_requeridas
    if columna not in pdf_raw.columns
]

if faltantes:
    raise ValueError(
        "Al archivo Excel le faltan estas columnas: "
        + ", ".join(faltantes)
    )

if pdf_raw.empty:
    raise ValueError("El archivo Excel no contiene registros.")


# =====================================================================
# 2. INICIALIZAR PYSPARK CON RECURSOS LIMITADOS
# =====================================================================

spark = (
    SparkSession.builder
    .appName("DashboardMercadoJusto")
    .master("local[2]")
    .config(
        "spark.driver.memory",
        os.environ.get("SPARK_DRIVER_MEMORY", "512m")
    )
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.default.parallelism", "2")
    .config("spark.driver.maxResultSize", "256m")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =====================================================================
# 3. PROCESAMIENTO DE DATOS CON PYSPARK
# =====================================================================

sdf = spark.createDataFrame(pdf_raw)

sdf = sdf.dropna(subset=columnas_requeridas)

sdf = sdf.withColumn(
    "fecha",
    F.to_date(F.col("fecha"))
)

sdf = (
    sdf
    .withColumn("anio", F.year(F.col("fecha")))
    .withColumn("mes", F.month(F.col("fecha")))
    .withColumn("dia_mes", F.dayofmonth(F.col("fecha")))
    .withColumn("dia_semana", F.dayofweek(F.col("fecha")) - 1)
)

df = sdf.toPandas()

df = df.dropna(subset=["fecha"]).copy()
df["fecha"] = pd.to_datetime(df["fecha"])

if df.empty:
    raise ValueError(
        "No quedaron registros válidos después de limpiar los datos."
    )

print(f"Datos cargados correctamente: {len(df)} registros.")


# =====================================================================
# 4. VARIABLES GLOBALES
# =====================================================================

productos = ["sirloin", "arrachera", "bistec", "costilla"]
estados = ["Puebla", "Jalisco", "Oaxaca", "Sonora", "CDMX"]
mercados = ["supermercado", "central_abasto", "digital"]
alimentacion = ["mixto", "engorda_corral", "libre_pastoreo"]

coordenadas_estados = {
    "CDMX": {
        "lat": 19.4326,
        "lon": -99.1332,
        "distancia_hub": 50
    },
    "Puebla": {
        "lat": 19.0414,
        "lon": -98.2063,
        "distancia_hub": 130
    },
    "Jalisco": {
        "lat": 20.6597,
        "lon": -103.3496,
        "distancia_hub": 550
    },
    "Oaxaca": {
        "lat": 17.0732,
        "lon": -96.7266,
        "distancia_hub": 460
    },
    "Sonora": {
        "lat": 29.0892,
        "lon": -110.9612,
        "distancia_hub": 1600
    },
}

conexiones_logisticas = [
    ("Sonora", "Jalisco", 1300),
    ("Sonora", "CDMX", 1600),
    ("Jalisco", "CDMX", 540),
    ("Jalisco", "Puebla", 670),
    ("CDMX", "Puebla", 130),
    ("CDMX", "Oaxaca", 460),
    ("Puebla", "Oaxaca", 340),
]


# =====================================================================
# 5. ENTRENAMIENTO DEL MODELO DE IA (RANDOM FOREST)
# =====================================================================

df_model = df.copy()

# Convertir variables numéricas, aceptando coma decimal.
for columna in ["inflacion", "temperatura", "precio_kg"]:
    df_model[columna] = pd.to_numeric(
        df_model[columna].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

df_model = df_model.dropna(
    subset=[
        "inflacion",
        "temperatura",
        "precio_kg",
        "anio",
        "mes",
        "dia_mes",
        "dia_semana",
        "estado",
        "producto",
        "mercado",
        "tipo_alimentacion",
    ]
).copy()

if df_model.empty:
    raise ValueError(
        "No hay registros suficientes para entrenar el modelo. "
        "Revisa las columnas numéricas y categóricas del Excel."
    )

encoders = {}
cat_cols = [
    "estado",
    "producto",
    "mercado",
    "tipo_alimentacion",
]

for col in cat_cols:
    encoders[col] = LabelEncoder()
    df_model[col] = encoders[col].fit_transform(
        df_model[col].astype(str)
    )

columnas_modelo = [
    "estado",
    "producto",
    "mercado",
    "tipo_alimentacion",
    "inflacion",
    "temperatura",
    "anio",
    "mes",
    "dia_mes",
    "dia_semana",
]

X = df_model[columnas_modelo]
y = df_model["precio_kg"]

modelo_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=2
)
modelo_rf.fit(X, y)


# =====================================================================
# 6. FUNCIONES REACTIVAS
# =====================================================================

def analizar_mercado_avanzado(producto, estado_filtro):
    df_filtrado = (
        df[
            (df["producto"].astype(str) == producto)
            & (df["estado"].astype(str) == estado_filtro)
        ]
        .sort_values("fecha")
        .copy()
    )

    if df_filtrado.empty:
        return (
            "<h2 style='color:red;text-align:center;'>"
            "Sin datos suficientes para esta selección</h2>",
            go.Figure()
        )

    precio_actual = float(df_filtrado.iloc[-1]["precio_kg"])
    promedio_global = float(df_filtrado["precio_kg"].mean())

    df_filtrado["tendencia_7dias"] = (
        df_filtrado["precio_kg"].rolling(window=7, min_periods=1).mean()
    )

    if precio_actual >= promedio_global:
        color_texto = "#1E4620"
        color_fondo = "#E8F5E9"
        color_borde = "#4CAF50"
    else:
        color_texto = "#B71C1C"
        color_fondo = "#FFEBEE"
        color_borde = "#F44336"

    kpi_html = f"""
    <div style='background-color:{color_fondo};padding:20px;
        border-radius:10px;text-align:center;border:1px solid {color_borde};
        border-left:8px solid {color_borde};margin-top:15px;'>
        <p style='margin:0;font-size:13px;font-weight:700;color:#555;
            text-transform:uppercase;letter-spacing:1px;'>
            Último precio registrado
        </p>
        <h1 style='margin:8px 0;font-size:2.8em;color:{color_texto};
            font-weight:800;'>
            ${precio_actual:.2f}
            <span style='font-size:16px;color:#666;font-weight:500;'>
                MXN/kg
            </span>
        </h1>
        <div style='background-color:rgba(255,255,255,0.6);
            padding:6px;border-radius:6px;display:inline-block;width:90%;'>
            <p style='margin:0;font-size:12px;font-weight:600;color:#444;'>
                Promedio histórico en {estado_filtro}:
                <span style='color:{color_texto};'>
                    ${promedio_global:.2f} MXN
                </span>
            </p>
        </div>
    </div>
    """

    fig_linea = go.Figure()

    fig_linea.add_trace(
        go.Scatter(
            x=df_filtrado["fecha"],
            y=df_filtrado["precio_kg"],
            mode="lines",
            name="Precio registrado",
            line=dict(color="#B0BEC5", width=1),
        )
    )

    fig_linea.add_trace(
        go.Scatter(
            x=df_filtrado["fecha"],
            y=df_filtrado["tendencia_7dias"],
            mode="lines",
            name="Tendencia (media móvil de 7 registros)",
            line=dict(color="#1E88E5", width=2.5),
        )
    )

    fig_linea.add_hline(
        y=promedio_global,
        line_dash="dash",
        line_color="#2E7D32",
        annotation_text="Promedio histórico",
    )

    fig_linea.update_layout(
        title=f"Comportamiento de {producto.capitalize()} en {estado_filtro}",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return kpi_html, fig_linea


def predecir_precio_rf_reactivo(
    producto,
    estado,
    mercado,
    alimentacion,
    inflacion,
    temperatura,
    fecha_pred,
):
    try:
        fecha_obj = pd.to_datetime(fecha_pred)
    except (TypeError, ValueError):
        fecha_obj = datetime.today()

    # Validar que las categorías elegidas existan en el entrenamiento.
    selecciones = {
        "producto": producto,
        "estado": estado,
        "mercado": mercado,
        "tipo_alimentacion": alimentacion,
    }

    for columna, valor in selecciones.items():
        if valor not in encoders[columna].classes_:
            return (
                "<div style='padding:18px;background:#FFEBEE;"
                "border-left:6px solid #D32F2F;border-radius:8px;'>"
                "<b>La categoría seleccionada no existe en los datos "
                "de entrenamiento.</b> Revisa los valores del Excel."
                "</div>"
            )

    prod_enc = encoders["producto"].transform([producto])[0]
    est_enc = encoders["estado"].transform([estado])[0]
    mer_enc = encoders["mercado"].transform([mercado])[0]
    al_enc = encoders["tipo_alimentacion"].transform([alimentacion])[0]

    input_data = pd.DataFrame(
        [[
            est_enc,
            prod_enc,
            mer_enc,
            al_enc,
            float(inflacion),
            float(temperatura),
            fecha_obj.year,
            fecha_obj.month,
            fecha_obj.day,
            fecha_obj.dayofweek,
        ]],
        columns=columnas_modelo,
    )

    prediccion = float(modelo_rf.predict(input_data)[0])

    kpi_pred = f"""
    <div style='background-color:#E3F2FD;padding:20px;
        border-radius:10px;text-align:center;border:1px solid #90CAF9;
        border-left:8px solid #1E88E5;margin-top:15px;'>
        <p style='margin:0;font-size:13px;font-weight:700;color:#0D47A1;
            text-transform:uppercase;'>
            Proyección de inteligencia artificial
        </p>
        <h1 style='margin:8px 0;font-size:2.8em;color:#1565C0;
            font-weight:800;'>
            ${prediccion:.2f}
            <span style='font-size:16px;color:#555;font-weight:500;'>
                MXN/kg
            </span>
        </h1>
        <p style='margin:0;font-size:11px;font-weight:600;color:#546E7A;'>
            Algoritmo: Random Forest Regressor
        </p>
        <p style='font-size:11px;color:#546E7A;'>
            Estimación del modelo; no representa una cotización garantizada.
        </p>
    </div>
    """

    return kpi_pred


# =====================================================================
# 7. OPTIMIZACIÓN FINANCIERA: GANANCIAS Y PÉRDIDAS
# =====================================================================

def optimizar_ganancias_reactivo(
    producto,
    costo_fijo,
    costo_variable,
    ganancia_deseada,
):
    df_filtrado = (
        df[df["producto"].astype(str) == producto]
        .sort_values("fecha")
    )

    precio_actual = (
        float(df_filtrado.iloc[-1]["precio_kg"])
        if not df_filtrado.empty
        else 250.0
    )

    if precio_actual <= costo_variable:
        return (
            """
            <div style='background-color:#FFEBEE;padding:20px;
                border-radius:8px;border-left:8px solid #D32F2F;'>
                <h4 style='color:#C62828;margin:0;'>
                    Alerta crítica: el costo variable es igual o superior
                    al precio de mercado. La operación no genera margen.
                </h4>
            </div>
            """,
            go.Figure(),
        )

    margen = precio_actual - costo_variable
    pto_equilibrio = costo_fijo / margen
    meta_kilos = (costo_fijo + ganancia_deseada) / margen

    resumen = f"""
    <div style='display:flex;justify-content:space-around;
        background-color:#F5F5F5;padding:15px;border-radius:10px;
        border:1px solid #E0E0E0;gap:10px;margin-top:15px;'>
        <div style='text-align:center;flex:1;'>
            <p style='margin:0;font-size:12px;font-weight:700;color:#666;'>
                Precio de mercado
            </p>
            <span style='font-size:1.8em;font-weight:800;color:#0288D1;'>
                ${precio_actual:.2f}
            </span>
        </div>
        <div style='text-align:center;flex:1;border-left:1px solid #DDD;
            border-right:1px solid #DDD;'>
            <p style='margin:0;font-size:12px;font-weight:700;color:#666;'>
                Punto de equilibrio
            </p>
            <span style='font-size:1.8em;font-weight:800;color:#F57C00;'>
                {pto_equilibrio:,.0f} kg
            </span>
        </div>
        <div style='text-align:center;flex:1;'>
            <p style='margin:0;font-size:12px;font-weight:700;color:#666;'>
                Meta de venta
            </p>
            <span style='font-size:1.8em;font-weight:800;color:#388E3C;'>
                {meta_kilos:,.0f} kg
            </span>
        </div>
    </div>
    """

    max_kg = max(
        int(meta_kilos * 1.6),
        int(pto_equilibrio * 1.2),
        100
    )

    kg_rango = np.linspace(0, max_kg, 100)
    ingresos = kg_rango * precio_actual
    costos = costo_fijo + (kg_rango * costo_variable)

    fig_opt = go.Figure()

    kg_perdida = np.linspace(0, pto_equilibrio, 50)
    ingresos_p = kg_perdida * precio_actual
    costos_p = costo_fijo + (kg_perdida * costo_variable)

    fig_opt.add_trace(
        go.Scatter(
            x=np.concatenate([kg_perdida, kg_perdida[::-1]]),
            y=np.concatenate([ingresos_p, costos_p[::-1]]),
            fill="toself",
            fillcolor="rgba(244,67,54,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Área de pérdida",
        )
    )

    kg_ganancia = np.linspace(pto_equilibrio, max_kg, 50)
    ingresos_g = kg_ganancia * precio_actual
    costos_g = costo_fijo + (kg_ganancia * costo_variable)

    fig_opt.add_trace(
        go.Scatter(
            x=np.concatenate([kg_ganancia, kg_ganancia[::-1]]),
            y=np.concatenate([ingresos_g, costos_g[::-1]]),
            fill="toself",
            fillcolor="rgba(76,175,80,0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Área de ganancia",
        )
    )

    fig_opt.add_trace(
        go.Scatter(
            x=kg_rango,
            y=costos,
            name="Costos totales",
            line=dict(color="#D32F2F", width=3),
        )
    )

    fig_opt.add_trace(
        go.Scatter(
            x=kg_rango,
            y=ingresos,
            name="Ingresos brutos",
            line=dict(color="#2E7D32", width=3),
        )
    )

    fig_opt.add_vline(
        x=pto_equilibrio,
        line_dash="dash",
        line_color="#E65100",
        line_width=2,
    )

    fig_opt.add_trace(
        go.Scatter(
            x=[pto_equilibrio],
            y=[pto_equilibrio * precio_actual],
            mode="markers+text",
            name="Punto de equilibrio",
            text=[f"Equilibrio ({pto_equilibrio:.0f} kg)"],
            textposition="top left",
            marker=dict(color="#E65100", size=10),
        )
    )

    fig_opt.update_layout(
        title=f"Análisis financiero: {producto.capitalize()}",
        xaxis_title="Volumen vendido / producido (kg)",
        yaxis_title="Flujo de capital (MXN)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return resumen, fig_opt


# =====================================================================
# 8. OPTIMIZACIÓN LOGÍSTICA CON KRUSKAL Y FOLIUM
# =====================================================================

def optimizar_logistica_mapa(
    producto,
    costo_base_produccion,
    costo_diesel_km,
):
    try:
        mejor_estado = ""
        mejor_margen = float("-inf")

        G = nx.Graph()

        for u, v, distancia in conexiones_logisticas:
            costo_viaje = distancia * costo_diesel_km
            G.add_edge(
                u,
                v,
                weight=costo_viaje,
                distancia=distancia,
            )

        arbol_minimo = nx.minimum_spanning_tree(
            G,
            algorithm="kruskal",
            weight="weight",
        )

        for estado, coords in coordenadas_estados.items():
            df_estado = (
                df[
                    (df["producto"].astype(str) == producto)
                    & (df["estado"].astype(str) == estado)
                ]
                .sort_values("fecha")
            )

            precio_mercado = (
                float(df_estado.iloc[-1]["precio_kg"])
                if not df_estado.empty
                else 230.0
            )

            costo_transporte = (
                coords["distancia_hub"] * costo_diesel_km
            ) / 500

            margen_neto = (
                precio_mercado
                - costo_base_produccion
                - costo_transporte
            )

            if margen_neto > mejor_margen:
                mejor_margen = margen_neto
                mejor_estado = estado

        mapa_folium = folium.Map(
            location=[23.6345, -102.5528],
            zoom_start=5,
            tiles="OpenStreetMap",
        )

        for estado, coords in coordenadas_estados.items():
            df_estado = df[
                (df["producto"].astype(str) == producto)
                & (df["estado"].astype(str) == estado)
            ].sort_values("fecha")

            precio_actual = (
                float(df_estado.iloc[-1]["precio_kg"])
                if not df_estado.empty
                else 230.0
            )

            info_popup = f"""
            <div style='font-family:Arial,sans-serif;min-width:160px;'>
                <b style='color:#1A237E;'>{estado}</b><br>
                <b>Precio registrado:</b> ${precio_actual:.2f} MXN/kg<br>
                <b>Distancia al centro:</b> {coords['distancia_hub']} km
            </div>
            """

            pin_color = "purple" if estado == mejor_estado else "red"

            folium.Marker(
                location=[coords["lat"], coords["lon"]],
                popup=folium.Popup(info_popup, max_width=250),
                tooltip=f"Haga clic para ver {estado}",
                icon=folium.Icon(
                    color=pin_color,
                    icon="truck",
                    prefix="fa",
                ),
            ).add_to(mapa_folium)

        costo_total_red = 0.0

        for u, v, datos in arbol_minimo.edges(data=True):
            costo_tramo = datos["weight"]
            distancia_tramo = datos["distancia"]
            costo_total_red += costo_tramo

            coord_u = [
                coordenadas_estados[u]["lat"],
                coordenadas_estados[u]["lon"],
            ]
            coord_v = [
                coordenadas_estados[v]["lat"],
                coordenadas_estados[v]["lon"],
            ]

            folium.PolyLine(
                locations=[coord_u, coord_v],
                color="#1B5E20",
                weight=5,
                opacity=0.85,
                tooltip=(
                    f"Enlace Kruskal: {u} ↔ {v} "
                    f"({distancia_tramo} km)"
                ),
            ).add_to(mapa_folium)

        mapa_html = mapa_folium._repr_html_()

        kpi_mapa = f"""
        <div style='background-color:#F3E5F5;padding:20px;
            border-radius:10px;text-align:center;border:1px solid #CE93D8;
            border-left:8px solid #8E24AA;margin-top:15px;'>
            <p style='margin:0;font-size:13px;font-weight:700;color:#4A148C;
                text-transform:uppercase;'>
                Estrategia del algoritmo Kruskal
            </p>
            <h1 style='margin:8px 0;font-size:2.2em;color:#6A1B9A;
                font-weight:800;'>
                {mejor_estado}
            </h1>
            <p style='margin:0;font-size:12px;font-weight:600;color:#2A085C;'>
                Margen neto comercial estimado:
                ${mejor_margen:.2f} MXN/kg
            </p>
            <div style='margin-top:10px;font-size:11px;
                background-color:rgba(255,255,255,0.5);padding:4px;
                border-radius:4px;'>
                <b>Costo total estimado de la red:</b>
                ${costo_total_red:,.2f} MXN
            </div>
        </div>
        """

        return kpi_mapa, mapa_html

    except Exception as e:
        return (
            "<h3 style='color:red;text-align:center;'>"
            f"Error en el mapeo logístico: {str(e)}"
            "</h3>",
            "",
        )


# =====================================================================
# 9. CONSTRUCCIÓN DE LA INTERFAZ GRÁFICA CON GRADIO
# =====================================================================

with gr.Blocks(theme=gr.themes.Soft()) as interfaz:

    gr.Markdown(
        """
        <div style='text-align:center;padding:25px;
            background:linear-gradient(90deg,#1A237E 0%,#0D47A1 100%);
            border-radius:12px;margin-bottom:25px;'>
            <h1 style='color:#fff;font-weight:800;margin:0;font-size:2.8em;'>
                Dashboard Mercado Justo
            </h1>
            <p style='font-size:18px;color:#E8EAF6;margin-top:8px;
                font-weight:500;'>
                Plataforma Analítica Integral para Productores de Ganado
                Bovino (Spark Engine)
            </p>
        </div>
        """
    )

    # -----------------------------------------------------------------
    # FASE 1: MERCADO HISTÓRICO
    # -----------------------------------------------------------------

    with gr.Tab("Fase 1: Mercado Histórico"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<h4 style='color:#1A237E;font-weight:bold;'>"
                    "Filtros de búsqueda</h4>"
                )

                prod_p1 = gr.Dropdown(
                    choices=productos,
                    value="sirloin",
                    label="Corte bovino",
                )

                edo_p1 = gr.Dropdown(
                    choices=estados,
                    value="CDMX",
                    label="Entidad",
                )

                kpi_display = gr.HTML()

            with gr.Column(scale=3):
                graf_historica = gr.Plot()

        inputs_p1 = [prod_p1, edo_p1]

        prod_p1.change(
            analizar_mercado_avanzado,
            inputs=inputs_p1,
            outputs=[kpi_display, graf_historica],
        )

        edo_p1.change(
            analizar_mercado_avanzado,
            inputs=inputs_p1,
            outputs=[kpi_display, graf_historica],
        )

        interfaz.load(
            analizar_mercado_avanzado,
            inputs=inputs_p1,
            outputs=[kpi_display, graf_historica],
        )

    # -----------------------------------------------------------------
    # FASE 2: INTELIGENCIA ARTIFICIAL Y PREDICCIÓN
    # -----------------------------------------------------------------

    with gr.Tab("Fase 2: IA y Predicción"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<h4 style='color:#1A237E;font-weight:bold;'>"
                    "Variables del modelo</h4>"
                )

                prod_p2 = gr.Dropdown(
                    choices=productos,
                    value="sirloin",
                    label="Corte",
                )

                edo_p2 = gr.Dropdown(
                    choices=estados,
                    value="CDMX",
                    label="Estado",
                )

                mer_p2 = gr.Dropdown(
                    choices=mercados,
                    value="central_abasto",
                    label="Canal de venta",
                )

                al_p2 = gr.Dropdown(
                    choices=alimentacion,
                    value="mixto",
                    label="Alimentación",
                )

                fecha_p2 = gr.Textbox(
                    value=datetime.today().strftime("%Y-%m-%d"),
                    label="Fecha a estimar",
                )

                gr.Markdown(
                    "<h4 style='color:#1A237E;font-weight:bold;'>"
                    "Simulador de entorno</h4>"
                )

                inf_p2 = gr.Slider(
                    minimum=3.0,
                    maximum=10.0,
                    value=6.0,
                    step=0.1,
                    label="Inflación (%)",
                )

                temp_p2 = gr.Slider(
                    minimum=10.0,
                    maximum=40.0,
                    value=24.0,
                    step=0.5,
                    label="Temperatura (°C)",
                )

            with gr.Column(scale=2):
                resultado_pred = gr.HTML()

        inputs_p2 = [
            prod_p2,
            edo_p2,
            mer_p2,
            al_p2,
            inf_p2,
            temp_p2,
            fecha_p2,
        ]

        for control in inputs_p2:
            control.change(
                predecir_precio_rf_reactivo,
                inputs=inputs_p2,
                outputs=[resultado_pred],
            )

        interfaz.load(
            predecir_precio_rf_reactivo,
            inputs=inputs_p2,
            outputs=[resultado_pred],
        )

    # -----------------------------------------------------------------
    # FASE 3: FINANZAS
    # -----------------------------------------------------------------

    with gr.Tab("Fase 3: Finanzas"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<h4 style='color:#1A237E;font-weight:bold;'>"
                    "Costos y rentabilidad</h4>"
                )

                prod_p3 = gr.Dropdown(
                    choices=productos,
                    value="sirloin",
                    label="Corte",
                )

                c_fijo = gr.Slider(
                    minimum=5000,
                    maximum=50000,
                    value=15000,
                    step=1000,
                    label="Costos fijos mensuales (MXN)",
                )

                c_variable = gr.Slider(
                    minimum=100,
                    maximum=250,
                    value=180,
                    step=5,
                    label="Costo variable por kg (MXN)",
                )

                ganancia_meta = gr.Slider(
                    minimum=10000,
                    maximum=100000,
                    value=30000,
                    step=5000,
                    label="Meta de ganancia (MXN)",
                )

            with gr.Column(scale=2):
                resumen_opt = gr.HTML()
                graf_opt = gr.Plot()

        inputs_p3 = [
            prod_p3,
            c_fijo,
            c_variable,
            ganancia_meta,
        ]

        for control in inputs_p3:
            control.change(
                optimizar_ganancias_reactivo,
                inputs=inputs_p3,
                outputs=[resumen_opt, graf_opt],
            )

        interfaz.load(
            optimizar_ganancias_reactivo,
            inputs=inputs_p3,
            outputs=[resumen_opt, graf_opt],
        )

    # -----------------------------------------------------------------
    # FASE 4: OPTIMIZACIÓN LOGÍSTICA
    # -----------------------------------------------------------------

    with gr.Tab("Fase 4: Optimización Logística"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<h4 style='color:#1A237E;font-weight:bold;'>"
                    "Parámetros operativos</h4>"
                )

                prod_p4 = gr.Dropdown(
                    choices=productos,
                    value="sirloin",
                    label="Producto a distribuir",
                )

                costo_base = gr.Slider(
                    minimum=100,
                    maximum=220,
                    value=150,
                    step=5,
                    label="Costo base de producción (MXN/kg)",
                )

                costo_diesel = gr.Slider(
                    minimum=5,
                    maximum=25,
                    value=12,
                    step=1,
                    label="Costo logístico diésel (MXN por km)",
                )

                kpi_mapa_display = gr.HTML()

            with gr.Column(scale=2):
                graf_mapa = gr.HTML()

        inputs_p4 = [
            prod_p4,
            costo_base,
            costo_diesel,
        ]

        for control in inputs_p4:
            control.change(
                optimizar_logistica_mapa,
                inputs=inputs_p4,
                outputs=[kpi_mapa_display, graf_mapa],
            )

        interfaz.load(
            optimizar_logistica_mapa,
            inputs=inputs_p4,
            outputs=[kpi_mapa_display, graf_mapa],
        )


# =====================================================================
# 10. INICIO DEL SERVIDOR PARA RENDER
# =====================================================================

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "10000"))

    print(f"Iniciando Mercado Justo en el puerto {puerto}...")

    interfaz.launch(
        server_name="0.0.0.0",
        server_port=puerto,
        show_error=True,
    )
