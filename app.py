"""Dashboard Dash: OSHA ITA 2023, clasificación del resultado de un caso (4 clases).

Tres pestañas: (1) contexto del problema, (2) EDA, (3) modelos base.
Solo lee las tablas pequeñas de data/ (ver build_data.py); no entrena nada.
"""
import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

DATA = Path(__file__).parent / "data"
CLASSES = ["Fallecimiento (Death)", "Días de ausencia", "Traslado / restricción", "Otro caso registrable"]
COLORS = dict(zip(CLASSES, ["#b2182b", "#377eb8", "#4daf4a", "#984ea3"]))
MODEL_COLORS = {"LogisticRegression": "#377eb8", "DummyClassifier": "#8c8c8c"}
MODEL_NAMES = {"LogisticRegression": "Regresión logística", "DummyClassifier": "Dummy (clase más frecuente)"}


def read(name, **kw):
    return pd.read_csv(DATA / name, **kw)


info = json.loads((DATA / "info.json").read_text(encoding="utf-8"))
metricas = read("modelo_metricas.csv").pivot(index="metrica", columns="modelo", values="valor")
boot = read("modelo_bootstrap.csv")
por_clase = read("modelo_por_clase.csv")
auc_clase = read("modelo_auc_por_clase.csv")
roc = read("modelo_roc.csv")
pr = read("modelo_pr.csv")
curva = read("modelo_curva_aprendizaje.csv")
coef_path = DATA / "modelo_coeficientes.csv"

LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=50, b=50), font=dict(size=13))


def fmt_int(n):
    return f"{int(n):,}".replace(",", ".")


def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def note(text):
    return html.P(text, className="note")


def card(title, children):
    return html.Div([html.H3(title)] + children, className="card")


# ---------------------------------------------------------------- pestaña 1
def tab_contexto():
    train, test = info["train_rows"], info["test_rows"]
    timeline = go.Figure()
    bars = [
        ("Desarrollo (40 % de establecimientos)", "2023-01-01", "2023-09-23", "#377eb8"),
        ("Purga de 7 días", "2023-09-24", "2023-09-30", "#bdbdbd"),
        ("Prueba (el otro 60 % de establecimientos)", "2023-10-01", "2023-12-31", "#e66101"),
    ]
    for name, a, b, c in bars:
        timeline.add_trace(go.Bar(y=["Partición"], x=[((pd.Timestamp(b) - pd.Timestamp(a)).days + 1) * 86400000], base=[a],
                                  orientation="h", name=name, marker_color=c,
                                  hovertemplate=f"{name}<br>{a} a {b}<extra></extra>"))
    timeline.update_layout(**LAYOUT, barmode="overlay", height=190, xaxis=dict(type="date"),
                           title="Cómo se separó desarrollo y prueba (empresas distintas y tiempo posterior)",
                           legend=dict(orientation="h", y=-0.35))
    timeline.update_traces(width=0.5)

    cc = info["class_counts"]
    tabla = pd.DataFrame({
        "Resultado": CLASSES,
        "Desarrollo": [fmt_int(cc["train"][k]) for k in "1234"],
        "Prueba": [fmt_int(cc["test"][k]) for k in "1234"],
    })
    kpis = html.Div([
        html.Div([html.Div("890.934", className="kpi-n"), html.Div("casos reportados a OSHA en 2023", className="kpi-t")], className="kpi"),
        html.Div([html.Div("91.334", className="kpi-n"), html.Div("establecimientos", className="kpi-t")], className="kpi"),
        html.Div([html.Div("4", className="kpi-n"), html.Div("resultados posibles de un caso", className="kpi-t")], className="kpi"),
        html.Div([html.Div("0,03 %", className="kpi-n"), html.Div("terminan en fallecimiento", className="kpi-t")], className="kpi"),
    ], className="kpis")
    # Imagen opcional: si existe assets/header.jpg se muestra arriba (Dash sirve la carpeta assets/ solo)
    header_img = [html.Img(src=app.get_asset_url("header.jpg"), className="hero")] \
        if (Path(__file__).parent / "assets" / "header.jpg").exists() else []
    return html.Div([
        *header_img,
        kpis,
        card("La pregunta", [
            html.P("¿En qué medida las características del lugar de trabajo, la ocupación y el incidente permiten "
                   "clasificar el resultado de un caso registrado por OSHA en el último trimestre de 2023, en "
                   "establecimientos que no se usaron para construir el modelo?"),
            note("Es clasificación retrospectiva de casos ya reportados. No predice quién se lesionará ni "
                 "atribuye causas."),
            html.P("Este tablero tiene tres pestañas: el contexto del problema, el análisis exploratorio de los datos "
                   "(EDA) y los modelos base con sus métricas."),
        ]),
        card("Por qué importa", [
            html.P("Una lesión o enfermedad laboral puede terminar en un simple registro, en traslado o restricción "
                   "del trabajador, en días de ausencia o en un fallecimiento. Saber qué características del lugar "
                   "de trabajo y de la ocupación se asocian con cada resultado es relevante para la salud "
                   "ocupacional, siempre que se lea con los límites de un archivo de casos reportados "
                   "(más abajo)."),
        ]),
        card("Los datos", [
            html.P("OSHA ITA 2023 (Form 300/301), datos abiertos del Departamento de Trabajo de EE. UU. "
                   "El archivo original tiene 890.934 casos y 91.334 establecimientos. Unidad de observación: un "
                   "caso reportado."),
            html.Ul([
                html.Li("Objetivo: resultado final del caso, 4 clases nominales."),
                html.Li("8 predictores: estado, sector (NAICS), tipo de establecimiento, ocupación (SOC), "
                        "tipo de incidente, hora de inicio, hora del incidente e indicador de hora desconocida."),
                html.Li("Excluidos por fuga de información: días de ausencia, días de restricción y fecha de muerte."),
            ]),
            html.P(f"Desarrollo: {fmt_int(train)} casos en {fmt_int(info['train_groups'])} establecimientos. "
                   f"Prueba: {fmt_int(test)} casos en {fmt_int(info['test_groups'])} establecimientos. "
                   "Ningún establecimiento aparece en los dos grupos."),
            dcc.Graph(figure=timeline, config={"displayModeBar": False}),
            dash_table.DataTable(tabla.to_dict("records"), [{"name": c, "id": c} for c in tabla.columns],
                                 style_cell={"textAlign": "left", "padding": "6px"},
                                 style_header={"fontWeight": "600"}),
        ]),
        card("Por qué es un problema difícil", [
            html.P("Fallecimiento es el 0,03 % de los casos: solo 39 en la prueba, cada uno en un establecimiento "
                   "distinto. Un modelo que siempre dice «días de ausencia» ya acierta el 36 %, así que la "
                   "accuracy sola no sirve y se usan balanced accuracy y macro F1 score."),
        ]),
        card("Límites", [
            html.Ul([
                html.Li("El código de ocupación lo asigna OSHA después del reporte: no hay prueba de que existiera "
                        "al momento del incidente."),
                html.Li("Son casos reportados, no tasas de lesión: los conteos dependen del tamaño del "
                        "establecimiento y de sus prácticas de reporte."),
                html.Li("Las asociaciones que se muestran no son causales."),
            ])
        ]),
    ])


# ---------------------------------------------------------------- pestaña 2
def fig_resultado():
    d = read("eda_resultado.csv")
    f = go.Figure(go.Bar(y=d["resultado"], x=d["casos"], orientation="h",
                         marker_color=[COLORS[c] for c in d["resultado"]],
                         text=[f"{fmt_int(c)} ({fmt(p)} %)" for c, p in zip(d["casos"], d["porcentaje"])],
                         textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}<br>%{x:,} casos<extra></extra>"))
    f.update_layout(**LAYOUT, title="Resultado de los casos (desarrollo)", xaxis=dict(range=[0, d["casos"].max() * 1.5],
                    title="Casos"), yaxis=dict(autorange="reversed"), height=330)
    return f


def fig_faltantes(modo):
    if modo == "total":
        d = read("eda_faltantes.csv").sort_values("faltantes_pct")
        f = go.Figure(go.Bar(y=d["predictor"], x=d["faltantes_pct"], orientation="h", marker_color="#377eb8",
                             text=[f"{fmt(v, 1)} %" for v in d["faltantes_pct"]], textposition="outside",
                             cliponaxis=False))
        f.update_layout(**LAYOUT, title="Información faltante por predictor (desarrollo)",
                        xaxis=dict(title="% faltante", range=[0, 24]), height=330)
        return f
    d = read("eda_faltantes_por_resultado.csv").set_index("resultado")
    d = d[["soc_code", "time_started_work_hour", "time_of_incident_hour"]]
    f = go.Figure()
    for col in d.columns:
        f.add_trace(go.Bar(name=col, x=d.index, y=d[col]))
    f.update_layout(**LAYOUT, barmode="group", title="% faltante según el resultado del caso",
                    yaxis=dict(title="% faltante"), height=330, legend=dict(orientation="h", y=-0.3))
    return f


def fig_sector():
    d = read("eda_sector_resultado.csv").set_index("sector")
    z = d[CLASSES[1:]]
    f = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale="Blues", zmin=0, zmax=60,
                             text=[[fmt(v, 1) for v in row] for row in z.values], texttemplate="%{text}",
                             hovertemplate="%{y}<br>%{x}: %{z:.1f} %<extra></extra>",
                             colorbar=dict(title="% del sector")))
    f.update_layout(**LAYOUT, title="Resultado dentro de cada sector (%)", yaxis=dict(autorange="reversed"),
                    height=430)
    f.update_layout(margin=dict(l=190, r=20, t=50, b=50))
    return f


def fig_cramer():
    d = read("eda_cramer.csv").sort_values("cramer_v")
    f = go.Figure(go.Bar(y=d["predictor"], x=d["cramer_v"], orientation="h", marker_color="#377eb8",
                         text=[fmt(v) for v in d["cramer_v"]], textposition="outside", cliponaxis=False))
    f.update_layout(**LAYOUT, title="Asociación de cada predictor con el resultado (V de Cramér)",
                    xaxis=dict(title="V de Cramér (0 = ninguna, 1 = total)", range=[0, 0.3]), height=330)
    return f


def fig_tiempo(modo):
    if modo == "mes":
        d = read("eda_mes.csv")
        x = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep*"]
        titulo = "Casos por mes (desarrollo; *septiembre llega solo hasta el 23)"
    else:
        d = read("eda_dia_semana.csv")
        x = d["dia"]
        titulo = "Casos por día de la semana (desarrollo)"
    f = go.Figure()
    for c in CLASSES[1:]:
        f.add_trace(go.Bar(name=c, x=x, y=d[c], marker_color=COLORS[c]))
    f.update_layout(**LAYOUT, barmode="stack", title=titulo, yaxis=dict(title="Casos"), height=360,
                    legend=dict(orientation="h", y=-0.25))
    return f


def tab_eda():
    return html.Div([
        card("1. Resultado: una clase casi no existe", [
            dcc.Graph(figure=fig_resultado()),
            note("Fallecimiento son 69 casos de 269.516 (0,026 %). Por eso se miden balanced accuracy y macro F1 score, "
                 "no solo accuracy, y los resultados de esa clase se leen con cautela."),
        ]),
        card("2. Información faltante", [
            dcc.RadioItems(id="falt-modo", options=[{"label": "Por predictor", "value": "total"},
                                                    {"label": "Según el resultado", "value": "resultado"}],
                           value="total", inline=True),
            dcc.Graph(id="falt-fig", figure=fig_faltantes("total")),
            note("La ocupación (SOC) falta en el 18,9 % y las horas en el 12-13 %. La falta de SOC no es al azar: "
                 "cambia con el sector (V de Cramér 0,17) y se concentra por establecimiento. Por eso no se borran "
                 "filas ni se imputa la moda: el faltante pasa a ser una categoría «Unknown» dentro del Pipeline."),
        ]),
        card("3. Resultado según el sector", [
            dcc.Graph(figure=fig_sector()),
            note("Los sectores se parecen poco entre sí. En transporte el 45 % de los casos son días de ausencia; en "
                 "salud el 47 % son «otro caso registrable»; en arte y recreación el 45 % son traslado o restricción. "
                 "Esta diferencia es la señal que aprovecha el modelo. No son tasas de riesgo por sector."),
        ]),
        card("4. Qué predictores se asocian con el resultado", [
            dcc.Graph(figure=fig_cramer()),
            note("Sector (0,24) y ocupación (0,21) pesan más; tipo de incidente (0,15) y estado (0,12) menos. La "
                 "hora casi no se relaciona con el resultado (información mutua normalizada de 0,004 y 0,002, "
                 "medida aparte porque es numérica). V de Cramér mide fuerza de asociación, no causalidad."),
        ]),
        card("5. Los casos en el tiempo", [
            dcc.RadioItems(id="tiempo-modo", options=[{"label": "Por mes", "value": "mes"},
                                                      {"label": "Por día de la semana", "value": "dia"}],
                           value="mes", inline=True),
            dcc.Graph(id="tiempo-fig", figure=fig_tiempo("mes")),
            note("El volumen cae en fines de semana a la mitad (unos 22-25 mil casos frente a 41-46 mil). Esto "
                 "justifica separar el tiempo con una purga de 7 días entre entrenamiento y evaluación, para que "
                 "la dependencia entre días cercanos no infle el resultado."),
        ]),
        note("Hay más análisis (normalidad de las horas, mapa por estado, PCA, valores atípicos, auditoría de "
             "fuga) que no cabían aquí y están en el informe Jupyter Book."),
    ])


# ---------------------------------------------------------------- pestaña 3
def tabla_metricas():
    nombres = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1 score",
               "roc_auc_macro_ovr": "Macro AUC (one-vs-rest)"}
    filas = []
    for k, n in nombres.items():
        b = boot[(boot["metric"] == k)].set_index("model")
        lr, du = metricas.loc[k, "LogisticRegression"], metricas.loc[k, "DummyClassifier"]
        filas.append({
            "Metric": n,
            "Regresión logística": f"{fmt(lr)}  [{fmt(b.loc['LogisticRegression', 'lower'])} – "
                                   f"{fmt(b.loc['LogisticRegression', 'upper'])}]",
            "Dummy": f"{fmt(du)}  [{fmt(b.loc['DummyClassifier', 'lower'])} – {fmt(b.loc['DummyClassifier', 'upper'])}]",
        })
    return dash_table.DataTable(filas, [{"name": c, "id": c} for c in filas[0]],
                                style_cell={"textAlign": "left", "padding": "6px"},
                                style_header={"fontWeight": "600"})


def fig_confusion(modelo, norm):
    cm = pd.read_csv(DATA / f"modelo_confusion_{modelo}.csv", index_col=0)
    z = cm.div(cm.sum(axis=1), axis=0) * 100 if norm == "pct" else cm
    txt = [[(f"{v:.1f} %".replace(".", ",") if norm == "pct" else fmt_int(v)) for v in row] for row in z.values]
    f = go.Figure(go.Heatmap(z=z.values, x=[c.split(" (")[0] for c in cm.columns], y=[c.split(" (")[0] for c in cm.index],
                             colorscale="Blues", text=txt, texttemplate="%{text}", showscale=False,
                             zmin=0, zmax=100 if norm == "pct" else None))
    f.update_layout(**LAYOUT, title=f"Confusion matrix: {MODEL_NAMES[modelo]}", height=420,
                    xaxis=dict(title="Predicted"), yaxis=dict(title="Actual", autorange="reversed"))
    return f


def fig_curvas(tipo):
    f = go.Figure()
    if tipo == "roc":
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#8c8c8c"))
        for c in CLASSES:
            d = roc[roc["clase"] == c]
            a = auc_clase.set_index("clase").loc[c, "auc"]
            f.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{c.split(' (')[0]} (AUC {fmt(a)})",
                                   line=dict(color=COLORS[c])))
        f.update_layout(**LAYOUT, title="ROC curves, one class vs the rest (logistic regression)",
                        xaxis=dict(title="False positive rate"), yaxis=dict(title="Recall (true positive rate)"), height=430)
    else:
        for c in CLASSES[1:]:
            d = pr[pr["clase"] == c]
            a = auc_clase.set_index("clase").loc[c]
            f.add_trace(go.Scatter(x=d["recall"], y=d["precision"], mode="lines",
                                   name=f"{c} (AP {fmt(a['ap'])}; base {fmt(a['prevalencia'])})",
                                   line=dict(color=COLORS[c])))
        f.update_layout(**LAYOUT, title="Precision-Recall curves (the three frequent classes)",
                        xaxis=dict(title="Recall"), yaxis=dict(title="Precision", range=[0, 1]), height=430)
    f.update_layout(legend=dict(orientation="h", y=-0.3))
    return f


def fig_aprendizaje():
    g = curva.groupby("fraccion").agg(casos=("casos", "mean"), ent=("entrenamiento", "mean"),
                                      val=("validacion", "mean"), vstd=("validacion", "std")).reset_index()
    f = go.Figure()
    f.add_trace(go.Scatter(x=g["casos"], y=g["ent"], mode="lines+markers", name="Training",
                           line=dict(color="#377eb8")))
    f.add_trace(go.Scatter(x=g["casos"], y=g["val"], mode="lines+markers", name="Validation (forward in time)",
                           line=dict(color="#e66101")))
    f.update_layout(**LAYOUT, title="Learning curve: balanced accuracy (mean of 3 folds)",
                    xaxis=dict(title="Training cases (average)"), yaxis=dict(title="Balanced accuracy",
                    range=[0.3, 0.75]), height=400, legend=dict(orientation="h", y=-0.25))
    return f


def bloque_coeficientes():
    if not coef_path.exists():
        return []
    coef = pd.read_csv(coef_path, index_col=0)
    coef.columns = [int(c) for c in coef.columns]
    contrastes = [("type_of_incident", "3", "1", "Afección respiratoria frente a lesión"),
                  ("type_of_incident", "5", "1", "Pérdida auditiva frente a lesión"),
                  ("soc_code", "29-1141", "53-7062", "Enfermería profesional frente a movimiento manual de materiales"),
                  ("naics_code", "622110", "493110", "Hospitales generales frente a almacenamiento general")]
    filas = []
    for campo, a, b, etiqueta in contrastes:
        ca, cb = f"{campo}_{a}", f"{campo}_{b}"
        if ca not in coef.index or cb not in coef.index:
            continue
        for k in (1, 2, 3):
            delta = (coef.loc[ca, k] - coef.loc[ca, 4]) - (coef.loc[cb, k] - coef.loc[cb, 4])
            filas.append({"Comparación": etiqueta, "Resultado frente a «otro caso registrable»": CLASSES[k - 1],
                          "Razón de posibilidades relativa": fmt(math.exp(delta), 3 if math.exp(delta) < 0.1 else 2)})
    if not filas:
        return []
    return [card("Coeficientes: cómo cambian las posibilidades entre categorías", [
        dash_table.DataTable(filas, [{"name": c, "id": c} for c in filas[0]],
                             style_cell={"textAlign": "left", "padding": "6px", "whiteSpace": "normal"},
                             style_header={"fontWeight": "600"}),
        note("Son contrastes entre dos categorías del mismo predictor con lo demás fijo, no efectos causales. "
             "Los coeficientes de Fallecimiento son inestables por tener solo 69 casos de entrenamiento."),
    ])]


def tabla_modelos():
    filas = []
    for m in ("LogisticRegression", "DummyClassifier"):
        pc = por_clase[por_clase["modelo"] == m]
        filas.append({"Model": MODEL_NAMES[m], "Accuracy": fmt(metricas.loc["accuracy", m]),
                      "Precision (macro)": fmt(pc["precision"].mean()), "Recall (macro)": fmt(pc["sensibilidad"].mean()),
                      "F1 score (macro)": fmt(metricas.loc["macro_f1", m]),
                      "AUC (macro, OvR)": fmt(metricas.loc["roc_auc_macro_ovr", m])})
    return dash_table.DataTable(filas, [{"name": c, "id": c} for c in filas[0]],
                                style_cell={"textAlign": "left", "padding": "6px"},
                                style_header={"fontWeight": "600"})


def tab_modelos():
    cls = por_clase.copy()
    cls["Model"] = cls["modelo"].map(MODEL_NAMES)
    cls["Precision"] = cls["precision"].map(fmt)
    cls["Recall"] = cls["sensibilidad"].map(fmt)
    cls["F1 score"] = cls["f1"].map(fmt)
    cls["Support"] = cls["casos"].map(fmt_int)
    cls = cls.rename(columns={"clase": "Resultado"})[["Model", "Resultado", "Precision", "Recall", "F1 score", "Support"]]
    return html.Div([
        card("Metrics by model (prueba: octubre-diciembre, empresas nuevas)", [
            html.P("Cada fila es un modelo evaluado sobre los mismos 127.391 casos. Precision, recall y F1 son el "
                   "promedio simple de las 4 clases (macro). El recall macro es igual a la balanced accuracy."),
            tabla_modelos(),
        ]),
        card("Regresión logística frente a la línea base, con intervalos de confianza", [
            html.P("Valor [intervalo de confianza del 95 % por bootstrap de establecimientos]."),
            tabla_metricas(),
            note("La logística mejora a la línea base en todas las métricas y los intervalos no se solapan, pero el "
                 "nivel es moderado: acierta 52 % de los casos. Una accuracy tan lejos del 80-90 % no es señal de "
                 "fuga de información. El intervalo cubre solo la variación entre establecimientos de este "
                 "trimestre, no la del entrenamiento."),
        ]),
        card("Metrics by class", [
            dash_table.DataTable(cls.to_dict("records"), [{"name": c, "id": c} for c in cls.columns],
                                 style_cell={"textAlign": "left", "padding": "6px"},
                                 style_header={"fontWeight": "600"},
                                 style_data_conditional=[{"if": {"filter_query": '{Model} = "' + MODEL_NAMES["DummyClassifier"] + '"'},
                                                          "color": "#6b7280"}]),
            note("Fallecimiento: la logística identifica 1 de 39 casos (recall 2,6 %; intervalo de Wilson "
                 "0,5 %-13,2 %). Con tan pocos casos no se puede afirmar nada fiable sobre esa clase. El Dummy "
                 "predice siempre «días de ausencia», por eso su recall es 1 en esa clase y 0 en las demás."),
        ]),
        card("Confusion matrix", [
            html.Div([
                dcc.RadioItems(id="cm-modelo", options=[{"label": MODEL_NAMES[m], "value": m} for m in MODEL_NAMES],
                               value="LogisticRegression", inline=True),
                dcc.RadioItems(id="cm-norm", options=[{"label": "Conteos", "value": "n"},
                                                      {"label": "% de cada clase real", "value": "pct"}],
                               value="pct", inline=True),
            ], className="controls"),
            dcc.Graph(id="cm-fig", figure=fig_confusion("LogisticRegression", "pct")),
            note("Cada fila suma 100 % de los casos reales de esa clase. El Dummy manda todo a «días de ausencia». "
                 "La logística confunde sobre todo las tres clases frecuentes entre sí."),
        ]),
        card("ROC and Precision-Recall curves", [
            dcc.RadioItems(id="curva-tipo", options=[{"label": "ROC", "value": "roc"},
                                                     {"label": "Precision-Recall", "value": "pr"}],
                           value="roc", inline=True),
            dcc.Graph(id="curva-fig", figure=fig_curvas("roc")),
            note("Las tres clases frecuentes quedan entre 0,70 y 0,73 de AUC; Fallecimiento 0,61, con un intervalo "
                 "muy amplio por sus 39 casos. En la curva Precision-Recall la línea de base de cada clase es su "
                 "proporción de casos: la logística la supera, pero con margen modesto."),
        ]),
        card("Learning curve", [
            dcc.Graph(figure=fig_aprendizaje()),
            note("La validación casi no mejora con más datos (0,377 a 0,385) y la distancia con el entrenamiento "
                 "sigue siendo grande: no parece que más casos, con estos 8 predictores y un modelo lineal, "
                 "cambien el resultado. La brecha de entrenamiento probablemente se debe en buena parte a "
                 "Fallecimiento, que con muy pocos casos el modelo memoriza (inferencia, no comprobada)."),
        ]),
    ] + bloque_coeficientes())


# ---------------------------------------------------------------- app
app = Dash(__name__, title="OSHA 2023: resultado de los casos", suppress_callback_exceptions=True)
server = app.server  # gunicorn app:server

app.layout = html.Div([
    html.H1("OSHA ITA 2023: ¿qué resultado tiene un caso reportado?"),
    html.P("Tablero de visualización: contexto del problema, análisis exploratorio y modelos base.", className="sub"),
    dcc.Tabs(id="tabs", value="contexto", children=[
        dcc.Tab(label="1. Contexto del problema", value="contexto"),
        dcc.Tab(label="2. EDA", value="eda"),
        dcc.Tab(label="3. Modelos base", value="modelos"),
    ]),
    html.Div(id="tab-contenido"),
], className="page")


@app.callback(Output("tab-contenido", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"contexto": tab_contexto, "eda": tab_eda, "modelos": tab_modelos}[tab]()


@app.callback(Output("falt-fig", "figure"), Input("falt-modo", "value"))
def cb_faltantes(modo):
    return fig_faltantes(modo)


@app.callback(Output("tiempo-fig", "figure"), Input("tiempo-modo", "value"))
def cb_tiempo(modo):
    return fig_tiempo(modo)


@app.callback(Output("cm-fig", "figure"), Input("cm-modelo", "value"), Input("cm-norm", "value"))
def cb_cm(modelo, norm):
    return fig_confusion(modelo, norm)


@app.callback(Output("curva-fig", "figure"), Input("curva-tipo", "value"))
def cb_curva(tipo):
    return fig_curvas(tipo)


app.index_string = app.index_string.replace("</head>", """<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2933}
.page{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:1.6rem;margin:8px 0 0}.sub{color:#52606d;margin-top:4px}
.card{background:#fff;border:1px solid #e4e7eb;border-radius:8px;padding:12px 16px;margin:14px 0}
.card h3{margin:4px 0 8px;font-size:1.05rem}
.note{color:#323f4b;background:#f0f4f8;border-left:3px solid #377eb8;padding:8px 10px;border-radius:4px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}
.kpi{background:#0f3d63;color:#fff;border-radius:8px;padding:14px 12px;text-align:center}
.kpi-n{font-size:1.8rem;font-weight:700}.kpi-t{font-size:.85rem;opacity:.9}
.hero{width:100%;max-height:260px;object-fit:cover;border-radius:8px;margin-top:14px}
.controls{display:flex;gap:28px;flex-wrap:wrap}
label{margin-right:14px}
@media(max-width:600px){.page{padding:8px}.kpis{grid-template-columns:repeat(2,1fr)}}
</style></head>""")

if __name__ == "__main__":
    app.run(debug=False, port=8050)
