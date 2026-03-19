"""
Climate Policy Simulator  Version simplificada
Pregunta clave: Cuantas vidas salvaria tu pais si adoptara
el modelo de energia limpia de Suecia?
Ejecutar: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Climate Policy Simulator", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.big-number { font-size: 2.8rem; font-weight: 800; line-height: 1.1; }
.label      { font-size: 0.82rem; color: #6b7280; text-transform: uppercase; letter-spacing:.05em; }
.card       { background: #f8fafc; border-left: 5px solid #10b981; border-radius: 8px; padding: 18px 22px; margin-bottom: 10px; }
.card-red   { border-left-color: #ef4444; }
.card-blue  { border-left-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

EXCLUIR = {"World","Asia","Europe","Africa","Americas","Oceania","North America","South America","European Union (27)","High-income countries","Low-income countries","Upper-middle-income countries","Lower-middle-income countries","East Asia & Pacific","Europe & Central Asia","Latin America & Caribbean","Middle East & North Africa","Sub-Saharan Africa","South Asia"}

SE_LOWCARBON = 98.5
SE_CO2PC     = 3.5
SE_LIFE      = 83.2
SE_RESP_RATE = 14.4
YEARS_FUTURE = list(range(2025, 2036))

@st.cache_data
def cargar_datos():
    return pd.read_csv("master_dataset.csv")

@st.cache_resource
def entrenar_modelo(_df):
    base = _df[~_df["Entity"].isin(EXCLUIR)].copy()
    dv = base.dropna(subset=["pct_lowcarbon","gdp_pc","life_exp"])
    sv = StandardScaler()
    Xv = sv.fit_transform(dv[["pct_lowcarbon","gdp_pc"]])
    mv = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    mv.fit(Xv, dv["life_exp"])
    dr = base.dropna(subset=["pct_lowcarbon","gdp_pc","resp_death_rate"])
    sr = StandardScaler()
    Xr = sr.fit_transform(dr[["pct_lowcarbon","gdp_pc"]])
    mr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    mr.fit(Xr, dr["resp_death_rate"])
    return mv, sv, mr, sr

def proyectar_pais(pais, df, mv, sv, mr, sr, anio_transicion):
    dp = df[df["Entity"] == pais].sort_values("Year")
    ult = dp.dropna(subset=["pct_lowcarbon"]).iloc[-1]
    lc0 = ult["pct_lowcarbon"]
    pop = dp.dropna(subset=["population"])["population"].iloc[-1] if dp["population"].notna().any() else 5e6
    dlc = dp.dropna(subset=["pct_lowcarbon"]).tail(15)
    if len(dlc) < 3:
        return None, None
    reg_lc = LinearRegression().fit(dlc[["Year"]], dlc["pct_lowcarbon"])
    dg = dp.dropna(subset=["gdp_pc"]).tail(15)
    if len(dg) >= 3:
        reg_gdp = LinearRegression().fit(dg[["Year"]], dg["gdp_pc"])
        gdp_fn = lambda yr: max(500, reg_gdp.predict([[yr]])[0])
    else:
        gdp_val = dp["gdp_pc"].dropna().iloc[-1] if dp["gdp_pc"].notna().any() else 15000
        gdp_fn = lambda yr: gdp_val
    filas = []
    for i, yr in enumerate(YEARS_FUTURE):
        gdp = gdp_fn(yr)
        lc_bau = float(np.clip(reg_lc.predict([[yr]])[0], 0, 100))
        progreso  = min(1.0, i / max(anio_transicion, 1))
        lc_suecia = lc0 + progreso * (SE_LOWCARBON - lc0)
        for tag, lc in [("BAU", lc_bau), ("Suecia", lc_suecia)]:
            X_v = sv.transform([[lc, gdp]])
            X_r = sr.transform([[lc, gdp]])
            vida  = float(np.clip(mv.predict(X_v)[0], 40, 95))
            resp  = float(np.clip(mr.predict(X_r)[0], 0, 200))
            filas.append({"Year": yr, "Escenario": tag, "pct_lowcarbon": round(lc, 1), "life_exp": round(vida, 2), "muertes_resp": round(resp * pop / 100000, 0), "resp_rate": round(resp, 2)})
    df_out = pd.DataFrame(filas)
    bau    = df_out[df_out["Escenario"] == "BAU"].set_index("Year")
    suecia = df_out[df_out["Escenario"] == "Suecia"].set_index("Year")
    return bau, suecia

# === SIDEBAR ===
df = cargar_datos()
mv, sv, mr, sr = entrenar_modelo(df)
paises = sorted([p for p in df["Entity"].unique() if p not in EXCLUIR and df[df["Entity"]==p]["pct_lowcarbon"].notna().sum()>=8 and df[df["Entity"]==p]["life_exp"].notna().sum()>=8])

st.sidebar.title("Climate Policy Simulator ")
st.sidebar.markdown("*Si tu país adopta el modelo energético de Suecia, esto es lo que ocurre.*")
st.sidebar.divider()
idx = paises.index("Spain") if "Spain" in paises else 0
pais_sel = st.sidebar.selectbox(" Selecciona tu país", paises, index=idx)
anio_trans = st.sidebar.slider("⚡ ¿En cuántos años completas la transición energética?", 1, 20, 10, 1)

# Mostrar el esfuerzo anual equivalente (se calcula después de cargar lc_actual)
# Lo mostramos aquí como placeholder; se actualiza tras cargar datos del país
_dp_pre = df[df["Entity"] == (paises[paises.index("Spain") if "Spain" in paises else 0] if "pais_sel" not in dir() else pais_sel)].sort_values("Year")
_lc_pre = _dp_pre.dropna(subset=["pct_lowcarbon"])["pct_lowcarbon"].iloc[-1] if not _dp_pre.empty else 50.0
_mejora_anual = (SE_LOWCARBON - _lc_pre) / anio_trans
if _mejora_anual > 0:
    st.sidebar.info(f"📈 Implica **+{_mejora_anual:.1f}% de energía limpia por año**\n\n"
                    f"(de {_lc_pre:.1f}% → {SE_LOWCARBON}% en {anio_trans} años)")
else:
    st.sidebar.success(f"✅ Este país ya supera el objetivo sueco")

st.sidebar.divider()
st.sidebar.markdown("**🇸🇪 Referencia Suecia:**")
st.sidebar.markdown(f"- Energia limpia: **{SE_LOWCARBON}%**")
st.sidebar.markdown(f"- CO2/hab: **{SE_CO2PC} t**")
st.sidebar.markdown(f"- Esperanza de vida: **{SE_LIFE} años**")
st.sidebar.markdown(f"- Mortalidad resp.: **{SE_RESP_RATE}/100k**")
st.sidebar.caption("Fuentes: Our World in Data, OMS, Banco Mundial  DADM 2025-2026")

# === DATOS DEL PAIS ===
dp = df[df["Entity"] == pais_sel].sort_values("Year")
ult_lc   = dp.dropna(subset=["pct_lowcarbon"]).iloc[-1]
ult_vida = dp.dropna(subset=["life_exp"]).iloc[-1]
ult_resp_row = dp.dropna(subset=["resp_death_rate"])
resp_actual  = ult_resp_row.iloc[-1]["resp_death_rate"] if not ult_resp_row.empty else 0.0
pop       = dp.dropna(subset=["population"])["population"].iloc[-1] if dp["population"].notna().any() else 5e6
lc_actual   = ult_lc["pct_lowcarbon"]
vida_actual = ult_vida["life_exp"]
lc_gap      = SE_LOWCARBON - lc_actual
bau, suecia = proyectar_pais(pais_sel, df, mv, sv, mr, sr, anio_trans)
if bau is None:
    st.error(f"No hay suficientes datos para {pais_sel}.")
    st.stop()

vida_bau_35  = bau.loc[2035, "life_exp"]
vida_se_35   = suecia.loc[2035, "life_exp"]
muertes_bau  = bau.loc[2026:, "muertes_resp"].sum()
muertes_se   = suecia.loc[2026:, "muertes_resp"].sum()
muertes_evit = muertes_bau - muertes_se
años_ganados = vida_se_35 - vida_bau_35

# === CABECERA ===
st.title(f" {pais_sel}  Si adoptamos el modelo de Suecia")
st.markdown(f"Transicion de **{lc_actual:.1f}%** a **{SE_LOWCARBON}%** de energia limpia en **{anio_trans} años**")
st.divider()

# === KPIs ===
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="card card-blue"><div class="label"> Brecha energética actual</div><div class="big-number" style="color:#3b82f6">{lc_gap:+.1f}%</div><div style="font-size:.9rem">{lc_actual:.1f}% actual  {SE_LOWCARBON}% (Suecia)</div></div>', unsafe_allow_html=True)
color_v = "#10b981" if años_ganados > 0 else "#ef4444"
c2.markdown(f'<div class="card"><div class="label"> Años de vida ganados (2035)</div><div class="big-number" style="color:{color_v}">{años_ganados:+.2f}</div><div style="font-size:.9rem">{vida_bau_35:.1f}  {vida_se_35:.1f} años</div></div>', unsafe_allow_html=True)
color_m = "#10b981" if muertes_evit > 0 else "#ef4444"
c3.markdown(f'<div class="card"><div class="label"> Muertes evitadas 2026-2035</div><div class="big-number" style="color:{color_m}">{int(muertes_evit):,}</div><div style="font-size:.9rem">por enf. respiratorias</div></div>', unsafe_allow_html=True)
brecha_vida = SE_LIFE - vida_actual
c4.markdown(f'<div class="card card-red"><div class="label"> Brecha vs Suecia hoy</div><div class="big-number" style="color:#ef4444">{brecha_vida:+.1f} años</div><div style="font-size:.9rem">{vida_actual:.1f} vs {SE_LIFE} (Suecia)</div></div>', unsafe_allow_html=True)

if lc_actual > 85:
    st.success(f" {pais_sel} ya tiene {lc_actual:.1f}% de energia limpia  cerca del nivel sueco. El reto es descarbonizar transporte e industria.")
elif lc_actual < 25:
    st.warning(f" {pais_sel} tiene solo {lc_actual:.1f}% de energia limpia. La transicion energetica es la palanca mas impactante para mejorar la salud publica.")

st.divider()

# === GRAFICOS ===
col_L, col_R = st.columns(2)

with col_L:
    hist_lc = dp.dropna(subset=["pct_lowcarbon"])[["Year","pct_lowcarbon"]]
    hist_lc = hist_lc[hist_lc["Year"] >= 1990]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hist_lc["Year"], y=hist_lc["pct_lowcarbon"], name="Historico", line=dict(color="#6b7280", width=2)))
    fig1.add_trace(go.Scatter(x=bau.index, y=bau["pct_lowcarbon"], name="BAU (sin cambio)", line=dict(color="#ef4444", width=2.5, dash="dash"), mode="lines+markers", marker=dict(size=5)))
    fig1.add_trace(go.Scatter(x=suecia.index, y=suecia["pct_lowcarbon"], name=f"Modelo Suecia ({anio_trans}a)", line=dict(color="#10b981", width=3), mode="lines+markers", marker=dict(size=5), fill="tonexty", fillcolor="rgba(16,185,129,0.08)"))
    fig1.add_hline(y=SE_LOWCARBON, line_dash="dot", line_color="#10b981", opacity=0.6, annotation_text=f"Objetivo Suecia ({SE_LOWCARBON}%)")
    fig1.add_vrect(x0=2024.5, x1=2035.5, fillcolor="#10b981", opacity=0.03)
    fig1.update_layout(title=" % Electricidad de fuentes limpias", xaxis_title="Ano", yaxis_title="% energia limpia", yaxis=dict(range=[0,105]), legend=dict(orientation="h", yanchor="bottom", y=-0.35), height=380, margin=dict(b=80), plot_bgcolor="white")
    st.plotly_chart(fig1, use_container_width=True)

with col_R:
    años_fut = list(range(2026, 2036))
    evitadas_acum = []
    total = 0
    for yr in años_fut:
        total += bau.loc[yr, "muertes_resp"] - suecia.loc[yr, "muertes_resp"]
        evitadas_acum.append(total)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=años_fut, y=[max(0,v) for v in evitadas_acum], marker_color="#10b981", name="Muertes evitadas", text=[f"{int(v):,}" for v in evitadas_acum], textposition="outside"))
    fig2.add_trace(go.Bar(x=años_fut, y=[abs(v) if v<0 else 0 for v in evitadas_acum], marker_color="#ef4444", name="Muertes adicionales"))
    fig2.update_layout(title=" Muertes respiratorias evitadas (acumuladas)", xaxis_title="Ano", yaxis_title="Muertes evitadas (acum.)", legend=dict(orientation="h", yanchor="bottom", y=-0.25), height=380, margin=dict(b=60), plot_bgcolor="white", barmode="relative", annotations=[dict(x=0.5, y=1.12, xref="paper", yref="paper", text=f"Total: <b>{int(muertes_evit):,} muertes evitadas en 10 años</b>", showarrow=False, font=dict(size=13, color="#10b981" if muertes_evit>=0 else "#ef4444"), bgcolor="rgba(240,253,244,0.9)", borderpad=6)])
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# === RESUMEN EJECUTIVO ===
st.subheader(" Resumen ejecutivo")
col_a, col_b = st.columns(2)
with col_a:
    st.info(f"""
**Situacion actual de {pais_sel}:**
- Energia limpia: **{lc_actual:.1f}%** (Suecia: {SE_LOWCARBON}%)
- Esperanza de vida: **{vida_actual:.1f} años** (Suecia: {SE_LIFE})
- Mortalidad resp.: **{resp_actual:.1f}/100k** (Suecia: {SE_RESP_RATE}/100k)
- Poblacion: **{pop/1e6:.1f} millones**
""")
with col_b:
    if muertes_evit > 0 and años_ganados > 0:
        msg = f"""
**Adoptando el modelo de Suecia en {anio_trans} años:**
-  Se evitarían **{int(muertes_evit):,} muertes** respiratorias (2026-2035)
-  La esperanza de vida mejoraría **{años_ganados:+.2f} años** para 2035
-  Objetivo energético: {lc_actual:.1f}% -> {SE_LOWCARBON}% electricidad limpia
"""
    elif lc_actual > 85:
        msg = f"""
**{pais_sel} ya tiene una matriz energetica limpia:**
-  {lc_actual:.1f}% de electricidad limpia  nivel nórdico alcanzado
-  El reto es descarbonizar transporte e industria
-  Potencial de mejora: {años_ganados:+.2f} años en esperanza de vida
"""
    else:
        msg = f"""
**Adoptando el modelo de Suecia en {anio_trans} años:**
- Impacto estimado: **{años_ganados:+.2f} años** de esperanza de vida
- Muertes resp. evitadas: **{int(muertes_evit):,}**
-  Clave: transicion energetica + politicas de calidad del aire
"""
    st.success(msg)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# BENCHMARKING vs SUECIA
# ════════════════════════════════════════════════════════════════════════════
st.subheader(f"Benchmarking: {pais_sel} vs 🇸🇪 Suecia")

# Obtener últimos datos disponibles del país
def ultimo_val(col):
    s = dp.dropna(subset=[col])
    return s.iloc[-1][col] if not s.empty else np.nan

co2pc_actual   = ultimo_val("co2_prod_pc")
gdp_actual     = ultimo_val("gdp_pc")
hdi_actual     = ultimo_val("hdi")
gini_actual    = ultimo_val("gini")

# Valores Suecia (último año disponible)
dse = df[df["Entity"] == "Sweden"].sort_values("Year")
def se_val(col):
    s = dse.dropna(subset=[col])
    return s.iloc[-1][col] if not s.empty else np.nan

se_co2pc  = se_val("co2_prod_pc")
se_gdp    = se_val("gdp_pc")
se_hdi    = se_val("hdi")
se_gini   = se_val("gini")
se_resp   = se_val("resp_death_rate")
se_vida   = se_val("life_exp")
se_lc     = se_val("pct_lowcarbon")

# ── Tabla comparativa ───────────────────────────────────────────────────────
indicadores = {
    "Energía limpia (%)":         (lc_actual,     se_lc,    True,  "{:.1f}%"),
    "CO₂ per cápita (t/hab)":    (co2pc_actual,  se_co2pc, False, "{:.2f} t"),
    " Esperanza de vida (años)":   (vida_actual,   se_vida,  True,  "{:.1f} años"),
    " Mortalidad resp. (/100k)":   (resp_actual,   se_resp,  False, "{:.1f}"),
    " PIB per cápita (USD)":       (gdp_actual,    se_gdp,   True,  "{:,.0f} $"),
    " IDH (0-1)":                  (hdi_actual,    se_hdi,   True,  "{:.3f}"),
    "⚖️ Índice Gini (desigualdad)":  (gini_actual,   se_gini,  False, "{:.1f}"),
}

col_tab, col_rad = st.columns([1, 1])

with col_tab:
    filas_tabla = []
    for nombre, (val_pais, val_se, mayor_mejor, fmt) in indicadores.items():
        if np.isnan(val_pais) or np.isnan(val_se):
            estado = "—"
            dif_str = "Sin datos"
        else:
            diferencia = val_pais - val_se
            pct = (diferencia / val_se * 100) if val_se != 0 else 0
            mejor = (diferencia > 0) == mayor_mejor
            estado = "✅ Mejor" if mejor else ("🟡 Igual" if abs(pct) < 2 else "❌ Peor")
            dif_str = f"{diferencia:+.2f} ({pct:+.1f}%)"
        filas_tabla.append({
            "Indicador": nombre,
            pais_sel: fmt.format(val_pais) if not np.isnan(val_pais) else "—",
            "🇸🇪 Suecia": fmt.format(val_se) if not np.isnan(val_se) else "—",
            "Diferencia": dif_str,
            "Estado": estado,
        })
    df_tabla = pd.DataFrame(filas_tabla)
    st.dataframe(df_tabla, hide_index=True, use_container_width=True)

# ── Radar comparativo ───────────────────────────────────────────────────────
with col_rad:
    CATS = ["Energía\nlimpia", "CO₂\nbajo", "Esp.\nvida", "Mortalidad\nbaja", "IDH", "PIB/hab"]

    def normalizar_global(val, col, inv=False):
        serie = df[~df["Entity"].isin(EXCLUIR)][col].dropna()
        if serie.empty or serie.max() == serie.min():
            return 0.5
        n = (val - serie.min()) / (serie.max() - serie.min())
        return float(np.clip(1 - n if inv else n, 0, 1))

    def perfil_radar(lc, co2, vida, resp, hdi, gdp):
        return [
            normalizar_global(lc,   "pct_lowcarbon"),
            normalizar_global(co2,  "co2_prod_pc",   inv=True),
            normalizar_global(vida, "life_exp"),
            normalizar_global(resp, "resp_death_rate", inv=True),
            normalizar_global(hdi,  "hdi"),
            normalizar_global(gdp,  "gdp_pc"),
        ]

    vals_pais = perfil_radar(lc_actual, co2pc_actual if not np.isnan(co2pc_actual) else 5,
                             vida_actual, resp_actual, hdi_actual if not np.isnan(hdi_actual) else 0.5,
                             gdp_actual if not np.isnan(gdp_actual) else 15000)
    vals_se   = perfil_radar(se_lc, se_co2pc, se_vida, se_resp,
                             se_hdi if not np.isnan(se_hdi) else 0.95,
                             se_gdp if not np.isnan(se_gdp) else 55000)

    fig_rad = go.Figure()
    for nombre, vals, color, fill in [
        (pais_sel, vals_pais, "#3b82f6", "rgba(59,130,246,0.15)"),
        ("🇸🇪 Suecia", vals_se, "#10b981", "rgba(16,185,129,0.15)"),
    ]:
        closed_v = vals + [vals[0]]
        closed_c = CATS + [CATS[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=closed_v, theta=closed_c,
            fill="toself", fillcolor=fill,
            name=nombre, line=dict(color=color, width=2.5)
        ))

    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
        title=f"Perfil comparativo (0 = peor mundial, 1 = mejor)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=400,
    )
    st.plotly_chart(fig_rad, use_container_width=True)

st.caption("Fuentes: Our World in Data, OMS, Banco Mundial | Modelo: GradientBoosting | DADM 2025-2026")

