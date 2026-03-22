"""
Simulador de Política Climàtica  Model Suècia
Pregunta clau: Quantes vides salvaria el teu país si adoptés
el model d'energia neta de Suècia?
Executar: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Simulador Política Climàtica",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CSS professional gov style 
st.markdown("""
<style>
.hero {
  background: linear-gradient(135deg, #1e3a5f 0%, #0d6e6e 55%, #10b981 100%);
  border-radius: 14px; padding: 30px 36px; margin-bottom: 22px; color: white;
}
.hero-title  { font-size: 2.5rem; font-weight: 900; margin: 0; letter-spacing: -0.02em; }
.hero-sub    { font-size: 1.05rem; opacity: .88; margin-top: 6px; }
.hero-badge  {
  display: inline-block; background: rgba(255,255,255,0.18);
  border-radius: 20px; padding: 5px 16px; font-size: .84rem; margin-top: 12px;
}
.kpi {
  background: white; border-radius: 12px; padding: 20px 22px;
  box-shadow: 0 2px 14px rgba(0,0,0,0.08); border-top: 4px solid #10b981;
  min-height: 108px;
}
.kpi.red   { border-top-color: #ef4444; }
.kpi.blue  { border-top-color: #3b82f6; }
.kpi.amber { border-top-color: #f59e0b; }
.kpi-n   { font-size: 2.5rem; font-weight: 800; line-height: 1.1; }
.kpi-lbl { font-size: .75rem; color: #6b7280; text-transform: uppercase;
           letter-spacing: .06em; font-weight: 600; }
.kpi-s   { font-size: .82rem; color: #9ca3af; margin-top: 4px; }
.sec-h {
  font-size: 1.08rem; font-weight: 700; color: #1e3a5f;
  border-left: 4px solid #10b981; padding-left: 10px; margin-bottom: 14px;
}
.pol        { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
              padding: 14px 18px; font-size: .87rem; }
.pol.amber  { background: #fffbeb; border-color: #fde68a; }
.pol.red    { background: #fff5f5; border-color: #fecaca; }
.rank-card {
  background: #f8fafc; border-radius: 8px; padding: 12px 16px;
  text-align: center; border: 1px solid #e2e8f0; margin-top: 6px;
}
.foot {
  background: #f8fafc; border-top: 1px solid #e2e8f0; border-radius: 8px;
  padding: 10px 18px; font-size: .76rem; color: #6b7280; margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

#  Constants 
EXCLUIR = {
    "World", "Asia", "Europe", "Africa", "Americas", "Oceania",
    "North America", "South America", "European Union (27)",
    "High-income countries", "Low-income countries",
    "Upper-middle-income countries", "Lower-middle-income countries",
    "East Asia & Pacific", "Europe & Central Asia",
    "Latin America & Caribbean", "Middle East & North Africa",
    "Sub-Saharan Africa", "South Asia",
}

SE_LOWCARBON = 98.5
SE_CO2PC     = 3.5
SE_LIFE      = 83.2
SE_RESP_RATE = 14.4
YEARS_FUTURE = list(range(2025, 2036))

#  Data & Model 
@st.cache_data
def cargar_datos():
    return pd.read_csv(os.path.join(BASE_DIR, "master_dataset.csv"))


@st.cache_resource
def entrenar_model(_df):
    base = _df[~_df["Entity"].isin(EXCLUIR)].copy()
    dv = base.dropna(subset=["pct_lowcarbon", "gdp_pc", "life_exp"])
    sv = StandardScaler()
    Xv = sv.fit_transform(dv[["pct_lowcarbon", "gdp_pc"]])
    mv = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    mv.fit(Xv, dv["life_exp"])
    r2v = round(r2_score(dv["life_exp"], mv.predict(Xv)), 3)
    dr = base.dropna(subset=["pct_lowcarbon", "gdp_pc", "resp_death_rate"])
    sr = StandardScaler()
    Xr = sr.fit_transform(dr[["pct_lowcarbon", "gdp_pc"]])
    mr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    mr.fit(Xr, dr["resp_death_rate"])
    r2r = round(r2_score(dr["resp_death_rate"], mr.predict(Xr)), 3)
    return mv, sv, mr, sr, r2v, r2r


def proyectar(pais, df, mv, sv, mr, sr, anys):
    dp  = df[df["Entity"] == pais].sort_values("Year")
    ult = dp.dropna(subset=["pct_lowcarbon"]).iloc[-1]
    lc0 = ult["pct_lowcarbon"]
    pop = (dp.dropna(subset=["population"])["population"].iloc[-1]
           if dp["population"].notna().any() else 5e6)
    dlc = dp.dropna(subset=["pct_lowcarbon"]).tail(15)
    if len(dlc) < 3:
        return None, None
    reg_lc = LinearRegression().fit(dlc[["Year"]], dlc["pct_lowcarbon"])
    dg = dp.dropna(subset=["gdp_pc"]).tail(15)
    if len(dg) >= 3:
        reg_gdp = LinearRegression().fit(dg[["Year"]], dg["gdp_pc"])
        gdp_fn = lambda yr: max(500, reg_gdp.predict([[yr]])[0])  # noqa: E731
    else:
        gdp_val = dp["gdp_pc"].dropna().iloc[-1] if dp["gdp_pc"].notna().any() else 15000
        gdp_fn = lambda yr: gdp_val  # noqa: E731

    # BAU d'esperança de vida: regressió lineal sobre l'historial real del país
    # (evita les oscil·lacions del model ML en zones extrapolades)
    dv = dp.dropna(subset=["life_exp"]).tail(15)
    vida0 = float(dv["life_exp"].iloc[-1])
    if len(dv) >= 3:
        reg_vida = LinearRegression().fit(dv[["Year"]], dv["life_exp"])
        # Pendents màxims ±0.15 anys/any per evitar projeccions absurdes
        slope_vida = float(np.clip(reg_vida.coef_[0], -0.15, 0.15))
        vida_bau_fn = lambda yr: vida0 + slope_vida * (yr - int(dv["Year"].iloc[-1]))  # noqa: E731
    else:
        vida_bau_fn = lambda yr: vida0  # noqa: E731

    # IdEm per a mortalitat respiratòria
    dr2 = dp.dropna(subset=["resp_death_rate"]).tail(15)
    resp0 = float(dr2["resp_death_rate"].iloc[-1]) if not dr2.empty else 30.0
    if len(dr2) >= 3:
        reg_resp = LinearRegression().fit(dr2[["Year"]], dr2["resp_death_rate"])
        slope_resp = float(np.clip(reg_resp.coef_[0], -2.0, 0.5))
        resp_bau_fn = lambda yr: max(1.0, resp0 + slope_resp * (yr - int(dr2["Year"].iloc[-1])))  # noqa: E731
    else:
        resp_bau_fn = lambda yr: resp0  # noqa: E731

    # Delta de Suècia: diferència que prediu el model ML entre escenari net i BAU
    # S'aplica sobre la tendència real, no sobre la predicció absoluta del model
    gdp_ref = gdp_fn(2025)
    vida_ml_bau = float(mv.predict(sv.transform([[lc0, gdp_ref]]))[0])
    resp_ml_bau = float(mr.predict(sr.transform([[lc0, gdp_ref]]))[0])
    vida_ml_se  = float(mv.predict(sv.transform([[SE_LOWCARBON, gdp_ref]]))[0])
    resp_ml_se  = float(mr.predict(sr.transform([[SE_LOWCARBON, gdp_ref]]))[0])
    delta_vida  = float(np.clip(vida_ml_se - vida_ml_bau, -3.0, 5.0))
    delta_resp  = float(np.clip(resp_ml_se - resp_ml_bau, -60.0, 10.0))

    rows = []
    for i, yr in enumerate(YEARS_FUTURE):
        lc_bau = float(np.clip(reg_lc.predict([[yr]])[0], 0, 100))
        prog   = min(1.0, i / max(anys, 1))
        lc_se  = lc0 + prog * (SE_LOWCARBON - lc0)

        vida_bau = round(float(np.clip(vida_bau_fn(yr), 40, 95)), 2)
        resp_bau = round(resp_bau_fn(yr), 2)

        # Suècia: tendència base + delta proporcional al progrés de la transició
        vida_se  = round(float(np.clip(vida_bau + delta_vida * prog, 40, 95)), 2)
        resp_se  = round(float(np.clip(resp_bau + delta_resp * prog, 1, 200)), 2)

        rows.append({"Year": yr, "tag": "BAU", "lc": round(lc_bau, 1),
                     "vida": vida_bau, "morts": round(resp_bau * pop / 100_000, 0), "resp": resp_bau})
        rows.append({"Year": yr, "tag": "SE",  "lc": round(lc_se, 1),
                     "vida": vida_se,  "morts": round(resp_se  * pop / 100_000, 0), "resp": resp_se})

    d = pd.DataFrame(rows)
    return d[d.tag == "BAU"].set_index("Year"), d[d.tag == "SE"].set_index("Year")


def rang_global(df, col, pais, inv=False):
    base = (df[~df["Entity"].isin(EXCLUIR)]
            .sort_values("Year")
            .groupby("Entity")[col].last()
            .dropna())
    if pais not in base.index:
        return None, None, None
    val   = base[pais]
    total = len(base)
    rank  = int((base < val).sum() + 1) if not inv else int((base > val).sum() + 1)
    pct   = round((total - rank) / total * 100)
    return rank, total, pct


# 
df = cargar_datos()
mv, sv, mr, sr, r2v, r2r = entrenar_model(df)
paises = sorted([
    p for p in df["Entity"].unique()
    if p not in EXCLUIR
    and df[df["Entity"] == p]["pct_lowcarbon"].notna().sum() >= 8
    and df[df["Entity"] == p]["life_exp"].notna().sum() >= 8
])

# 
# SIDEBAR
# 
with st.sidebar:
    st.markdown("##  Simulador Climàtic")
    st.markdown("*Quantes vides salvaria adoptar el model d'energia neta de Suècia?*")
    st.divider()

    idx        = paises.index("Spain") if "Spain" in paises else 0
    pais_sel   = st.selectbox(" Selecciona el teu país", paises, index=idx)
    anio_trans = st.slider(" Anys fins a completar la transició", 1, 20, 10, 1)

    _dp = df[df["Entity"] == pais_sel].sort_values("Year")
    _lc = float(
        _dp.dropna(subset=["pct_lowcarbon"])["pct_lowcarbon"].iloc[-1]
        if not _dp.dropna(subset=["pct_lowcarbon"]).empty else 50.0
    )
    _dif = (SE_LOWCARBON - _lc) / anio_trans
    if _lc >= SE_LOWCARBON:
        st.success(f" {pais_sel} ja supera l'objectiu suec ({_lc:.1f}%)")
    else:
        st.info(
            f" Cal créixer **+{_dif:.1f}%/any**\n\n"
            f"De {_lc:.1f}%  {SE_LOWCARBON}% en {anio_trans} anys"
        )
    st.divider()
    st.markdown("** Referència Suècia:**")
    st.markdown(
        f"- Energia neta: **{SE_LOWCARBON}%**\n"
        f"- CO/hab: **{SE_CO2PC} t**\n"
        f"- Esperança de vida: **{SE_LIFE} anys**\n"
        f"- Mortalitat resp.: **{SE_RESP_RATE}/100k**"
    )
    st.divider()
    st.caption(
        f" Model: GradientBoosting\n"
        f" R(vida): **{r2v}** · R(resp): **{r2r}**\n"
        f" Dades fins 2022\n"
        f" Fonts: OWID · OMS · BM"
    )

# 
# DADES DEL PAÍS
# 
dp = df[df["Entity"] == pais_sel].sort_values("Year")


def ult(col):
    s = dp.dropna(subset=[col])
    return float(s.iloc[-1][col]) if not s.empty else np.nan


lc_actual    = ult("pct_lowcarbon")
vida_actual  = ult("life_exp")
resp_actual  = ult("resp_death_rate")
co2pc_actual = ult("co2_prod_pc")
gdp_actual   = ult("gdp_pc")
hdi_actual   = ult("hdi")
gini_actual  = ult("gini")
pop = float(
    dp.dropna(subset=["population"])["population"].iloc[-1]
    if dp["population"].notna().any() else 5e6
)

bau, suecia = proyectar(pais_sel, df, mv, sv, mr, sr, anio_trans)
if bau is None:
    st.error(f"No hi ha prou dades per a {pais_sel}.")
    st.stop()

vida_bau35  = bau.loc[2035, "vida"]
vida_se35   = suecia.loc[2035, "vida"]
morts_bau   = bau.loc[2026:, "morts"].sum()
morts_se    = suecia.loc[2026:, "morts"].sum()
morts_evit  = morts_bau - morts_se
anys_guany  = vida_se35 - vida_bau35
lc_gap      = SE_LOWCARBON - lc_actual

rank_lc,   tot_lc,   pct_lc   = rang_global(df, "pct_lowcarbon", pais_sel, inv=True)
rank_vida, tot_vida, pct_vida  = rang_global(df, "life_exp",      pais_sel, inv=True)

resp_str = f"{resp_actual:.1f}" if not np.isnan(resp_actual) else "N/A"

# ── Índex de Salut Climàtica Composta (ISCC) ──────────────────────────────────
def _chi_norm(val, col, inv=False):
    s = (df[~df["Entity"].isin(EXCLUIR)]
         .sort_values("Year").groupby("Entity")[col].last().dropna())
    if s.empty or s.max() == s.min():
        return 0.5
    n = (val - s.min()) / (s.max() - s.min())
    return float(np.clip(1 - n if inv else n, 0, 1))

chi_lc   = _chi_norm(lc_actual,   "pct_lowcarbon")
chi_vida = _chi_norm(vida_actual, "life_exp")  if not np.isnan(vida_actual)  else 0.5
chi_resp = _chi_norm(resp_actual, "resp_death_rate", inv=True) if not np.isnan(resp_actual) else 0.5
chi_score = round((chi_lc * 0.40 + chi_vida * 0.35 + chi_resp * 0.25) * 100, 1)
chi_grade = ("A+" if chi_score >= 90 else "A" if chi_score >= 80 else
             "B"  if chi_score >= 65 else "C" if chi_score >= 50 else
             "D"  if chi_score >= 35 else "F")
_CHI_COL  = {"A+":"#047857","A":"#10b981","B":"#3b82f6","C":"#f59e0b","D":"#f97316","F":"#ef4444"}[chi_grade]
_CHI_DESC = {"A+":"Excel\u00b7lent mundial","A":"Excel\u00b7lent","B":"Notable",
             "C":"Suficient","D":"Deficient","F":"Crític"}[chi_grade]

# 
# HERO BANNER
# 
if lc_actual >= 85:
    headline = f"Energia neta al {lc_actual:.0f}%  líder mundial"
elif lc_actual >= 50:
    headline = f"A {lc_gap:.0f} punts percentuals de l'objectiu suec"
else:
    headline = f"Oportunitat de transició: {lc_gap:.0f} pp fins a l'objectiu suec"

badge_txt = (
    f" Fins a {int(morts_evit):,} morts evitades  ·  "
    f" +{anys_guany:.2f} anys d'esperança de vida per a 2035"
    if morts_evit > 0
    else f"Impacte estimat 2035: {anys_guany:+.2f} anys d'esperança de vida"
)

st.markdown(f"""
<div class="hero">
  <div class="hero-title"> {pais_sel}</div>
  <div class="hero-sub">{headline}</div>
  <div class="hero-badge">{badge_txt}</div>
</div>
""", unsafe_allow_html=True)

# 
# KPI CARDS
# 
c1, c2, c3, c4 = st.columns(4)

lc_cls = "" if lc_actual >= 75 else ("amber" if lc_actual >= 40 else "red")
lc_col = "#10b981" if lc_actual >= 75 else ("#f59e0b" if lc_actual >= 40 else "#ef4444")
v_cls  = "" if anys_guany >= 0 else "red"
v_col  = "#10b981" if anys_guany >= 0 else "#ef4444"
m_cls  = "" if morts_evit >= 0 else "red"
m_col  = "#10b981" if morts_evit >= 0 else "#ef4444"
r_col  = "#10b981" if (pct_lc or 0) >= 60 else ("#f59e0b" if (pct_lc or 0) >= 30 else "#ef4444")

c1.markdown(
    f'<div class="kpi {lc_cls}"><div class="kpi-lbl"> Energia neta actual</div>'
    f'<div class="kpi-n" style="color:{lc_col}">{lc_actual:.1f}%</div>'
    f'<div class="kpi-s">Objectiu Suècia: {SE_LOWCARBON}%</div></div>',
    unsafe_allow_html=True,
)
c2.markdown(
    f'<div class="kpi {v_cls}"><div class="kpi-lbl"> Anys de vida guanyats (2035)</div>'
    f'<div class="kpi-n" style="color:{v_col}">{anys_guany:+.2f}</div>'
    f'<div class="kpi-s">{vida_bau35:.1f}  {vida_se35:.1f} anys</div></div>',
    unsafe_allow_html=True,
)
c3.markdown(
    f'<div class="kpi {m_cls}"><div class="kpi-lbl"> Morts evitades 20262035</div>'
    f'<div class="kpi-n" style="color:{m_col}">{int(morts_evit):,}</div>'
    f'<div class="kpi-s">per malalties respiratòries</div></div>',
    unsafe_allow_html=True,
)
rank_badge = f"#{rank_lc}" if rank_lc else ""
rank_sub   = f"Millor que el {pct_lc}% ({tot_lc} països)" if rank_lc else "Sense dades"
c4.markdown(
    f'<div class="kpi blue"><div class="kpi-lbl"> Rànquing energia neta</div>'
    f'<div class="kpi-n" style="color:{r_col}">{rank_badge}</div>'
    f'<div class="kpi-s">{rank_sub}</div></div>',
    unsafe_allow_html=True,
)

# ── ISCC Strip ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(90deg,{_CHI_COL}18 0%,transparent 100%);
  border-left:5px solid {_CHI_COL};border-radius:10px;padding:14px 24px;
  margin:10px 0 14px 0;display:flex;align-items:center;gap:28px">
  <div style="text-align:center;min-width:72px">
    <div style="font-size:3.2rem;font-weight:900;color:{_CHI_COL};line-height:1">{chi_grade}</div>
    <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;letter-spacing:.07em;margin-top:3px">Índex ISCC</div>
  </div>
  <div style="flex:1">
    <div style="font-size:1rem;font-weight:700;color:{_CHI_COL}">{_CHI_DESC} &nbsp;·&nbsp; {chi_score} / 100 punts</div>
    <div style="background:#e5e7eb;border-radius:5px;height:10px;margin:7px 0">
      <div style="background:{_CHI_COL};width:{chi_score}%;height:10px;border-radius:5px"></div>
    </div>
    <div style="font-size:.75rem;color:#6b7280">
      Índex de Salut Climàtica Composta (ISCC) &nbsp;·&nbsp;
      Energia neta <b>40%</b> + Esperança de vida <b>35%</b> + Salut respiratòria <b>25%</b>
    </div>
  </div>
  <div style="text-align:right;min-width:88px">
    <div style="font-size:1.8rem;font-weight:900;color:{_CHI_COL}">{chi_score}</div>
    <div style="font-size:.72rem;color:#6b7280">/ 100 punts</div>
    <div style="font-size:.78rem;font-weight:600;color:{_CHI_COL}">Top {100 - (pct_lc or 50)}% mundial</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Alertes contextuals
if (not np.isnan(co2pc_actual) and not np.isnan(gdp_actual)
        and co2pc_actual < 1.5 and gdp_actual < 5000):
    st.warning(
        " **Efecte Kuznets detectat:** Aquest país ja té emissions molt baixes "
        "degut al baix consum energètic. La millora de l'esperança de vida depèn "
        "més del desenvolupament econòmic i l'accés a la salut que de la transició energètica."
    )
elif lc_actual > 85:
    st.success(
        f" **{pais_sel}** ja té {lc_actual:.1f}% d'energia neta  a prop del nivell nòrdic. "
        "El repte és descarbonitzar el transport i la indústria."
    )
elif lc_actual < 25:
    st.warning(
        f" **{pais_sel}** té només {lc_actual:.1f}% d'energia neta. "
        "La transició energètica és la palanca més impactant per millorar la salut pública."
    )

st.divider()

# 
# FILA 1  Gauge + Projecció esperança de vida
# 
st.markdown(
    '<div class="sec-h"> Estat actual i projecció d\'esperança de vida</div>',
    unsafe_allow_html=True,
)
col_g, col_ev = st.columns([1, 2])

with col_g:
    fig_g = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = lc_actual,
        delta = {
            "reference": SE_LOWCARBON, "valueformat": ".1f",
            "suffix": "%", "relative": False,
        },
        title = {
            "text": (
                " Energia Neta Actual<br>"
                "<span style='font-size:.75em;color:#6b7280'>"
                f"Objectiu Suècia: {SE_LOWCARBON}%</span>"
            )
        },
        gauge = {
            "axis":        {"range": [0, 100], "tickwidth": 1},
            "bar":         {"color": "#3b82f6", "thickness": 0.35},
            "bgcolor":     "white",
            "borderwidth": 1,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0,  25], "color": "#fee2e2"},
                {"range": [25, 60], "color": "#fef3c7"},
                {"range": [60, 85], "color": "#d1fae5"},
                {"range": [85,100], "color": "#a7f3d0"},
            ],
            "threshold": {
                "line":      {"color": "#10b981", "width": 4},
                "thickness": 0.8,
                "value":     SE_LOWCARBON,
            },
        },
        number = {"suffix": "%", "font": {"size": 30}},
    ))
    fig_g.update_layout(
        height=270,
        margin=dict(t=65, b=10, l=20, r=20),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_g, use_container_width=True)

    if rank_lc is not None:
        bar_w = pct_lc or 0
        st.markdown(f"""
<div class="rank-card">
  <div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">
    Rànquing global · {pais_sel}
  </div>
  <div style="font-size:1.9rem;font-weight:800;color:#3b82f6;margin:4px 0">
    #{rank_lc}<span style="font-size:.85rem;color:#9ca3af"> / {tot_lc}</span>
  </div>
  <div style="background:#e5e7eb;border-radius:4px;height:7px;margin:6px 0">
    <div style="background:#3b82f6;width:{bar_w}%;height:7px;border-radius:4px"></div>
  </div>
  <div style="font-size:.78rem;color:#6b7280">Millor que el {pct_lc}% dels països</div>
</div>
""", unsafe_allow_html=True)

with col_ev:
    hist_v = dp.dropna(subset=["life_exp"])[["Year", "life_exp"]]
    hist_v = hist_v[hist_v["Year"] >= 2000]

    # Punt de connexió: últim valor real del país
    last_hist_year = int(hist_v["Year"].iloc[-1])
    last_hist_vida = float(hist_v["life_exp"].iloc[-1])

    # Afegim el punt de connexió al principi de les sèries de projecció
    bau_x    = [last_hist_year] + list(bau.index)
    bau_y    = [last_hist_vida] + list(bau["vida"])
    suecia_x = [last_hist_year] + list(suecia.index)
    suecia_y = [last_hist_vida] + list(suecia["vida"])

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=hist_v["Year"], y=hist_v["life_exp"],
        name="Històric", mode="lines",
        line=dict(color="#6b7280", width=2.5),
    ))
    fig_ev.add_trace(go.Scatter(
        x=bau_x, y=bau_y,
        name="Tendència actual (BAU)",
        mode="lines+markers",
        line=dict(color="#ef4444", width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig_ev.add_trace(go.Scatter(
        x=suecia_x, y=suecia_y,
        name=f"Model Suècia ({anio_trans}a)",
        mode="lines+markers",
        line=dict(color="#10b981", width=3),
        marker=dict(size=5),
        fill="tonexty",
        fillcolor="rgba(16,185,129,0.10)",
    ))
    if abs(anys_guany) > 0.05:
        fig_ev.add_annotation(
            x=2035, y=vida_se35,
            text=f"<b>{anys_guany:+.2f} anys</b>",
            showarrow=True, arrowhead=2, arrowcolor="#10b981",
            font=dict(color="#10b981", size=12),
            bgcolor="rgba(240,253,244,0.9)", borderpad=4,
        )
    fig_ev.add_vline(
        x=2024.5, line_dash="dot", line_color="#9ca3af",
        annotation_text="Projecció ", annotation_position="top",
    )
    fig_ev.add_hline(
        y=SE_LIFE, line_dash="dot", line_color="#10b981", opacity=0.5,
        annotation_text=f"Suècia: {SE_LIFE}a",
    )
    fig_ev.update_layout(
        title=f"❤️ Esperança de vida: {pais_sel} (2000–2035)",
        xaxis_title="Any",
        yaxis_title="Anys",
        legend=dict(orientation="h", yanchor="bottom", y=-0.30),
        height=350,
        plot_bgcolor="white",
        margin=dict(b=80),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

st.divider()

# 
# FILA 2  Transició energètica + Morts evitades
# 
st.markdown(
    '<div class="sec-h"> Transició energètica i impacte en salut pública</div>',
    unsafe_allow_html=True,
)
col_L, col_R = st.columns(2)

with col_L:
    hist_lc = dp.dropna(subset=["pct_lowcarbon"])[["Year", "pct_lowcarbon"]]
    hist_lc = hist_lc[hist_lc["Year"] >= 1990]

    # Punt de connexió: últim valor real d'energia neta
    last_lc_year = int(hist_lc["Year"].iloc[-1])
    last_lc_val  = float(hist_lc["pct_lowcarbon"].iloc[-1])

    bau_lc_x    = [last_lc_year] + list(bau.index)
    bau_lc_y    = [last_lc_val]  + list(bau["lc"])
    suecia_lc_x = [last_lc_year] + list(suecia.index)
    suecia_lc_y = [last_lc_val]  + list(suecia["lc"])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hist_lc["Year"], y=hist_lc["pct_lowcarbon"],
        name="Històric", line=dict(color="#6b7280", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=bau_lc_x, y=bau_lc_y,
        name="BAU (sense canvi)",
        line=dict(color="#ef4444", width=2.5, dash="dash"),
        mode="lines+markers", marker=dict(size=5),
    ))
    fig1.add_trace(go.Scatter(
        x=suecia_lc_x, y=suecia_lc_y,
        name=f"Model Suècia ({anio_trans}a)",
        line=dict(color="#10b981", width=3),
        mode="lines+markers", marker=dict(size=5),
        fill="tonexty", fillcolor="rgba(16,185,129,0.08)",
    ))
    fig1.add_hline(
        y=SE_LOWCARBON, line_dash="dot", line_color="#10b981", opacity=0.6,
        annotation_text=f"Objectiu Suècia ({SE_LOWCARBON}%)",
    )
    fig1.add_vline(
        x=last_lc_year + 0.5, line_dash="dot", line_color="#9ca3af",
        annotation_text="Projecció →", annotation_position="top",
    )
    fig1.add_vrect(x0=last_lc_year + 0.5, x1=2035.5, fillcolor="#10b981", opacity=0.03)
    fig1.update_layout(
        title="⚡ % Electricitat de fonts netes",
        xaxis_title="Any", yaxis_title="% energia neta",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        height=380, margin=dict(b=80), plot_bgcolor="white",
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_R:
    anys_fut    = list(range(2026, 2036))
    evit_acum   = []
    total_acum  = 0.0
    for yr in anys_fut:
        total_acum += bau.loc[yr, "morts"] - suecia.loc[yr, "morts"]
        evit_acum.append(total_acum)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=anys_fut, y=[max(0, v) for v in evit_acum],
        marker_color="#10b981", name="Morts evitades",
        text=[f"{int(v):,}" for v in evit_acum], textposition="outside",
    ))
    fig2.add_trace(go.Bar(
        x=anys_fut, y=[abs(v) if v < 0 else 0 for v in evit_acum],
        marker_color="#ef4444", name="Morts addicionals",
    ))
    fig2.update_layout(
        title=" Morts respiratòries evitades (acumulades)",
        xaxis_title="Any", yaxis_title="Morts evitades (acum.)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        height=380, margin=dict(b=60), plot_bgcolor="white",
        barmode="relative",
        annotations=[dict(
            x=0.5, y=1.12, xref="paper", yref="paper",
            text=f"Total: <b>{int(morts_evit):,} morts evitades en 10 anys</b>",
            showarrow=False,
            font=dict(size=13, color="#10b981" if morts_evit >= 0 else "#ef4444"),
            bgcolor="rgba(240,253,244,0.9)", borderpad=6,
        )],
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# 
# RESUM EXECUTIU + RECOMANACIONS
# 
st.markdown(
    '<div class="sec-h"> Resum executiu i recomanacions de política pública</div>',
    unsafe_allow_html=True,
)
col_a, col_b = st.columns(2)

with col_a:
    st.info(f"""
** Situació actual de {pais_sel}:**
- Energia neta: **{lc_actual:.1f}%** (Suècia: {SE_LOWCARBON}%)
- Esperança de vida: **{vida_actual:.1f} anys** (Suècia: {SE_LIFE})
- Mortalitat resp.: **{resp_str}/100k** (Suècia: {SE_RESP_RATE}/100k)
- Població: **{pop / 1e6:.1f} milions**
- Rànquing energia neta: **#{rank_lc}** de {tot_lc} països
""")

with col_b:
    if morts_evit > 0 and anys_guany > 0:
        st.success(f"""
** Adoptant el model de Suècia en {anio_trans} anys:**
-  S'evitarien **{int(morts_evit):,} morts** respiratòries (20262035)
-  L'esperança de vida milloraria **{anys_guany:+.2f} anys** el 2035
-  Objectiu: {lc_actual:.1f}%  {SE_LOWCARBON}% d'energia neta
-  Increment anual necessari: +{(SE_LOWCARBON - lc_actual) / anio_trans:.1f}%/any
""")
    elif lc_actual > 85:
        st.success(f"""
** {pais_sel} ja té una matriu energètica neta:**
-  {lc_actual:.1f}% d'electricitat neta  nivell nòrdic assolit
-  El repte és descarbonitzar el transport i la indústria
-  Potencial de millora: {anys_guany:+.2f} anys d'esperança de vida
""")
    else:
        st.warning(f"""
** Adoptant el model de Suècia en {anio_trans} anys:**
-  Impacte estimat: **{anys_guany:+.2f} anys** d'esperança de vida
-  Morts resp. evitades: **{int(morts_evit):,}**
-  Clau: transició energètica + polítiques de qualitat de l'aire
""")

st.divider()

# 
# COMPARATIVA vs SUÈCIA
# 
st.markdown(
    f'<div class="sec-h"> Comparativa: {pais_sel} vs  Suècia</div>',
    unsafe_allow_html=True,
)

dse = df[df["Entity"] == "Sweden"].sort_values("Year")


def se_v(col):
    s = dse.dropna(subset=[col])
    return float(s.iloc[-1][col]) if not s.empty else np.nan


se_lc2   = se_v("pct_lowcarbon")
se_co2   = se_v("co2_prod_pc")
se_gdp   = se_v("gdp_pc")
se_hdi   = se_v("hdi")
se_gini  = se_v("gini")
se_resp2 = se_v("resp_death_rate")
se_vida2 = se_v("life_exp")

indicadors = {
    " Energia neta (%)":          (lc_actual,    se_lc2,   True,  "{:.1f}%"),
    " CO/càpita (t/hab)":       (co2pc_actual, se_co2,   False, "{:.2f} t"),
    " Esperança de vida (anys)":  (vida_actual,  se_vida2, True,  "{:.1f} anys"),
    " Mortalitat resp. (/100k)":  (resp_actual,  se_resp2, False, "{:.1f}"),
    " PIB per càpita (USD)":      (gdp_actual,   se_gdp,   True,  "{:,.0f} $"),
    " IDH (01)":                  (hdi_actual,   se_hdi,   True,  "{:.3f}"),
    " Índex Gini (desigualtat)":  (gini_actual,  se_gini,  False, "{:.1f}"),
}

col_tab, col_rad = st.columns([1, 1])

with col_tab:
    taula = []
    for nom, (vp, vs, major, fmt) in indicadors.items():
        if np.isnan(vp) or np.isnan(vs):
            estat, dif_s = "", "Sense dades"
        else:
            dif  = vp - vs
            pct  = (dif / vs * 100) if vs else 0
            bon  = (dif > 0) == major
            estat = " Igual" if abs(pct) < 2 else (" Millor" if bon else " Pitjor")
            dif_s = f"{dif:+.2f} ({pct:+.1f}%)"
        taula.append({
            "Indicador":  nom,
            pais_sel:     fmt.format(vp) if not np.isnan(vp) else "",
            " Suècia": fmt.format(vs) if not np.isnan(vs) else "",
            "Diferència": dif_s,
            "Estat":      estat,
        })
    st.dataframe(pd.DataFrame(taula), hide_index=True, use_container_width=True)

with col_rad:
    CATS = ["Energia\nneta", "CO\nbaix", "Esp.\nvida", "Mortalitat\nbaixa", "IDH", "PIB/hab"]

    def norm(val, col, inv=False):
        s = df[~df["Entity"].isin(EXCLUIR)][col].dropna()
        if s.empty or s.max() == s.min():
            return 0.5
        n = (val - s.min()) / (s.max() - s.min())
        return float(np.clip(1 - n if inv else n, 0, 1))

    def radar(lc, co2, vida, resp, hdi, gdp):
        return [
            norm(lc,   "pct_lowcarbon"),
            norm(co2,  "co2_prod_pc",     inv=True),
            norm(vida, "life_exp"),
            norm(resp, "resp_death_rate",  inv=True),
            norm(hdi,  "hdi"),
            norm(gdp,  "gdp_pc"),
        ]

    vp = radar(
        lc_actual,
        co2pc_actual if not np.isnan(co2pc_actual) else 5,
        vida_actual,
        resp_actual  if not np.isnan(resp_actual)  else 30,
        hdi_actual   if not np.isnan(hdi_actual)   else 0.5,
        gdp_actual   if not np.isnan(gdp_actual)   else 15000,
    )
    vs_r = radar(
        se_lc2, se_co2, se_vida2, se_resp2,
        se_hdi  if not np.isnan(se_hdi)  else 0.95,
        se_gdp  if not np.isnan(se_gdp)  else 55000,
    )

    fig_r = go.Figure()
    for nom, vals, color, fill in [
        (pais_sel,     vp,   "#3b82f6", "rgba(59,130,246,0.18)"),
        (" Suècia", vs_r, "#10b981", "rgba(16,185,129,0.18)"),
    ]:
        cv = vals + [vals[0]]
        cc = CATS + [CATS[0]]
        fig_r.add_trace(go.Scatterpolar(
            r=cv, theta=cc, fill="toself",
            fillcolor=fill, name=nom,
            line=dict(color=color, width=2.5),
        ))
    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickformat=".0%", tickfont=dict(size=9),
            ),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        title="Perfil comparatiu (0 = pitjor mundial · 1 = millor mundial)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=420,
    )
    st.plotly_chart(fig_r, use_container_width=True)

# 
# MAPA ANIMAT CO2
# 
st.divider()
st.markdown(
    '<div class="sec-h">&#127757; Mapa animat mundial: emissions CO&#8322; per c&#224;pita (2000&#8211;2035)</div>',
    unsafe_allow_html=True,
)
st.caption(
    "\u25b6\ufe0f Prem **Play** per veure l'evoluci\u00f3 hist\u00f2rica i la **projecci\u00f3 2025\u20132035**. "
    "Colors vermells = m\u00e9s emissions. Verd = m\u00e9s net. "
    f"{pais_sel} destacat amb vorera daurada."
)

# --- Dades hist\u00f2riques CO2 per paese (des de 2000) ---
_co2_hist = (
    df[~df["Entity"].isin(EXCLUIR)]
    .dropna(subset=["co2_prod_pc"])
    [["Entity", "Year", "co2_prod_pc"]]
    .copy()
)
_co2_hist = _co2_hist[_co2_hist["Year"] >= 2000].copy()
_co2_hist["tipus"] = "Hist\u00f2ric"

# --- Projecci\u00f3 2025-2035 per regressi\u00f3 lineal per a cada pa\u00eds ---
_proj_rows = []
for _ent, _grp in _co2_hist.groupby("Entity"):
    _grp2 = _grp.sort_values("Year")
    if len(_grp2) < 5:
        continue
    _tail = _grp2.tail(12)
    _reg  = LinearRegression().fit(_tail[["Year"]], _tail["co2_prod_pc"])
    _slope = float(np.clip(_reg.coef_[0], -1.5, 1.5))
    _last_val = float(_tail["co2_prod_pc"].iloc[-1])
    _last_yr  = int(_tail["Year"].iloc[-1])
    for _yr in range(2025, 2036):
        _v = max(0.0, _last_val + _slope * (_yr - _last_yr))
        _proj_rows.append({
            "Entity": _ent, "Year": _yr,
            "co2_prod_pc": round(_v, 2), "tipus": "Projecci\u00f3",
        })

_co2_all = pd.concat([_co2_hist, pd.DataFrame(_proj_rows)], ignore_index=True)
_co2_all["co2_prod_pc"] = _co2_all["co2_prod_pc"].clip(0, 35)
_co2_all = _co2_all.sort_values("Year")
_co2_all["Any"] = _co2_all["Year"].astype(str)

# --- Construir el choropleth animat ---
_CSCALE = [
    [0.00, "#d1fae5"], [0.10, "#a7f3d0"], [0.25, "#fef3c7"],
    [0.45, "#f97316"], [0.70, "#dc2626"], [1.00, "#7f1d1d"],
]
fig_co2map = px.choropleth(
    _co2_all,
    locations     = "Entity",
    locationmode  = "country names",
    color         = "co2_prod_pc",
    animation_frame = "Any",
    hover_name    = "Entity",
    hover_data    = {"co2_prod_pc": ":.2f", "tipus": True, "Any": False},
    color_continuous_scale = _CSCALE,
    range_color   = [0, 20],
    labels        = {"co2_prod_pc": "CO\u2082/hab (t)", "tipus": "Tipus", "Entity": "Pa\u00eds"},
)

fig_co2map.update_layout(
    title="Emissions CO\u2082 per c\u00e0pita (t/hab) \u00b7 2000\u20132024 hist\u00f2ric + 2025\u20132035 projecci\u00f3",
    coloraxis_colorbar=dict(
        title="CO\u2082/hab (t)",
        thickness=14, len=0.75,
        tickvals=[0, 5, 10, 15, 20],
        ticktext=["0", "5", "10", "15", "20+"],
    ),
    geo=dict(
        showframe=False, showcoastlines=True,
        projection_type="natural earth",
        coastlinecolor="white", landcolor="#f9fafb",
        showocean=True, oceancolor="#dbeafe",
    ),
    margin=dict(t=50, b=10, l=0, r=0), height=480,
)
# Velocitat de l'animaci\u00f3: 600ms per frame
try:
    fig_co2map.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = 600
    fig_co2map.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
except Exception:
    pass
st.plotly_chart(fig_co2map, use_container_width=True)

# 
# EQUIVALENCIES VISCERALS
# 
st.divider()
st.markdown(
    '<div class="sec-h">&#128161; Impacte en perspectiva: quant \u00e9s realment?</div>',
    unsafe_allow_html=True,
)
_co2_gap   = max(0.0, (co2pc_actual - SE_CO2PC) if not np.isnan(co2pc_actual) else 0.0)
_co2_saved = _co2_gap * pop * anio_trans
_eq_flights = int(_co2_saved / 0.9)
_eq_trees   = int(_co2_saved / (0.022 * anio_trans)) if anio_trans > 0 else 0
_eq_cars_km = int(_co2_saved / 0.00021)
_morts_any  = int(abs(morts_evit) / max(1, anio_trans))
_eq_cols = st.columns(4)
for _col, _icon, _val, _title, _sub, _c in [
    (_eq_cols[0], "&#9992;&#65039;", f"{_eq_flights:,}",
     "Vols transatl\u00e0ntics",
     f"emissions CO\u2082 estalviades en {anio_trans}a", "#3b82f6"),
    (_eq_cols[1], "&#127795;", f"{_eq_trees:,}",
     "Arbres necessaris",
     f"per absorbir el CO\u2082 equivalent en {anio_trans}a", "#10b981"),
    (_eq_cols[2], "&#128664;", f"{_eq_cars_km / 1e9:.1f}B km",
     "Recorregut en cotxe",
     "km equivalents en emissions estalviades", "#f59e0b"),
    (_eq_cols[3], "&#129753;", f"{_morts_any:,}/any",
     "Morts resp. evitades",
     f"cada any durant la transici\u00f3 ({anio_trans}a)",
     "#10b981" if morts_evit >= 0 else "#ef4444"),
]:
    _col.markdown(f"""
<div style="background:white;border-radius:10px;padding:18px 12px;
  box-shadow:0 2px 10px rgba(0,0,0,0.07);border-top:3px solid {_c};text-align:center">
  <div style="font-size:2.2rem;margin-bottom:4px">{_icon}</div>
  <div style="font-size:1.4rem;font-weight:800;color:{_c};margin:4px 0">{_val}</div>
  <div style="font-size:.76rem;font-weight:600;color:#374151">{_title}</div>
  <div style="font-size:.70rem;color:#9ca3af;margin-top:3px">{_sub}</div>
</div>""", unsafe_allow_html=True)

# 
# FOOTER
# 
st.markdown(f"""
<div class="foot">
   <b>Fonts:</b> Our World in Data · OMS · Banc Mundial · Global Carbon Project
  &nbsp;|&nbsp;
   <b>Model:</b> GradientBoostingRegressor
  &nbsp;|&nbsp;
   R(vida) = {r2v} · R(resp) = {r2r}
  &nbsp;|&nbsp;
   Dades fins 2022
  &nbsp;|&nbsp;
   DADM 20252026
  &nbsp;|&nbsp;
   <a href="https://emisionesco2.streamlit.app/" target="_blank">emisionesco2.streamlit.app</a>
</div>
""", unsafe_allow_html=True)
