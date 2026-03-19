"""
REPTE 7 — Emissions de CO₂ Globals
Fase 3: Correlacions
  3.1 Corba de Kuznets — PIB/HDI vs CO₂
  3.2 Producció vs Consum — qui externalitza emissions
  3.3 Renovables vs intensitat de carboni
  3.4 Desigualtat (Gini) vs CO₂
  3.5 Exportacions vs CO₂
  3.6 Salut — esperança de vida i malalties respiratòries vs CO₂
  3.7 Heatmap general de correlacions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "grafics")
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})

master = pd.read_csv(os.path.join(BASE, "master_dataset.csv"))

# Any de referència per correlacions (màxima cobertura de dades)
ANY_REF = 2019  # Pre-COVID, molts indicators disponibles

df = master[master["Year"] == ANY_REF].dropna(subset=["co2_prod_pc", "gdp_pc", "population"]).copy()

# Helper per imprimir correlació
def print_corr(x, y, label_x, label_y, df_sub):
    d = df_sub[[x, y]].dropna()
    r_p, p_p = stats.pearsonr(d[x], d[y])
    r_s, p_s = stats.spearmanr(d[x], d[y])
    print(f"  Pearson r={r_p:.3f} (p={p_p:.3e}) | Spearman r={r_s:.3f} (p={p_s:.3e})")
    return r_p, p_p

# =============================================================================
# 3.1 CORBA DE KUZNETS — PIB vs CO₂ per càpita
# =============================================================================
print("3.1 Corba de Kuznets...")

d_kuz = df.dropna(subset=["gdp_pc", "co2_prod_pc", "hdi"]).copy()
d_kuz = d_kuz[d_kuz["co2_prod_pc"] > 0]

# Regió per color
gdp_col = df_gdp_region = pd.read_csv(os.path.join(BASE, "gdp-per-capita-worldbank.csv"))
gdp_col = gdp_col[gdp_col["Year"] == ANY_REF][["Code", "World region according to OWID"]].rename(
    columns={"World region according to OWID": "region"})
d_kuz = d_kuz.merge(gdp_col, on="Code", how="left")

regions = d_kuz["region"].dropna().unique()
palette = dict(zip(regions, sns.color_palette("tab10", len(regions))))

# --- 3.1a: PIB vs CO₂ per càpita ---
r_gdp, _ = stats.spearmanr(np.log(d_kuz["gdp_pc"]), np.log(d_kuz["co2_prod_pc"]))
fig, ax = plt.subplots(figsize=(9, 7))
for reg in regions:
    sub = d_kuz[d_kuz["region"] == reg]
    ax.scatter(sub["gdp_pc"], sub["co2_prod_pc"],
               label=reg, alpha=0.7, s=sub["population"]/5e6 + 20,
               color=palette.get(reg, "gray"))

x_log = np.log(d_kuz["gdp_pc"])
y_log = np.log(d_kuz["co2_prod_pc"])
z = np.polyfit(x_log, y_log, 2)
p = np.poly1d(z)
x_range = np.linspace(x_log.min(), x_log.max(), 200)
ax.plot(np.exp(x_range), np.exp(p(x_range)), color="black",
        linewidth=2, linestyle="--", label="Tendència (polinòmica)", zorder=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("PIB per Càpita (USD, escala log)")
ax.set_ylabel("CO₂ per Càpita (t, escala log)")
ax.set_title(f"PIB vs Emissions per Càpita\nSpearman r = {r_gdp:.3f}", fontweight="bold")
leg_reg = ax.legend(fontsize=8, loc="upper left", title="Regió")
ax.add_artist(leg_reg)
# Llegenda de mida (població) — només cercles grisos
import matplotlib.lines as mlines
size_handles = [mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                              markersize=(pop_ref/5e6 + 20)**0.5, alpha=0.6, label=lbl)
                for pop_ref, lbl in [(100e6, "100M hab."), (500e6, "500M hab."), (1400e6, "1.400M hab.")]]
ax.legend(handles=size_handles, fontsize=8, loc="lower right", title="Població")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.1a_pib_co2_percapita.png"), bbox_inches="tight")
plt.show()
print(f"   ✅ Gràfic guardat: 3.1a_pib_co2_percapita.png")

# --- 3.1b: HDI vs CO₂ per càpita ---
d_hdi = d_kuz.dropna(subset=["hdi"])
r_hdi, _ = stats.spearmanr(d_hdi["hdi"], d_hdi["co2_prod_pc"])
fig, ax2 = plt.subplots(figsize=(9, 7))
for reg in regions:
    sub = d_hdi[d_hdi["region"] == reg]
    ax2.scatter(sub["hdi"], sub["co2_prod_pc"],
                label=reg, alpha=0.7, s=sub["population"]/5e6 + 20,
                color=palette.get(reg, "gray"))

ax2.set_xlabel("Índex de Desenvolupament Humà (HDI)")
ax2.set_ylabel("CO₂ per Càpita (t)")
ax2.set_title(f"HDI vs Emissions per Càpita\nSpearman r = {r_hdi:.3f}", fontweight="bold")
leg_reg2 = ax2.legend(fontsize=8, loc="upper left", title="Regió")
ax2.add_artist(leg_reg2)
# Llegenda de mida (població) — només cercles grisos
size_handles2 = [mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                               markersize=(pop_ref/5e6 + 20)**0.5, alpha=0.6, label=lbl)
                 for pop_ref, lbl in [(100e6, "100M hab."), (500e6, "500M hab."), (1400e6, "1.400M hab.")]]
ax2.legend(handles=size_handles2, fontsize=8, loc="upper right", title="Població")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.1b_hdi_co2_percapita.png"), bbox_inches="tight")
plt.show()
print_corr("gdp_pc", "co2_prod_pc", "GDP pc", "CO₂ pc", d_kuz)
print(f"   ✅ Gràfic guardat: 3.1b_hdi_co2_percapita.png")

# =============================================================================
# 3.2 PRODUCCIÓ vs CONSUM — qui externalitza emissions
# =============================================================================
print("3.2 Producció vs Consum...")

d_trade = master[master["Year"] >= 1990].dropna(subset=["co2_prod", "co2_cons"]).copy()
d_trade["co2_trade_Gt"] = d_trade["co2_trade_balance"] / 1e9

# Any recent per balanç
any_trade = d_trade["Year"].max()
d_trade_yr = d_trade[d_trade["Year"] == any_trade].copy()
d_trade_yr = d_trade_yr.dropna(subset=["co2_trade_Gt"])

# Top exportadors i importadors nets
top_export = d_trade_yr.nlargest(10, "co2_trade_Gt")[["Entity", "co2_trade_Gt"]]
top_import = d_trade_yr.nsmallest(10, "co2_trade_Gt")[["Entity", "co2_trade_Gt"]]
top_both = pd.concat([top_export, top_import]).sort_values("co2_trade_Gt")

fig, ax = plt.subplots(figsize=(12, 9))
colors_trade = ["#d62728" if v > 0 else "#1f77b4" for v in top_both["co2_trade_Gt"]]
bars = ax.barh(range(len(top_both)), top_both["co2_trade_Gt"], color=colors_trade,
               edgecolor="white", height=0.7)

# Noms de països sempre fora de la barra
for i, (val, name) in enumerate(zip(top_both["co2_trade_Gt"], top_both["Entity"])):
    offset = 0.012 if val > 0 else -0.012
    ha = "left" if val > 0 else "right"
    ax.text(val + offset, i, name, ha=ha, va="center",
            fontsize=9, fontweight="bold", color="black")

ax.set_yticks([])  # Eliminar eix Y (ja posem noms a les barres)
ax.axvline(0, color="black", linewidth=1.5)
max_abs = top_both["co2_trade_Gt"].abs().max()
ax.set_xlim(-max_abs * 1.45, max_abs * 1.45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Balanç comercial de CO₂ (Gt)  ·  Positiu = exporta emissions  ·  Negatiu = importa emissions",
              fontsize=10)
ax.set_title("Qui Externalitza Emissions?\nProducció vs Consum de CO₂ — Vermell: exporta  |  Blau: importa",
             fontweight="bold", fontsize=13)

# Llegenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#d62728", label="Exporta emissions (fabrica per al món)"),
                   Patch(facecolor="#1f77b4", label="Importa emissions (compra productes d'altri)")]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.2_produccio_vs_consum_co2.png"), bbox_inches="tight")
plt.show()
print(f"   ✅ Gràfic guardat: 3.2_produccio_vs_consum_co2.png")

# Evolució temporal del balanç dels grans actors
actors = ["China", "United States", "Germany", "United Kingdom", "India", "France"]
fig, ax = plt.subplots(figsize=(13, 6))
for actor in actors:
    data = d_trade[d_trade["Entity"] == actor].sort_values("Year")
    if len(data) > 0:
        ax.plot(data["Year"], data["co2_trade_Gt"], linewidth=2, label=actor)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.fill_between([], [], [], color="#d62728", alpha=0.3, label="Zona: exporta CO₂")
ax.fill_between([], [], [], color="#1f77b4", alpha=0.3, label="Zona: importa CO₂")
ax.set_title("Evolució del Balanç Comercial de CO₂ — Grans Actors (1990–2022)", fontweight="bold")
ax.set_xlabel("Any")
ax.set_ylabel("Gt CO₂ (producció − consum)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.2b_balang_co2_temporal.png"), bbox_inches="tight")
plt.show()
print(f"   ✅ Gràfic guardat: 3.2b_balang_co2_temporal.png")

# =============================================================================
# 3.3 RENOVABLES vs INTENSITAT DE CARBONI
# =============================================================================
print("3.3 Renovables vs intensitat carboni...")

d_ren = df.dropna(subset=["pct_lowcarbon", "carbon_intensity", "co2_prod_pc", "hdi"]).copy()
d_ren = d_ren[(d_ren["carbon_intensity"] > 0) & (d_ren["hdi"] >= 0.7)]

r_ren, _ = stats.spearmanr(d_ren["pct_lowcarbon"], d_ren["co2_prod_pc"])

fig, ax = plt.subplots(figsize=(10, 8))
axes = [ax]  # compatibilitat

# Barres de països seleccionats, ordenats per % renovables
paisos_sel = ["Norway", "Sweden", "France", "Spain", "Brazil",
              "Italy", "United Kingdom", "Germany", "Japan",
              "United States", "China", "South Korea", "Australia",
              "Canada", "Qatar"]
d_bar = d_ren[d_ren["Entity"].isin(paisos_sel)].copy()
d_bar = d_bar.sort_values("pct_lowcarbon", ascending=True)

ax = axes[0]
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
bar_colors = ["#2ca02c" if v >= 50 else "#ff7f0e" if v >= 25 else "#d62728"
              for v in d_bar["pct_lowcarbon"]]
bars = ax.barh(d_bar["Entity"], d_bar["co2_prod_pc"], color=bar_colors, edgecolor="white", height=0.7)

# Valor de % renovables a la dreta de cada barra
for bar, (_, row) in zip(bars, d_bar.iterrows()):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
            f"{row['pct_lowcarbon']:.0f}% ren.", va="center", fontsize=8, color="dimgray")

ax.set_xlabel("CO₂ per Càpita (t)")
ax.set_title(f"Més Renovables → Menys CO₂?\nSpearman r = {r_ren:.3f}  (ordenat per % renovables, baix→alt)",
             fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Llegenda colors
from matplotlib.patches import Patch
leg = [Patch(facecolor="#2ca02c", label="≥ 50% renovables"),
       Patch(facecolor="#ff7f0e", label="25–50% renovables"),
       Patch(facecolor="#d62728", label="< 25% renovables")]
ax.legend(handles=leg, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.3_renovables_carboni.png"), bbox_inches="tight")
plt.show()
print_corr("pct_lowcarbon", "co2_prod_pc", "% Low-carbon", "CO₂ pc", d_ren)
print(f"   ✅ Gràfic guardat: 3.3_renovables_carboni.png")

# =============================================================================
# 3.4 DESIGUALTAT (GINI) vs CO₂
# =============================================================================
print("3.4 Gini vs CO₂...")

d_gini = master[(master["Year"] >= 2015) & (master["Year"] <= 2020)].copy()
d_gini = d_gini.dropna(subset=["gini", "co2_prod_pc", "gdp_pc"])
d_gini = d_gini.groupby("Entity")[["gini", "co2_prod_pc", "gdp_pc", "population"]].mean().reset_index()

# Eliminar outliers extrems: països petrolers amb Gini alt i CO₂ altíssim (>25t)
d_gini = d_gini[d_gini["co2_prod_pc"] <= 25]

fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(d_gini["gini"], d_gini["co2_prod_pc"],
                c=d_gini["gdp_pc"], cmap="Blues", alpha=0.75,
                s=d_gini["population"]/5e6 + 25, edgecolors="gray", linewidth=0.3)
plt.colorbar(sc, ax=ax, label="PIB per Càpita (USD)")

# Etiquetes dels outliers
outliers = pd.concat([
    d_gini.nlargest(4, "co2_prod_pc"),
    d_gini.nlargest(5, "gini"),
])
for _, row in outliers.drop_duplicates("Entity").iterrows():
    ax.annotate(row["Entity"], (row["gini"], row["co2_prod_pc"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")

r_gini_p, p_gini = stats.pearsonr(d_gini["gini"], d_gini["co2_prod_pc"])
r_gini_s, _ = stats.spearmanr(d_gini["gini"], d_gini["co2_prod_pc"])

# Línia de regressió
m, b = np.polyfit(d_gini["gini"], d_gini["co2_prod_pc"], 1)
x_line = np.linspace(d_gini["gini"].min(), d_gini["gini"].max(), 100)
ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5, linestyle="--")

ax.set_xlabel("Índex de Gini (0 = igualtat total, 1 = màxima desigualtat)")
ax.set_ylabel("CO₂ per Càpita (t, mitjana 2010-2020)")
ax.set_title(f"Desigualtat (Gini) vs Emissions CO₂ per Càpita\n"
             f"Pearson r = {r_gini_p:.3f} | Spearman r = {r_gini_s:.3f}", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.4_gini_co2.png"), bbox_inches="tight")
plt.show()
print(f"  Pearson r={r_gini_p:.3f} (p={p_gini:.3e}) | Spearman r={r_gini_s:.3f}")
print(f"   ✅ Gràfic guardat: 3.4_gini_co2.png")

# =============================================================================
# 3.5 EXPORTACIONS vs CO₂
# =============================================================================
print("3.5 Exportacions vs CO₂...")

d_exp = df.dropna(subset=["exports_usd", "co2_prod", "gdp_pc", "population"]).copy()
d_exp = d_exp[d_exp["exports_usd"] > 0]
d_exp["exports_pc"] = d_exp["exports_usd"] / d_exp["population"]

r_exp, p_exp = stats.spearmanr(d_exp["exports_pc"], d_exp["co2_prod_pc"])

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(d_exp["exports_pc"], d_exp["co2_prod_pc"],
           alpha=0.6, s=50, color="#2ca02c", edgecolors="gray", linewidth=0.3)

# Corba exponencial: ajust quadràtic en espai log → corba ascendent
log_x = np.log10(d_exp["exports_pc"])
z = np.polyfit(log_x, d_exp["co2_prod_pc"], 2)
p_trend = np.poly1d(z)
x_range = np.linspace(log_x.min(), log_x.max(), 300)
ax.plot(10**x_range, p_trend(x_range), color="#d62728", linewidth=2.5,
        linestyle="--", label="Tendencia exponencial")
ax.legend(loc="upper left", fontsize=10)

ax.set_xscale("log")
ax.set_xlabel("Exportacions per Càpita (USD, escala log)")
ax.set_ylabel("CO₂ per Càpita (t)")
ax.set_title(f"Exportacions vs CO₂ per Càpita\nSpearman r = {r_exp:.3f}", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.5_exportacions_co2.png"), bbox_inches="tight")
plt.show()
print(f"  Spearman r={r_exp:.3f}")
print(f"   ✅ Gràfic guardat: 3.5_exportacions_co2.png")

# =============================================================================
# 3.6 SALUT — esperança de vida i mortalitat respiratòria vs CO₂
# =============================================================================
print("3.6 Salut vs CO₂...")

d_health = df.dropna(subset=["life_exp", "co2_prod_pc"]).copy()

# --- 3.6a: CO₂ vs Esperança de vida ---
r_life_p, _ = stats.pearsonr(d_health["co2_prod_pc"], d_health["life_exp"])
r_life_s, _ = stats.spearmanr(d_health["co2_prod_pc"], d_health["life_exp"])

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(d_health["co2_prod_pc"], d_health["life_exp"],
                c=d_health["gdp_pc"], cmap="YlOrRd", alpha=0.7,
                s=d_health["population"]/5e6 + 20, edgecolors="gray", linewidth=0.3)
plt.colorbar(sc, ax=ax, label="PIB per Càpita")
ax.set_xlabel("CO₂ per Càpita (t)")
ax.set_ylabel("Esperança de Vida (anys)")
ax.set_title(f"CO₂ per Càpita vs Esperança de Vida\nPearson r = {r_life_p:.3f} | Spearman r = {r_life_s:.3f}",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.6a_co2_esperanca_vida.png"), bbox_inches="tight")
plt.show()
print(f"   ✅ Gràfic guardat: 3.6a_co2_esperanca_vida.png")

# --- 3.6b: CO₂ vs Mortalitat respiratòria ---
d_resp = df.dropna(subset=["resp_death_rate", "co2_prod_pc"]).copy()
r_resp_p, _ = stats.pearsonr(d_resp["co2_prod_pc"], d_resp["resp_death_rate"])
r_resp_s, _ = stats.spearmanr(d_resp["co2_prod_pc"], d_resp["resp_death_rate"])

fig, ax2 = plt.subplots(figsize=(9, 7))
sc2 = ax2.scatter(d_resp["co2_prod_pc"], d_resp["resp_death_rate"],
                  c=d_resp["gdp_pc"], cmap="Blues", alpha=0.7,
                  s=50, edgecolors="gray", linewidth=0.3)
plt.colorbar(sc2, ax=ax2, label="PIB per Càpita")
ax2.set_xlabel("CO₂ per Càpita (t)")
ax2.set_ylabel("Morts per Malalties Respiratòries (per 100k hab.)")
ax2.set_title(f"CO₂ vs Mortalitat Respiratòria\nSpearman r = {r_resp_s:.3f}",
              fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.6b_co2_mortalitat_resp.png"), bbox_inches="tight")
plt.show()
print(f"  Esperança de vida — Pearson r={r_life_p:.3f} | Spearman r={r_life_s:.3f}")
print(f"  Malalties resp.   — Pearson r={r_resp_p:.3f} | Spearman r={r_resp_s:.3f}")
print(f"   ✅ Gràfic guardat: 3.6b_co2_mortalitat_resp.png")

# =============================================================================
# 3.7 HEATMAP GENERAL DE CORRELACIONS
# =============================================================================
print("3.7 Heatmap de correlacions...")

# Dataset per heatmap: variables clau, any de referència
vars_heatmap = ["co2_prod_pc", "gdp_pc", "hdi", "gini", "life_exp",
                "resp_death_rate", "exports_pc", "pct_lowcarbon", "carbon_intensity"]

d_heat = df.copy()
d_heat["exports_pc"] = d_heat["exports_usd"] / d_heat["population"]
d_heat = d_heat[vars_heatmap].dropna()

labels = {
    "co2_prod_pc":      "CO₂ per càpita",
    "gdp_pc":           "PIB per càpita",
    "hdi":              "HDI",
    "gini":             "Gini",
    "life_exp":         "Esperança de vida",
    "resp_death_rate":  "Mort. respiratòria",
    "exports_pc":       "Exportacions pc",
    "pct_lowcarbon":    "% Renovables",
    "carbon_intensity": "Intensitat carboni",
}
d_heat = d_heat.rename(columns=labels)

corr_matrix = d_heat.corr(method="spearman")

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size": 10})
ax.set_title("Matriu de Correlació Spearman — Variables Clau", fontsize=13, fontweight="bold")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "3.7_heatmap_correlacions.png"), bbox_inches="tight")
plt.show()
print(f"   ✅ Gràfic guardat: 3.7_heatmap_correlacions.png")

print("\n✅ Fase 3 completada. Tots els gràfics guardats a /grafics/")
