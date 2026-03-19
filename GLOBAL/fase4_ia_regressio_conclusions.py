"""
REPTE 7 — Emissions de CO₂ Globals
Fase 4: IA & Data Centers + Regressió Múltiple + Conclusions
  4.1 Anàlisi de l'acceleració d'emissions post-2020
  4.2 Data Centers vs emissions per país
  4.3 Impacte estimat de la IA (consum energètic)
  4.4 Model de regressió múltiple
  4.5 Clustering de països per perfil emission/desenvolupament
  4.6 Conclusions narratives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings("ignore")
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
df_dc  = pd.read_csv(os.path.join(BASE, "datacenters_per_pais.csv"))

# =============================================================================
# 4.2 DATA CENTERS vs EMISSIONS PER PAÍS
# =============================================================================
print("4.2 Data Centers vs emissions...")

# Netejar df_dc: extreure valors numèrics
def parse_num(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace("~", "").replace("+", "").replace(",", "").strip()
    try:
        return float(s)
    except:
        return np.nan

df_dc_clean = df_dc[["country", "total_data_centers", "power_capacity_MW_total",
                      "growth_rate_of_data_centers_percent_per_year",
                      "average_renewable_energy_usage_percent"]].copy()
df_dc_clean.columns = ["Entity", "total_dc", "power_MW", "growth_pct", "renew_pct"]
df_dc_clean["total_dc"] = df_dc_clean["total_dc"].apply(parse_num)
df_dc_clean["power_MW"]  = df_dc_clean["power_MW"].apply(parse_num)
df_dc_clean["growth_pct"] = df_dc_clean["growth_pct"].apply(parse_num)
def parse_renew(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%","").replace("~","").replace("+","").strip()
    # Agafa només la primera part numèrica (p.ex. "45 (target 2030)" → "45")
    s = s.split()[0]
    try:
        return float(s)
    except:
        return np.nan

df_dc_clean["renew_pct"] = df_dc_clean["renew_pct"].apply(parse_renew)

# Merge amb emissions recents
co2_recent = master[master["Year"] == 2020][["Entity", "co2_prod", "co2_prod_pc", "gdp_pc"]].copy()
df_dc_merged = df_dc_clean.merge(co2_recent, on="Entity", how="inner")
df_dc_merged = df_dc_merged.dropna(subset=["total_dc", "co2_prod"])

# =============================================================================
# 4.3 IMPACTE ESTIMAT DE LA IA — Exercici de calcul
# =============================================================================
print("4.3 Impacte estimat IA...")

FACTOR_CO2 = 0.475  # kg CO2/kWh (mix global IEA 2023)

# -----------------------------------------------------------------------
# 4.3d: Quins DC contaminen mes? — Stacked bar MW net vs brut + cotxes equiv.
# -----------------------------------------------------------------------
HORES_ANY   = 8760
KG_CO2_KWH  = 0.475
CO2_COTXE   = 4600   # kg CO2/any cotxe mitja (IEA)

dc_bar = df_dc_clean.copy()
dc_bar["power_MW"]  = pd.to_numeric(dc_bar["power_MW"],  errors="coerce")
dc_bar["renew_pct"] = pd.to_numeric(dc_bar["renew_pct"], errors="coerce").fillna(0)
dc_bar = dc_bar.dropna(subset=["power_MW"]).nlargest(15, "power_MW").copy()
dc_bar = dc_bar.sort_values("power_MW")

dc_bar["MW_net"]    = dc_bar["power_MW"] * (dc_bar["renew_pct"] / 100)
dc_bar["MW_brut"]   = dc_bar["power_MW"] * (1 - dc_bar["renew_pct"] / 100)
dc_bar["co2_Mt"]    = dc_bar["MW_brut"] * 1e3 * HORES_ANY * KG_CO2_KWH / 1e12
dc_bar["cotxes_M"]  = dc_bar["MW_brut"] * 1e3 * HORES_ANY * KG_CO2_KWH / CO2_COTXE / 1e6

fig, ax = plt.subplots(figsize=(13, 9))
bars_r = ax.barh(dc_bar["Entity"], dc_bar["MW_net"],
                 color="#4CAF50", label="Energia renovable (MW)")
bars_b = ax.barh(dc_bar["Entity"], dc_bar["MW_brut"],
                 left=dc_bar["MW_net"],
                 color="#d62728", label="Energia no-renovable (MW)")

# % renovable al costat de cada barra (a la dreta)
for bar, (_, row) in zip(bars_r, dc_bar.iterrows()):
    pct = row["renew_pct"]
    total = row["power_MW"]
    ax.text(total + 30, bar.get_y() + bar.get_height() / 2,
            f"{pct:.0f}% renovable", ha="left", va="center",
            fontsize=8.5, color="#2e7d32", fontweight="bold")

# Etiqueta: Mt CO2 + equiv. cotxes (només si > 0.05 Mt)
for _, row in dc_bar.iterrows():
    if row["co2_Mt"] < 0.05:
        continue
    total = row["power_MW"]
    label = f"  {row['co2_Mt']:.1f} Mt CO2/any  (~{row['cotxes_M']:.1f}M cotxes)"
    ax.text(total + 30, dc_bar["Entity"].tolist().index(row["Entity"]) - 0.28,
            label, va="center", fontsize=7.5, color="#999")

ax.set_xlabel("Capacitat Instal.lada (MW)", fontsize=11)
ax.set_title("Contaminacio Real dels Data Centers per Pais (Top 15)\n"
             "Verd = potencia renovable  |  Vermell = potencia bruta",
             fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, dc_bar["power_MW"].max() * 1.75)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "4.3d_dc_contaminacio_real.png"), bbox_inches="tight")
plt.show()
print("   Grafic guardat: 4.3d_dc_contaminacio_real.png")

# =============================================================================
# 4.5 CLUSTERING — Perfils de països per emissions i desenvolupament
# =============================================================================
print("4.5 Clustering de països...")

ANY_MODEL = 2019
cluster_vars = ["co2_prod_pc", "gdp_pc", "hdi"]
df_clust = master[master["Year"] == ANY_MODEL].copy()
df_clust = df_clust.dropna(subset=cluster_vars)

scaler2 = StandardScaler()
X_clust = scaler2.fit_transform(df_clust[cluster_vars])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clust["cluster"] = kmeans.fit_predict(X_clust)

# Etiquetes descriptives per clúster
cluster_means = df_clust.groupby("cluster")[cluster_vars].mean()
print("\n  Perfil mitjà per clúster:")
print(cluster_means.round(2).to_string())

# Assignar noms descriptius manualment basant-nos en els perfils
# Assignem noms per ranking (evita col·lisions amb la mediana)
sorted_hdi = cluster_means["hdi"].sort_values(ascending=False)
rich_clusters = sorted_hdi.index[:2].tolist()
poor_clusters  = sorted_hdi.index[2:].tolist()
rich_co2  = cluster_means.loc[rich_clusters,  "co2_prod_pc"].sort_values()
poor_co2  = cluster_means.loc[poor_clusters,  "co2_prod_pc"].sort_values()
cluster_names = {
    rich_co2.index[0]: "Rics & Nets",
    rich_co2.index[1]: "Rics & Contaminants",
    poor_co2.index[0]: "Pobres & Emissors Baixos",
    poor_co2.index[1]: "En Desenvolupament",
}
print("  Noms assignats:", cluster_names)

df_clust["cluster_nom"] = df_clust["cluster"].map(cluster_names)

palette_clust = {
    "Rics & Contaminants":    "#d62728",
    "Rics & Nets":            "#2ca02c",
    "En Desenvolupament":     "#ff7f0e",
    "Pobres & Emissors Baixos": "#1f77b4",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Esquerra: HDI vs CO₂ per càpita, color = clúster
ax = axes[0]
for nom, grup in df_clust.groupby("cluster_nom"):
    ax.scatter(grup["hdi"], grup["co2_prod_pc"],
               label=nom, alpha=0.75, s=60,
               color=palette_clust.get(nom, "gray"), edgecolors="none")

# Etiquetes només per a països seleccionats
ETIQUETES = [
    # Rics & Nets
    "Sweden", "France", "Norway",
    # Rics & Contaminants
    "Qatar", "Kuwait",
    # Pobres & Emissors Baixos
    "Democratic Republic of Congo", "Ethiopia", "Bangladesh",
]
# Offsets personalitzats per evitar solapaments
OFFSETS = {
    "Democratic Republic of Congo": (-20, 18),
    "Ethiopia":                     (-45, -12),
    "Bangladesh":                   (-10,  -14),
    "Sweden":                       (4,    -4),
    "France":                       (4,    6),
    "Norway":                       (4,    4),
    "Qatar":                        (4,    4),
    "Kuwait":                       (4,    4),
}
LABELS_DISPLAY = {
    "Democratic Republic of Congo": "Democratic Republic\nof Congo",
}
for _, row in df_clust[df_clust["Entity"].isin(ETIQUETES)].iterrows():
    xoff, yoff = OFFSETS.get(row["Entity"], (4, 4))
    label = LABELS_DISPLAY.get(row["Entity"], row["Entity"])
    ax.annotate(label, (row["hdi"], row["co2_prod_pc"]),
                fontsize=8, fontweight="bold",
                xytext=(xoff, yoff), textcoords="offset points")

ax.set_xlabel("HDI")
ax.set_ylabel("CO₂ per Càpita (t)")
ax.set_title("Clustering de Països per Perfil\nDesenvolupament vs Emissions", fontweight="bold")
ax.legend(fontsize=9)

# Dreta: Recompte de països per clúster + mitjana CO₂
ax2 = axes[1]
resum = df_clust.groupby("cluster_nom").agg(
    n_paisos=("Entity", "count"),
    co2_mitja=("co2_prod_pc", "mean"),
    hdi_mitja=("hdi", "mean"),
).reset_index()

bars = ax2.bar(resum["cluster_nom"], resum["co2_mitja"],
               color=[palette_clust.get(n, "gray") for n in resum["cluster_nom"]],
               edgecolor="white")
for bar, n in zip(bars, resum["n_paisos"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"n={n}", ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("CO₂ per Càpita Mitjana (t)")
ax2.set_title("CO₂ Mitjana per Clúster\n(n = nº de països)", fontweight="bold")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "4.5_clustering_paisos.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 4.5_clustering_paisos.png")

# =============================================================================
# 4.6 CONCLUSIONS NARRATIVES — Resum imprès
# =============================================================================
print("\n" + "="*70)
print("               CONCLUSIONS PRINCIPALS DEL PROJECTE")
print("="*70)

conclusions = [
    ("1. Les emissions globals NO s'han aturat",
     "Les emissions han crescut de ~6 Gt el 1950 a +37 Gt el 2023.\n"
     "   Tot i el COVID (-6% el 2020), el rebrot del 2021 va superar\n"
     "   els nivells anteriors en menys de 2 anys."),

    ("2. Concentració extrema: 3 països = +50% de les emissions",
     "Xina (~30%), EUA (~14%) i India (~7%) dominen les emissions actuals.\n"
     "   Però en termes acumulats històrics, EUA i Europa lideren clarament."),

    ("3. Els rics contaminen molt, però externalitzen",
     "Spearman r=0.895 entre PIB i CO₂ per càpita, però producció vs consum\n"
     "   revela que EUA, Alemanya i França 'importen' CO₂ via productes.\n"
     "   Xina i India carreguen amb les emissions dels béns que produeixen pels rics."),

    ("4. Renovables: necessàries però insuficients per si soles",
     "La correlació renovables-emissions és r=-0.214, relativament feble.\n"
     "   El motiu: molts països redueixen intensitat però augmenten consum total.\n"
     "   Paradox: Alemanya té el 40% de renovables però emetia 9t CO₂/càpita."),

    ("5. Desigualtat interna (Gini) correlació negativa amb emissions",
     "Països molt desiguals (alta Gini) tendeixen a tenir MENYS emissions per càpita.\n"
     "   Per què? La majoria de la població pobra consumeix molt poc.\n"
     "   Consisteix la desigualtat en que els rics contaminen però és la\n"
     "   massa empobri­da la que 'baixa' la mitjana nacional."),

    ("6. La salut: una paradoxa amb el CO₂",
     "r=+0.750 entre CO₂ per càpita i esperança de vida.\n"
     "   Els països rics contaminen molt però viuen més (accés a sanitat).\n"
     "   La mortalitat respiratòria és major als països pobres, malgrat emetre\n"
     "   menys CO₂, per altres factors (biomassa, contaminació local, NO₂)."),

    ("7. Les exportacions expliquen bona part de les emissions",
     "Spearman r=0.843 entre exportacions per càpita i CO₂.\n"
     "   Les economies orientades a l'exportació manufactures emeten molt més.\n"
     "   Cas extrem: Qatar i Kuwait — riquesa basada en exportació d'hidrocarburs,\n"
     "   emissions per càpita de 30–50 tones (vs 6t de mitjana global)."),

    ("8. Data Centers i IA: el nou front invisible",
     "Els EUA concentren >5.400 data centers i ~12.000 MW de capacitat.\n"
     "   L'entrenament d'un model com GPT-4 emet ~25.000–50.000 t CO₂.\n"
     "   Equivalent a ~5.000–10.000 vols BCN-NYC.\n"
     "   Les emissions dels data centers globals (~200 Mt/any) creixien al 12%\n"
     "   anual pre-2023, accelerant-se amb l'expansió de la IA generativa."),

    ("9. El model de regressió múltiple confirma: la població i el PIB ho expliquen tot",
     "GDP, HDI, renovables, exportacions i població expliquen les emissions.\n"
     "   La variable amb més pes: POBLACIÓ i PIB per càpita.\n"
     "   Les renovables tenen coeficient negatiu (redueixen emissions) però\n"
     "   l'efecte queda ofuscat per la mida econòmica del país."),

    ("10. Conclusió global: és possible desacoblar creixement i emissions",
     "Suècia, França, Suïssa i Costa Rica demostren que és possible tenir\n"
     "   HDI alt (>0.9) amb emissions per càpita moderades (<5t).\n"
     "   L'eina: electrificació neta (nuclear + renovables) + eficiència energètica.\n"
     "   El principal obstacle: voluntat política i desigualtat global en l'accés\n"
     "   a tecnologies netes."),
]

for titol, text in conclusions:
    print(f"\n{'─'*70}")
    print(f"  ★ {titol}")
    print(f"    {text}")

print(f"\n{'='*70}")
print("  Gràfics generats a la carpeta /grafics/")
print(f"  Total gràfics: {len([f for f in os.listdir(OUTPUT) if f.endswith('.png')])}")
print("="*70)

print("\n✅ Fase 4 completada. Projecte finalitzat!")
