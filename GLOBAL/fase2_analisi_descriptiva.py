"""
REPTE 7 — Emissions de CO₂ Globals
Fase 2: Anàlisi Descriptiva
  2.1 Evolució global de CO₂ al llarg del temps
  2.2 Top emissors per país (ranking + mapa)
  2.3 CO₂ per sector — qui és el culpable real?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

BASE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
})

# Carregar master i sectors
master = pd.read_csv(os.path.join(BASE, "master_dataset.csv"))
df_sectors = pd.read_csv(os.path.join(BASE, "ghg-emissions-by-sector.csv"))

# Netejar sectors: només països reals
df_sectors = df_sectors[df_sectors["Code"].notna() & (df_sectors["Code"].str.len() == 3)].copy()

OUTPUT = os.path.join(BASE, "grafics")
os.makedirs(OUTPUT, exist_ok=True)

# =============================================================================
# 2.1 EVOLUCIÓ GLOBAL DE CO₂ AL LLARG DEL TEMPS
# =============================================================================

print("2.1 Evolució global...")

# CO₂ global total per any (excloem registres sense codi = agreg.)
co2_global = (
    master.groupby("Year")["co2_prod"]
    .sum()
    .reset_index()
    .rename(columns={"co2_prod": "co2_total"})
)
co2_global = co2_global[co2_global["Year"] >= 1850]
co2_global["co2_Gt"] = co2_global["co2_total"] / 1e9  # passa a gigatones

fig, ax = plt.subplots(figsize=(13, 6))

# Àrea de fons per dècades clau
ax.axvspan(1939, 1945, alpha=0.12, color="red",   label="2a Guerra Mundial")
ax.axvspan(1973, 1974, alpha=0.12, color="orange", label="Crisis petroli 1973")
ax.axvspan(2008, 2009, alpha=0.12, color="purple", label="Crisis financera 2008")
ax.axvspan(2020, 2021, alpha=0.12, color="green",  label="COVID-19")

ax.plot(co2_global["Year"], co2_global["co2_Gt"], color="#d62728", linewidth=2.5, zorder=5)
ax.fill_between(co2_global["Year"], co2_global["co2_Gt"], alpha=0.15, color="#d62728")

ax.set_title("Evolució de les Emissions Globals de CO₂ (1850–2025)", fontsize=14, fontweight="bold")
ax.set_xlabel("Any")
ax.set_ylabel("Gigatones de CO₂ (Gt)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f} Gt"))
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.1_evolucio_global_co2.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.1_evolucio_global_co2.png")

# =============================================================================
# 2.2 TOP EMISSORS PER PAÍS
# =============================================================================

print("2.2 Top emissors...")

# --- 2.2a Top 15 per emissions totals acumulades (tota la història) ---
top_hist = (
    master.groupby("Entity")["co2_prod"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
top_hist["co2_Gt"] = top_hist["co2_prod"] / 1e9

fig, ax = plt.subplots(figsize=(12, 7))
colors = ["#d62728" if e in ["United States", "China"] else "#1f77b4" for e in top_hist["Entity"]]
bars = ax.barh(top_hist["Entity"][::-1], top_hist["co2_Gt"][::-1], color=colors[::-1], edgecolor="white")
for bar, val in zip(bars, top_hist["co2_Gt"][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.0f} Gt", va="center", fontsize=9)
ax.set_title("Top 15 Països per Emissions Acumulades Històriques de CO₂", fontsize=13, fontweight="bold")
ax.set_xlabel("Gigatones de CO₂ acumulades")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.2a_top_emissors_historics.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.2a_top_emissors_historics.png")

# --- 2.2b Top 15 emissors ACTUALS (any més recent disponible) ---
any_recent = master[master["co2_prod"].notna()]["Year"].max()
top_actual = (
    master[master["Year"] == any_recent]
    .nlargest(15, "co2_prod")[["Entity", "co2_prod"]]
    .reset_index(drop=True)
)
top_actual["co2_Gt"] = top_actual["co2_prod"] / 1e9

fig, ax = plt.subplots(figsize=(12, 7))
colors2 = ["#d62728" if e == "China" else "#ff7f0e" if e == "United States" else "#2ca02c"
           for e in top_actual["Entity"]]
bars2 = ax.barh(top_actual["Entity"][::-1], top_actual["co2_Gt"][::-1],
                color=colors2[::-1], edgecolor="white")
for bar, val in zip(bars2, top_actual["co2_Gt"][::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.2f} Gt", va="center", fontsize=9)
ax.set_title("Top 15 Països per Emissions Anuals de CO₂", fontsize=13, fontweight="bold")
ax.set_xlabel("Gigatones de CO₂")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.2b_top_emissors_actuals.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.2b_top_emissors_actuals.png")

# --- 2.2c Evolució dels Top 8 emissors actuals al llarg del temps ---
top8 = top_actual["Entity"].head(8).tolist()
df_top8 = master[master["Entity"].isin(top8)].copy()
df_top8["co2_Gt"] = df_top8["co2_prod"] / 1e9
df_top8 = df_top8[df_top8["Year"] >= 1950]

fig, ax = plt.subplots(figsize=(13, 7))
for pais in top8:
    data = df_top8[df_top8["Entity"] == pais].sort_values("Year")
    ax.plot(data["Year"], data["co2_Gt"], linewidth=2, label=pais)

ax.set_title("Evolució Emissions CO₂ — Top 8 Emissors Actuals (1950–2024)", fontsize=13, fontweight="bold")
ax.set_xlabel("Any")
ax.set_ylabel("Gigatones de CO₂")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.2c_evolucio_top8.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.2c_evolucio_top8.png")

# --- 2.2d CO₂ per càpita: qui contamina realment més per persona? ---
any_pc = master.dropna(subset=["co2_prod_pc"])["Year"].max()
top_pc = (
    master[master["Year"] == any_pc]
    .dropna(subset=["co2_prod_pc"])
    .nlargest(20, "co2_prod_pc")[["Entity", "co2_prod_pc"]]
)
top_pc["co2_prod_pc_t"] = top_pc["co2_prod_pc"]  # ja en tones

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top_pc["Entity"][::-1], top_pc["co2_prod_pc_t"][::-1],
        color="#17becf", edgecolor="white")
ax.axvline(x=master[master["Year"]==any_pc]["co2_prod_pc"].mean(),
           color="red", linestyle="--", label="Mitjana mundial")
ax.set_title("CO₂ per Càpita — Top 20 Països", fontsize=13, fontweight="bold")
ax.set_xlabel("Tones de CO₂ per persona")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.2d_co2_percapita_top20.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.2d_co2_percapita_top20.png")

# =============================================================================
# 2.3 EMISSIONS PER SECTOR
# =============================================================================

print("2.3 Emissions per sector...")

sector_cols = ["Agriculture", "Land-use change and forestry", "Waste",
               "Buildings", "Industry", "Manufacturing and construction",
               "Transport", "Electricity and heat", "Fugitive emissions",
               "Aviation and shipping"]

# Globals per any
df_sec_global = (
    df_sectors.groupby("Year")[sector_cols]
    .sum()
    .reset_index()
)
df_sec_global = df_sec_global[df_sec_global["Year"] >= 1990]

# Convertir a Gt
for col in sector_cols:
    df_sec_global[col] = df_sec_global[col] / 1e9

# --- 2.3b Distribució per sector (any recent) — Pastís ---
any_sec = df_sectors[~df_sectors[sector_cols[0]].isna()]["Year"].max()
totals_sector = df_sectors[df_sectors["Year"] == any_sec][sector_cols].sum()
totals_sector = totals_sector[totals_sector > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 9))
wedges, texts, autotexts = ax.pie(
    totals_sector,
    labels=totals_sector.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("tab10", len(totals_sector)),
    startangle=140,
    pctdistance=0.8,
)
for t in autotexts:
    t.set_fontsize(9)
ax.set_title("Distribució Global d'Emissions de CO₂ per Sector", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.3b_sectors_pastis.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.3b_sectors_pastis.png")

# --- 2.3c Evolució dels 4 sectors principals ---
top4_sectors = totals_sector.head(4).index.tolist()
palette = sns.color_palette("tab10", 4)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
for i, sec in enumerate(top4_sectors):
    data = df_sec_global[["Year", sec]].dropna()
    axes[i].plot(data["Year"], data[sec], color=palette[i], linewidth=2.5)
    axes[i].fill_between(data["Year"], data[sec], alpha=0.15, color=palette[i])
    axes[i].set_title(sec, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Gt CO₂eq")
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
fig.suptitle("Evolució dels 4 Sectors amb Més Emissions de CO₂ (1990–2025)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "2.3c_top4_sectors_evolucio.png"), bbox_inches="tight")
plt.show()
print("   ✅ Gràfic guardat: 2.3c_top4_sectors_evolucio.png")

print("\n✅ Fase 2 completada. Tots els gràfics guardats a /grafics/")
