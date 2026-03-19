"""
REPTE 7 — Emissions de CO₂ Globals
Fase 1: Càrrega i preprocessament de dades
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 1. CÀRREGA DE DATASETS
# =============================================================================

print("Carregant datasets...")

df_co2_prod  = pd.read_csv(os.path.join(BASE, "annual-co2-emissions-per-country.csv"))
df_co2_cons  = pd.read_csv(os.path.join(BASE, "consumption-co2-emissions.csv"))
df_sectors   = pd.read_csv(os.path.join(BASE, "ghg-emissions-by-sector.csv"))
df_pop       = pd.read_csv(os.path.join(BASE, "population.csv"))
df_gdp       = pd.read_csv(os.path.join(BASE, "gdp-per-capita-worldbank.csv"))
df_hdi       = pd.read_csv(os.path.join(BASE, "human-development-index.csv"))
df_gini      = pd.read_csv(os.path.join(BASE, "economic-inequality-gini-index.csv"))
df_life      = pd.read_csv(os.path.join(BASE, "life-expectancy.csv"))
df_resp      = pd.read_csv(os.path.join(BASE, "chronic-respiratory-diseases-death-rate-who-mdb.csv"))
df_exports   = pd.read_csv(os.path.join(BASE, "value-of-exported-goods-and-services.csv"))
df_energy    = pd.read_csv(os.path.join(BASE, "primary-energy-cons.csv"))
df_renew     = pd.read_csv(os.path.join(BASE, "share-electricity-low-carbon.csv"))
df_forest    = pd.read_csv(os.path.join(BASE, "annual-change-forest-area.csv"))
df_temp      = pd.read_csv(os.path.join(BASE, "monthly-average-surface-temperatures-by-year.csv"))
df_dc        = pd.read_csv(os.path.join(BASE, "datacenters_per_pais.csv"))

print("✅ Tots els datasets carregats\n")

# Resum inicial
datasets = {
    "CO₂ Producció":        df_co2_prod,
    "CO₂ Consum":           df_co2_cons,
    "GHG per Sector":       df_sectors,
    "Població":             df_pop,
    "GDP per Càpita":       df_gdp,
    "HDI":                  df_hdi,
    "Gini":                 df_gini,
    "Esperança de Vida":    df_life,
    "Malalties Resp.":      df_resp,
    "Exportacions":         df_exports,
    "Energia Primària":     df_energy,
    "Electricitat Neta":    df_renew,
    "Canvi Forestal":       df_forest,
    "Temperatures":         df_temp,
    "Data Centers":         df_dc,
}

print(f"{'Dataset':<25} {'Registres':>10} {'Columnes':>10}")
print("-" * 47)
for name, df in datasets.items():
    print(f"{name:<25} {df.shape[0]:>10,} {df.shape[1]:>10}")

# =============================================================================
# 2. NORMALITZACIÓ DE COLUMNES
# =============================================================================

print("\nNormalitzant columnes...")

# Renombrar columnes per coherència
df_co2_prod.columns  = ["Entity", "Code", "Year", "co2_prod"]
df_co2_cons.columns  = ["Entity", "Code", "Year", "co2_cons"]
df_pop.columns       = ["Entity", "Code", "Year", "population"]
df_gdp               = df_gdp.rename(columns={"GDP per capita": "gdp_pc"})
df_hdi               = df_hdi.rename(columns={"Human Development Index": "hdi"})
df_gini              = df_gini.rename(columns={"Gini coefficient": "gini"})
df_life              = df_life.rename(columns={"Life expectancy": "life_exp"})
df_resp.columns      = ["Entity", "Code", "Year", "resp_death_rate"]
df_exports           = df_exports.rename(columns={df_exports.columns[3]: "exports_usd"})
df_energy            = df_energy.rename(columns={"Primary energy consumption": "energy_twh"})
df_renew             = df_renew.rename(columns={"Share of electricity from low-carbon sources": "pct_lowcarbon"})
df_forest            = df_forest.rename(columns={"Annual change in forest area": "forest_change_ha"})

print("✅ Columnes normalitzades\n")

# =============================================================================
# 3. FILTRE: NOMÉS PAÏSOS REALS (eliminar regions agregades)
# =============================================================================

# Els codis ISO de 3 lletres identifiquen països reals
def keep_countries(df):
    if "Code" in df.columns:
        return df[df["Code"].notna() & (df["Code"].str.len() == 3)].copy()
    return df

df_co2_prod  = keep_countries(df_co2_prod)
df_co2_cons  = keep_countries(df_co2_cons)
df_sectors   = keep_countries(df_sectors)
df_pop       = keep_countries(df_pop)
df_gdp       = keep_countries(df_gdp)
df_hdi       = keep_countries(df_hdi)
df_gini      = keep_countries(df_gini)
df_life      = keep_countries(df_life)
df_resp      = keep_countries(df_resp)
df_exports   = keep_countries(df_exports)
df_energy    = keep_countries(df_energy)
df_renew     = keep_countries(df_renew)
df_forest    = keep_countries(df_forest)

print("✅ Regions agregades eliminades\n")

# =============================================================================
# 4. CONSTRUCCIÓ DEL DATASET MASTER (merge per Entity + Year)
# =============================================================================

print("Construint dataset master...")

# Base: CO₂ producció
master = df_co2_prod[["Entity", "Code", "Year", "co2_prod"]].copy()

# Merge seqüencial
merges = [
    (df_co2_cons[["Entity", "Code", "Year", "co2_cons"]],       ["Entity", "Code", "Year"]),
    (df_pop[["Entity", "Code", "Year", "population"]],           ["Entity", "Code", "Year"]),
    (df_gdp[["Entity", "Code", "Year", "gdp_pc"]],               ["Entity", "Code", "Year"]),
    (df_hdi[["Entity", "Code", "Year", "hdi"]],                  ["Entity", "Code", "Year"]),
    (df_gini[["Entity", "Code", "Year", "gini"]],                ["Entity", "Code", "Year"]),
    (df_life[["Entity", "Code", "Year", "life_exp"]],            ["Entity", "Code", "Year"]),
    (df_resp[["Entity", "Code", "Year", "resp_death_rate"]],     ["Entity", "Code", "Year"]),
    (df_exports[["Entity", "Code", "Year", "exports_usd"]],      ["Entity", "Code", "Year"]),
    (df_energy[["Entity", "Code", "Year", "energy_twh"]],        ["Entity", "Code", "Year"]),
    (df_renew[["Entity", "Code", "Year", "pct_lowcarbon"]],      ["Entity", "Code", "Year"]),
    (df_forest[["Entity", "Code", "Year", "forest_change_ha"]], ["Entity", "Code", "Year"]),
]

for df_merge, keys in merges:
    master = master.merge(df_merge, on=keys, how="left")

print(f"  Shape master: {master.shape}")
print(f"  Rang temporal: {master['Year'].min()} – {master['Year'].max()}")
print(f"  Països únics: {master['Entity'].nunique()}")

# =============================================================================
# 5. VARIABLES DERIVADES
# =============================================================================

print("\nCalculant variables derivades...")

# CO₂ per càpita (tones)
master["co2_prod_pc"] = master["co2_prod"] / master["population"]

# CO₂ consumit per càpita
master["co2_cons_pc"] = master["co2_cons"] / master["population"]

# Balanç comercial de CO₂ (positiu = exporta emissions, negatiu = importa)
master["co2_trade_balance"] = master["co2_prod"] - master["co2_cons"]

# Intensitat de carboni (CO₂ per unitat d'energia)
master["carbon_intensity"] = master["co2_prod"] / (master["energy_twh"] * 1e6)  # tones/TWh → kg/MWh

# Exportacions com % del PIB (proxy d'obertura econòmica)
master["exports_share_gdp"] = master["exports_usd"] / (master["gdp_pc"] * master["population"])

print("✅ Variables derivades calculades\n")

# =============================================================================
# 6. INFORME DE VALORS NULS
# =============================================================================

print("=== Valors nuls per columna (%) ===")
nulls = (master.isnull().sum() / len(master) * 100).round(1)
for col, pct in nulls[nulls > 0].items():
    print(f"  {col:<25} {pct:>6.1f}%")

# =============================================================================
# 7. GUARDAR DATASET MASTER
# =============================================================================

out_path = os.path.join(BASE, "master_dataset.csv")
master.to_csv(out_path, index=False)
print(f"\n✅ Dataset master guardat a: master_dataset.csv")
print(f"   {master.shape[0]:,} files × {master.shape[1]} columnes")
