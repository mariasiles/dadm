import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. CARGAR Y UNIR DATOS
# ──────────────────────────────────────────────
df_vida = pd.read_csv('life-expectancy.csv')
df_co2  = pd.read_csv('annual-co2-emissions-per-country.csv')
df_gdp  = pd.read_csv('gdp-per-capita-worldbank.csv')[['Entity','Year','GDP per capita']]
df_pop  = pd.read_csv('population.csv').rename(columns={'all years': 'Population'})

# Unir todo por país y año
df = df_vida.merge(df_co2[['Entity','Year','Annual CO₂ emissions']], on=['Entity','Year'], how='inner')
df = df.merge(df_gdp, on=['Entity','Year'], how='left')
df = df.merge(df_pop[['Entity','Year','Population']], on=['Entity','Year'], how='left')

# Eliminar agregados mundiales/regionales
excluir = ['World','Asia','Europe','Africa','Americas','Oceania',
           'North America','South America','European Union (27)',
           'High-income countries','Low-income countries',
           'Upper-middle-income countries','Lower-middle-income countries']
df = df[~df['Entity'].isin(excluir)]

# Crear variable de CO₂ per cápita (más representativa que total)
df['CO2_per_capita'] = df['Annual CO₂ emissions'] / df['Population'].replace(0, np.nan)

# Quedarse con filas completas
features = ['CO2_per_capita', 'GDP per capita']
target   = 'Life expectancy'
df_model = df.dropna(subset=features + [target])

print(f"Registros disponibles: {len(df_model):,}  |  Países: {df_model['Entity'].nunique()}")

# ──────────────────────────────────────────────
# 2. MODELO: esperanza de vida ~ CO₂ per cápita + PIB
# ──────────────────────────────────────────────
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nModelo Gradient Boosting:")
print(f"  R²:   {r2:.3f}")
print(f"  RMSE: {rmse:.2f} años")

# ──────────────────────────────────────────────
# 3. PROYECTAR CO₂ per cápita Y PIB FUTURO (regresión temporal)
# ──────────────────────────────────────────────
years_futuro = list(range(2026, 2036))

def proyectar_serie(df_pais, col, years_fut, min_obs=5):
    """Regresión lineal sobre los últimos 15 años de la serie."""
    serie = df_pais[['Year', col]].dropna().sort_values('Year').tail(15)
    if len(serie) < min_obs:
        return None
    reg = LinearRegression().fit(serie[['Year']], serie[col])
    vals = reg.predict(pd.DataFrame({'Year': years_fut}))
    return vals

predicciones = []

for pais in df_model['Entity'].unique():
    dp = df_model[df_model['Entity'] == pais].sort_values('Year')

    co2_fut = proyectar_serie(dp, 'CO2_per_capita', years_futuro)
    gdp_fut = proyectar_serie(dp, 'GDP per capita', years_futuro)

    if co2_fut is None or gdp_fut is None:
        continue

    # Asegurar valores no negativos
    co2_fut = np.maximum(co2_fut, 0)
    gdp_fut = np.maximum(gdp_fut, 0)

    X_fut = scaler.transform(np.column_stack([co2_fut, gdp_fut]))
    vida_pred = model.predict(X_fut)
    vida_pred = np.clip(vida_pred, 40, 95)  # límites biológicos razonables

    for i, yr in enumerate(years_futuro):
        predicciones.append({
            'Entity': pais,
            'Year': yr,
            'Predicted Life Expectancy': round(vida_pred[i], 2),
            'Projected CO2 per capita': round(co2_fut[i], 4),
            'Projected GDP per capita': round(gdp_fut[i], 2)
        })

df_pred = pd.DataFrame(predicciones)
print(f"\nPredicciones generadas para {df_pred['Entity'].nunique()} países")

# ──────────────────────────────────────────────
# 4. GUARDAR RESULTADOS
# ──────────────────────────────────────────────
df_pred.to_csv('prediccion_esperanza_vida_co2.csv', index=False)
print("Guardado: prediccion_esperanza_vida_co2.csv")

# ──────────────────────────────────────────────
# 5. GRÁFICOS
# ──────────────────────────────────────────────

paises_clave = ['China', 'India', 'United States', 'Germany',
                'Brazil', 'Japan', 'Spain', 'Russia', 'Nigeria']

# --- Gráfico 1: Esperanza de vida histórica + predicción por país ---
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, pais in enumerate(paises_clave):
    hist = df_vida[df_vida['Entity'] == pais][['Year','Life expectancy']].dropna().sort_values('Year').tail(40)
    pred = df_pred[df_pred['Entity'] == pais].sort_values('Year')

    if hist.empty or pred.empty:
        continue

    axes[i].plot(hist['Year'], hist['Life expectancy'],
                 color='steelblue', linewidth=2, label='Histórico')
    axes[i].plot(pred['Year'], pred['Predicted Life Expectancy'],
                 color='tomato', linewidth=2.5, linestyle='--',
                 marker='o', markersize=4, label='Predicción')
    # Conectar
    axes[i].plot([hist['Year'].iloc[-1], pred['Year'].iloc[0]],
                 [hist['Life expectancy'].iloc[-1], pred['Predicted Life Expectancy'].iloc[0]],
                 color='gray', linewidth=1, linestyle='--')
    axes[i].axvline(x=2025.5, color='gray', linestyle=':', linewidth=1)
    axes[i].set_title(pais, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Años', fontsize=8)
    axes[i].set_xlabel('Año', fontsize=8)
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(bottom=max(0, hist['Life expectancy'].min() - 5))

plt.suptitle('Predicción de Esperanza de Vida (2026-2035)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('grafics/pred_esperanza_vida_paises.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafics/pred_esperanza_vida_paises.png")

# --- Gráfico 2: Dispersión CO₂ per cápita vs Esperanza de vida (año reciente) ---
ultimo_anio = df_model['Year'].max()
df_reciente = df_model[df_model['Year'] == ultimo_anio].copy()

fig2, ax2 = plt.subplots(figsize=(11, 7))
sc = ax2.scatter(df_reciente['CO2_per_capita'],
                 df_reciente['Life expectancy'],
                 c=df_reciente['GDP per capita'],
                 cmap='RdYlGn', s=60, alpha=0.7, edgecolors='gray', linewidths=0.3)
cbar = plt.colorbar(sc, ax=ax2)
cbar.set_label('PIB per cápita (USD)', fontsize=10)

# Etiquetar países clave
for _, row in df_reciente[df_reciente['Entity'].isin(paises_clave)].iterrows():
    ax2.annotate(row['Entity'], (row['CO2_per_capita'], row['Life expectancy']),
                 fontsize=7, ha='left', va='bottom',
                 xytext=(4, 4), textcoords='offset points')

# Línea de tendencia
z = np.polyfit(df_reciente['CO2_per_capita'].dropna(),
               df_reciente.loc[df_reciente['CO2_per_capita'].notna(), 'Life expectancy'], 2)
p = np.poly1d(z)
x_line = np.linspace(df_reciente['CO2_per_capita'].min(), df_reciente['CO2_per_capita'].max(), 200)
ax2.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.6, label='Tendencia')

ax2.set_xlabel('CO₂ per cápita (toneladas)', fontsize=12)
ax2.set_ylabel('Esperanza de vida (años)', fontsize=12)
ax2.set_title(f'CO₂ per cápita vs Esperanza de vida ({ultimo_anio})\n(color = PIB per cápita)',
              fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafics/scatter_co2_vida.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafics/scatter_co2_vida.png")

# --- Gráfico 3: Cambio previsto esperanza de vida 2025→2035 ---
vida_2026 = df_pred[df_pred['Year'] == 2026][['Entity','Predicted Life Expectancy']].rename(
    columns={'Predicted Life Expectancy': 'vida_2026'})
vida_2035 = df_pred[df_pred['Year'] == 2035][['Entity','Predicted Life Expectancy']].rename(
    columns={'Predicted Life Expectancy': 'vida_2035'})
df_cambio = vida_2026.merge(vida_2035, on='Entity')
df_cambio['cambio'] = df_cambio['vida_2035'] - df_cambio['vida_2026']

top_mejora   = df_cambio.nlargest(10, 'cambio')
top_deterioro = df_cambio.nsmallest(10, 'cambio')
df_barras = pd.concat([top_mejora, top_deterioro]).sort_values('cambio')

fig3, ax3 = plt.subplots(figsize=(10, 8))
colors = ['tomato' if v < 0 else 'seagreen' for v in df_barras['cambio']]
ax3.barh(df_barras['Entity'], df_barras['cambio'], color=colors, edgecolor='white')
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_xlabel('Cambio en esperanza de vida (años, 2026→2035)', fontsize=11)
ax3.set_title('Países con mayor mejora y deterioro previsto\nen esperanza de vida (2026-2035)',
              fontsize=13, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('grafics/cambio_esperanza_vida.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado: grafics/cambio_esperanza_vida.png")

# ──────────────────────────────────────────────
# 6. RESUMEN EN CONSOLA
# ──────────────────────────────────────────────
print("\n── Países clave: esperanza de vida predicha en 2035 ──")
resumen = df_pred[df_pred['Year'] == 2035][df_pred['Entity'].isin(paises_clave)][
    ['Entity','Predicted Life Expectancy','Projected CO2 per capita']].sort_values(
    'Predicted Life Expectancy', ascending=False)
print(resumen.to_string(index=False))

print("\n── Top 5 países con mayor mejora en esperanza de vida (2026→2035) ──")
print(df_cambio.nlargest(5, 'cambio')[['Entity','cambio']].to_string(index=False))

print("\n── Top 5 países con mayor deterioro esperado ──")
print(df_cambio.nsmallest(5, 'cambio')[['Entity','cambio']].to_string(index=False))
