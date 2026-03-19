import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. CARGAR DATOS
# ──────────────────────────────────────────────
df_co2 = pd.read_csv('annual-co2-emissions-per-country.csv')
target_col = 'Annual CO₂ emissions'

# Eliminar filas de agregados mundiales/regionales
excluir = ['World', 'Asia', 'Europe', 'Africa', 'Americas', 'Oceania',
           'North America', 'South America', 'European Union (27)',
           'High-income countries', 'Low-income countries',
           'Upper-middle-income countries', 'Lower-middle-income countries']
df_co2 = df_co2[~df_co2['Entity'].isin(excluir)]

# ──────────────────────────────────────────────
# 2. PREDICCIÓN POR PAÍS (regresión lineal sobre serie temporal)
# ──────────────────────────────────────────────
years_futuro = list(range(2026, 2036))  # 2026 - 2035
predicciones = []
metricas = []

paises = df_co2['Entity'].unique()

for pais in paises:
    df_pais = df_co2[df_co2['Entity'] == pais][['Year', target_col]].dropna().sort_values('Year')
    
    # Necesitamos al menos 5 años de datos para una predicción fiable
    if df_pais.shape[0] < 5:
        continue

    # Usar los últimos 20 años disponibles para capturar tendencia reciente
    df_pais = df_pais.tail(20)

    X = df_pais['Year'].values.reshape(-1, 1)
    y = df_pais[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Métricas sobre datos históricos
    y_pred_hist = model.predict(X)
    r2 = r2_score(y, y_pred_hist)
    rmse = np.sqrt(mean_squared_error(y, y_pred_hist))
    metricas.append({'Entity': pais, 'R2': round(r2, 3), 'RMSE': round(rmse, 0),
                     'Tendencia (t/año)': round(model.coef_[0], 0)})

    # Predicción futura
    for year in years_futuro:
        pred = model.predict([[year]])[0]
        pred = max(pred, 0)  # emisiones no pueden ser negativas
        predicciones.append({
            'Entity': pais,
            'Year': year,
            'Predicted CO₂ emissions': round(pred, 2)
        })

df_pred = pd.DataFrame(predicciones)
df_metricas = pd.DataFrame(metricas)

print(f"Países con predicción: {df_pred['Entity'].nunique()}")
print(f"\nMétricas promedio del modelo:")
print(f"  R² medio:  {df_metricas['R2'].mean():.3f}")
print(f"  RMSE medio: {df_metricas['RMSE'].mean():,.0f} toneladas")

# ──────────────────────────────────────────────
# 3. GUARDAR RESULTADOS
# ──────────────────────────────────────────────
df_pred.to_csv('prediccion_co2_futuro.csv', index=False)
df_metricas.to_csv('metricas_modelo_co2.csv', index=False)
print("\nPredicciones guardadas en prediccion_co2_futuro.csv")
print("Métricas guardadas en metricas_modelo_co2.csv")

# ──────────────────────────────────────────────
# 4. GRÁFICOS
# ──────────────────────────────────────────────

# --- Gráfico 1: Top 10 países más emisores - predicción futura ---
top10 = (df_pred.groupby('Entity')['Predicted CO₂ emissions']
         .mean().nlargest(10).index.tolist())

df_top10_hist = df_co2[df_co2['Entity'].isin(top10)][['Entity','Year', target_col]].dropna()
df_top10_pred = df_pred[df_pred['Entity'].isin(top10)]

fig, ax = plt.subplots(figsize=(12, 6))
colors = cm.tab10(np.linspace(0, 1, 10))

for i, pais in enumerate(top10):
    hist = df_top10_hist[df_top10_hist['Entity'] == pais].sort_values('Year').tail(30)
    pred = df_top10_pred[df_top10_pred['Entity'] == pais].sort_values('Year')
    ax.plot(hist['Year'], hist[target_col], color=colors[i], linewidth=1.5)
    ax.plot(pred['Year'], pred['Predicted CO₂ emissions'], color=colors[i],
            linewidth=2, linestyle='--', label=pais)
    # Conectar histórico con predicción
    if not hist.empty and not pred.empty:
        ax.plot([hist['Year'].iloc[-1], pred['Year'].iloc[0]],
                [hist[target_col].iloc[-1], pred['Predicted CO₂ emissions'].iloc[0]],
                color=colors[i], linewidth=1.5, linestyle='--')

ax.axvline(x=2025.5, color='gray', linestyle=':', linewidth=1.5, label='Inicio predicción')
ax.set_title('Predicción de emisiones de CO₂ - Top 10 países (2026-2035)', fontsize=14)
ax.set_xlabel('Año')
ax.set_ylabel('Emisiones CO₂ (toneladas)')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}Gt'))
plt.tight_layout()
plt.savefig('grafics/pred_co2_top10.png', dpi=150)
plt.show()
print("Gráfico guardado en grafics/pred_co2_top10.png")

# --- Gráfico 2: Emisiones globales futuras (suma todos los países) ---
df_hist_global = (df_co2[~df_co2['Entity'].isin(excluir)]
                  .groupby('Year')[target_col].sum().reset_index().tail(50))
df_pred_global = df_pred.groupby('Year')['Predicted CO₂ emissions'].sum().reset_index()

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.fill_between(df_hist_global['Year'], df_hist_global[target_col],
                 alpha=0.3, color='steelblue', label='Histórico')
ax2.plot(df_hist_global['Year'], df_hist_global[target_col], color='steelblue', linewidth=2)
ax2.fill_between(df_pred_global['Year'], df_pred_global['Predicted CO₂ emissions'],
                 alpha=0.3, color='tomato', label='Predicción (2026-2035)')
ax2.plot(df_pred_global['Year'], df_pred_global['Predicted CO₂ emissions'],
         color='tomato', linewidth=2.5, linestyle='--', marker='o', markersize=5)
ax2.axvline(x=2025.5, color='gray', linestyle=':', linewidth=1.5)
ax2.set_title('Emisiones globales de CO₂: histórico + predicción', fontsize=14)
ax2.set_xlabel('Año')
ax2.set_ylabel('Emisiones CO₂ (Gt)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}Gt'))
ax2.legend()
plt.tight_layout()
plt.savefig('grafics/pred_co2_global.png', dpi=150)
plt.show()
print("Gráfico guardado en grafics/pred_co2_global.png")

# --- Gráfico 3: Tendencias por región (top países clave) ---
paises_clave = {
    'China': 'Asia', 'India': 'Asia', 'United States': 'América',
    'Russia': 'Europa/Asia', 'Germany': 'Europa', 'Brazil': 'América',
    'Japan': 'Asia', 'Spain': 'Europa', 'Saudi Arabia': 'Oriente Medio'
}

fig3, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()

for i, (pais, region) in enumerate(paises_clave.items()):
    hist = df_co2[df_co2['Entity'] == pais][['Year', target_col]].dropna().sort_values('Year').tail(30)
    pred = df_pred[df_pred['Entity'] == pais].sort_values('Year')
    if hist.empty or pred.empty:
        continue
    axes[i].plot(hist['Year'], hist[target_col], color='steelblue', linewidth=2, label='Histórico')
    axes[i].plot(pred['Year'], pred['Predicted CO₂ emissions'],
                 color='tomato', linewidth=2, linestyle='--', marker='o', markersize=4, label='Predicción')
    axes[i].plot([hist['Year'].iloc[-1], pred['Year'].iloc[0]],
                 [hist[target_col].iloc[-1], pred['Predicted CO₂ emissions'].iloc[0]],
                 color='gray', linewidth=1, linestyle='--')
    axes[i].set_title(f'{pais} ({region})', fontsize=11)
    axes[i].set_xlabel('Año', fontsize=8)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}Mt'))
    axes[i].legend(fontsize=7)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Predicción de emisiones CO₂ por país (2026-2035)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('grafics/pred_co2_paises_clave.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico guardado en grafics/pred_co2_paises_clave.png")

# --- Resumen en consola ---
print("\n── TOP 5 países con mayor crecimiento previsto de emisiones ──")
df_tendencia = df_metricas.sort_values('Tendencia (t/año)', ascending=False).head(5)
print(df_tendencia[['Entity', 'Tendencia (t/año)', 'R2']].to_string(index=False))

print("\n── TOP 5 países con mayor reducción prevista de emisiones ──")
df_reduccion = df_metricas.sort_values('Tendencia (t/año)', ascending=True).head(5)
print(df_reduccion[['Entity', 'Tendencia (t/año)', 'R2']].to_string(index=False))
