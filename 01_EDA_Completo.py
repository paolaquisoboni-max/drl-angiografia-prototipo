#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
ANÁLISIS EXPLORATORIO DE DATOS (EDA) COMPLETO
Microproyecto: Niveles de Referencia Diagnósticos en Angiografía
========================================================================
Universidad de los Andes - Maestría en IA
Febrero 2026
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS - NIVELES DE REFERENCIA DIAGNÓSTICOS")
print("="*80)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("\n[1/8] CARGANDO DATASET...")
df = pd.read_csv('data/angiografia_dataset_1000_clean.csv')
print(f"✓ Dataset cargado: {df.shape[0]} registros × {df.shape[1]} columnas")

# ============================================================================
# 2. INFORMACIÓN GENERAL
# ============================================================================
print("\n[2/8] ANÁLISIS DE ESTRUCTURA...")
print(f"\nDimensiones: {df.shape}")
print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"\nTipos de datos:")
print(df.dtypes)
print(f"\nValores faltantes:")
print(df.isnull().sum())

# ============================================================================
# 3. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("\n[3/8] CALCULANDO ESTADÍSTICAS DESCRIPTIVAS...")

# Variables numéricas
numeric_cols = ['Edad', 'Peso', 'PKA_Gycm2', 'Kar_mGy', 'Tiempo_Fluoroscopia_min']
stats = df[numeric_cols].describe().T
stats['missing'] = df[numeric_cols].isnull().sum()
stats['missing_pct'] = (stats['missing'] / len(df) * 100).round(2)

# Guardar estadísticas
stats.to_csv('results/estadisticas_descriptivas.csv')
print("✓ Estadísticas guardadas: results/estadisticas_descriptivas.csv")
print(f"\n{stats}")

# ============================================================================
# 4. CÁLCULO DE P75 (DRL) POR TIPO DE PROCEDIMIENTO
# ============================================================================
print("\n[4/8] CALCULANDO NIVELES DE REFERENCIA (P75)...")

p75_por_tipo = df.groupby('Tipo_Procedimiento').agg({
    'PKA_Gycm2': [
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('P25', lambda x: x.quantile(0.25)),
        ('P50', 'median'),
        ('P75 (DRL)', lambda x: x.quantile(0.75)),
        ('Max', 'max')
    ]
}).round(2)

p75_por_tipo.columns = p75_por_tipo.columns.droplevel(0)
p75_por_tipo = p75_por_tipo.reset_index()
p75_por_tipo.to_csv('results/P75_por_procedimiento.csv', index=False)
print("✓ P75 guardados: results/P75_por_procedimiento.csv")
print(f"\n{p75_por_tipo}")

# ============================================================================
# 5. ANÁLISIS DE LA VARIABLE OBJETIVO
# ============================================================================
print("\n[5/8] ANALIZANDO VARIABLE OBJETIVO (excede_DRL)...")
target_dist = df['excede_DRL'].value_counts().sort_index()
target_pct = df['excede_DRL'].value_counts(normalize=True).sort_index() * 100

print(f"\nDistribución:")
print(f"  No excede (0): {target_dist[0]:4d} casos ({target_pct[0]:.1f}%)")
print(f"  Excede (1):    {target_dist[1]:4d} casos ({target_pct[1]:.1f}%)")

# ============================================================================
# 6. VISUALIZACIONES
# ============================================================================
print("\n[6/8] GENERANDO VISUALIZACIONES...")

# -------------------------------------------------------------------------
# FIGURA 1: Distribuciones de variables numéricas
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribuciones de Variables Numéricas', fontsize=16, fontweight='bold')

variables = [
    ('Edad', 'años', 'skyblue'),
    ('Peso', 'kg', 'lightcoral'),
    ('PKA_Gycm2', 'Gy·cm²', 'lightgreen'),
    ('Kar_mGy', 'mGy', 'lightsalmon'),
    ('Tiempo_Fluoroscopia_min', 'minutos', 'plum')
]

for idx, (var, unit, color) in enumerate(variables):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[var], bins=30, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'{var} ({unit})', fontsize=10)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.set_title(f'Distribución de {var}', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir estadísticas
    mean_val = df[var].mean()
    median_val = df[var].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.1f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.1f}')
    ax.legend(fontsize=8)

# Última celda: distribución del target
ax = axes[1, 2]
target_counts = df['excede_DRL'].value_counts().sort_index()
colors_target = ['#90EE90', '#FFB6C1']
bars = ax.bar(['No excede', 'Excede'], target_counts.values, color=colors_target, edgecolor='black', alpha=0.8)
ax.set_ylabel('Frecuencia', fontsize=10)
ax.set_title('Variable Objetivo: excede_DRL', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/fig1_distribuciones.png', dpi=300, bbox_inches='tight')
print("✓ Figura 1 guardada: results/fig1_distribuciones.png")
plt.close()

# -------------------------------------------------------------------------
# FIGURA 2: Box plots de PKA por tipo de procedimiento
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
df_sorted = df.sort_values('Tipo_Procedimiento')
box_plot = ax.boxplot(
    [df_sorted[df_sorted['Tipo_Procedimiento'] == tipo]['PKA_Gycm2'].values 
     for tipo in sorted(df['Tipo_Procedimiento'].unique())],
    labels=[tipo.replace(' ', '\n') for tipo in sorted(df['Tipo_Procedimiento'].unique())],
    patch_artist=True,
    notch=True,
    showmeans=True
)

# Colorear cajas
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FF99CC']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Añadir líneas de P75
p75_values = p75_por_tipo.sort_values('Tipo_Procedimiento')['P75 (DRL)'].values
for i, p75 in enumerate(p75_values, 1):
    ax.hlines(p75, i-0.4, i+0.4, colors='red', linestyles='--', linewidth=2, label='P75 (DRL)' if i == 1 else '')

ax.set_xlabel('Tipo de Procedimiento', fontsize=12, fontweight='bold')
ax.set_ylabel('PKA (Gy·cm²)', fontsize=12, fontweight='bold')
ax.set_title('Distribución de PKA por Tipo de Procedimiento', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/fig2_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Figura 2 guardada: results/fig2_boxplots.png")
plt.close()

# -------------------------------------------------------------------------
# FIGURA 3: Relaciones entre variables
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Relaciones entre Variables Dosimétricas', fontsize=16, fontweight='bold')

# Subplot 1: PKA vs Tiempo
ax = axes[0, 0]
scatter = ax.scatter(df['Tiempo_Fluoroscopia_min'], df['PKA_Gycm2'], 
                     c=df['excede_DRL'], cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black')
ax.set_xlabel('Tiempo de Fluoroscopia (min)', fontsize=11)
ax.set_ylabel('PKA (Gy·cm²)', fontsize=11)
ax.set_title('PKA vs Tiempo de Fluoroscopia', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Excede DRL')

# Calcular correlación
corr = df['Tiempo_Fluoroscopia_min'].corr(df['PKA_Gycm2'])
ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot 2: PKA vs Ka,r
ax = axes[0, 1]
scatter = ax.scatter(df['Kar_mGy'], df['PKA_Gycm2'], 
                     c=df['excede_DRL'], cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black')
ax.set_xlabel('Ka,r (mGy)', fontsize=11)
ax.set_ylabel('PKA (Gy·cm²)', fontsize=11)
ax.set_title('PKA vs Ka,r', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Excede DRL')

corr = df['Kar_mGy'].corr(df['PKA_Gycm2'])
ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot 3: Edad vs PKA
ax = axes[1, 0]
scatter = ax.scatter(df['Edad'], df['PKA_Gycm2'], 
                     c=df['excede_DRL'], cmap='RdYlGn_r', alpha=0.6, s=50, edgecolors='black')
ax.set_xlabel('Edad (años)', fontsize=11)
ax.set_ylabel('PKA (Gy·cm²)', fontsize=11)
ax.set_title('PKA vs Edad del Paciente', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Excede DRL')

corr = df['Edad'].corr(df['PKA_Gycm2'])
ax.text(0.05, 0.95, f'Correlación: {corr:.3f}', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot 4: Matriz de correlación
ax = axes[1, 1]
corr_matrix = df[numeric_cols].corr()
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([col.replace('_', '\n') for col in numeric_cols], fontsize=9)
ax.set_title('Matriz de Correlación', fontsize=12, fontweight='bold')

# Añadir valores
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8, fontweight='bold')

plt.colorbar(im, ax=ax, label='Correlación')
plt.tight_layout()
plt.savefig('results/fig3_relaciones.png', dpi=300, bbox_inches='tight')
print("✓ Figura 3 guardada: results/fig3_relaciones.png")
plt.close()

# ============================================================================
# 7. ANÁLISIS POR TIPO DE PROCEDIMIENTO
# ============================================================================
print("\n[7/8] ANÁLISIS DETALLADO POR TIPO DE PROCEDIMIENTO...")

summary_by_type = df.groupby('Tipo_Procedimiento').agg({
    'PKA_Gycm2': ['mean', 'std', 'min', 'max'],
    'Kar_mGy': ['mean', 'std'],
    'Tiempo_Fluoroscopia_min': ['mean', 'std'],
    'Edad': ['mean', 'std'],
    'Peso': ['mean', 'std'],
    'excede_DRL': ['sum', lambda x: (x.sum() / len(x) * 100)]
}).round(2)

summary_by_type.columns = ['_'.join(col).strip() for col in summary_by_type.columns.values]
summary_by_type = summary_by_type.rename(columns={'excede_DRL_<lambda_0>': 'excede_DRL_pct'})
summary_by_type.to_csv('results/resumen_por_tipo.csv')
print("✓ Resumen guardado: results/resumen_por_tipo.csv")
print(f"\n{summary_by_type}")

# ============================================================================
# 8. REPORTE FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DEL ANÁLISIS EXPLORATORIO")
print("="*80)
print(f"\n📊 DATASET:")
print(f"   • Registros totales: {len(df)}")
print(f"   • Variables: {len(df.columns)}")
print(f"   • Valores faltantes: {df.isnull().sum().sum()} (0%)")

print(f"\n📈 VARIABLE OBJETIVO (excede_DRL):")
print(f"   • No excede DRL: {target_dist[0]} casos ({target_pct[0]:.1f}%)")
print(f"   • Excede DRL: {target_dist[1]} casos ({target_pct[1]:.1f}%)")

print(f"\n🏥 TIPOS DE PROCEDIMIENTO:")
for tipo, count in df['Tipo_Procedimiento'].value_counts().sort_index().items():
    p75 = p75_por_tipo[p75_por_tipo['Tipo_Procedimiento'] == tipo]['P75 (DRL)'].values[0]
    print(f"   • {tipo:35s}: {count:3d} casos | P75 = {p75:7.2f} Gy·cm²")

print(f"\n💡 ESTADÍSTICAS DOSIMÉTRICAS:")
print(f"   • PKA:    {df['PKA_Gycm2'].mean():6.2f} ± {df['PKA_Gycm2'].std():.2f} Gy·cm²")
print(f"   • Ka,r:   {df['Kar_mGy'].mean():6.2f} ± {df['Kar_mGy'].std():.2f} mGy")
print(f"   • Tiempo: {df['Tiempo_Fluoroscopia_min'].mean():6.2f} ± {df['Tiempo_Fluoroscopia_min'].std():.2f} min")

print(f"\n📂 ARCHIVOS GENERADOS:")
print(f"   1. results/estadisticas_descriptivas.csv")
print(f"   2. results/P75_por_procedimiento.csv")
print(f"   3. results/resumen_por_tipo.csv")
print(f"   4. results/fig1_distribuciones.png")
print(f"   5. results/fig2_boxplots.png")
print(f"   6. results/fig3_relaciones.png")

print("\n" + "="*80)
print("✅ ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
print("="*80)
