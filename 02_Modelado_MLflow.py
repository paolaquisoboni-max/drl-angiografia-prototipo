#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
MODELADO SUPERVISADO CON MLFLOW - ACTUALIZADO PARA 976 REGISTROS
Microproyecto: Niveles de Referencia Diagnósticos en Angiografía
========================================================================
Universidad de los Andes - Maestría en IA
Febrero 2026
========================================================================
"""

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import joblib
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MODELADO SUPERVISADO CON MLFLOW - PREDICCIÓN DE EXCEDENCIA DRL")
print("="*80)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================
print("\n[1/8] CARGANDO DATASET...")
df = pd.read_csv('data/angiografia_dataset_1000_clean.csv')
print(f"✓ Dataset cargado: {df.shape[0]} registros × {df.shape[1]} columnas")

# Verificar distribución del target
print(f"\nDistribución del target:")
print(df['excede_DRL'].value_counts())
print(f"\nBalance: {df['excede_DRL'].value_counts(normalize=True) * 100}")

# ============================================================================
# 2. PREPROCESAMIENTO
# ============================================================================
print("\n[2/8] PREPROCESANDO DATOS...")

# Codificar tipo de procedimiento
le = LabelEncoder()
df['Tipo_Procedimiento_encoded'] = le.fit_transform(df['Tipo_Procedimiento'])

# Seleccionar features
feature_cols = ['Kar_mGy', 'Tiempo_Fluoroscopia_min', 
                'Edad', 'Peso', 'Tipo_Procedimiento_encoded']
X = df[feature_cols]
y = df['excede_DRL']

print(f"✓ Features seleccionadas: {len(feature_cols)}")
print(f"  {feature_cols}")

# ============================================================================
# 3. DIVISIÓN TRAIN/TEST
# ============================================================================
print("\n[3/8] DIVIDIENDO DATOS...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"✓ Train: {X_train.shape[0]} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test:  {X_test.shape[0]} registros ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nDistribución train - No excede: {(y_train==0).sum()}, Excede: {(y_train==1).sum()}")
print(f"Distribución test  - No excede: {(y_test==0).sum()}, Excede: {(y_test==1).sum()}")

# ============================================================================
# 4. CONFIGURACIÓN DE MLFLOW
# ============================================================================
print("\n[4/8] CONFIGURANDO MLFLOW...")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("DRL_Angiografia_Prediction_976")
print("✓ Experimento: DRL_Angiografia_Prediction_976")
print("✓ Tracking URI: file:./mlruns")

# ============================================================================
# 5. ENTRENAMIENTO DE MODELOS
# ============================================================================
print("\n[5/8] ENTRENANDO MODELOS...")

modelos = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

resultados = []

for nombre_modelo, modelo in modelos.items():
    print(f"\n--- Entrenando {nombre_modelo} ---")
    
    with mlflow.start_run(run_name=nombre_modelo):
        # Entrenar modelo
        modelo.fit(X_train, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Guardar métricas
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        # Log en MLflow
        mlflow.log_params(modelo.get_params())
        mlflow.log_metrics(metrics)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Visualizar matriz de confusión
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No excede', 'Excede'],
                   yticklabels=['No excede', 'Excede'],
                   annot_kws={'size': 16, 'weight': 'bold'})
        ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
        ax.set_ylabel('Real', fontsize=12, fontweight='bold')
        ax.set_title(f'Matriz de Confusión - {nombre_modelo}', fontsize=14, fontweight='bold')
        
        # Añadir texto con métricas
        textstr = f'Accuracy:  {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\nF1-Score:  {f1:.4f}\nROC-AUC:   {roc_auc:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.55, 0.5, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='center', bbox=props, family='monospace')
        
        plt.tight_layout()
        confusion_path = f'results/confusion_matrix_{nombre_modelo}.png'
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log artefacto en MLflow
        mlflow.log_artifact(confusion_path)
        
        # Log modelo en MLflow
        mlflow.sklearn.log_model(modelo, "model")
        
        # Imprimir resultados
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Matriz de confusión guardada: {confusion_path}")
        
        # Guardar resultados
        resultados.append({
            'Modelo': nombre_modelo,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'TN': cm[0, 0],
            'FP': cm[0, 1],
            'FN': cm[1, 0],
            'TP': cm[1, 1],
            'modelo_obj': modelo
        })

# ============================================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================================
print("\n[6/8] COMPARANDO MODELOS...")

# Crear DataFrame de comparación
df_resultados = pd.DataFrame(resultados)
df_comparacion = df_resultados[['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].copy()
df_comparacion.to_csv('results/comparacion_modelos.csv', index=False)
print("✓ Tabla de comparación guardada: results/comparacion_modelos.csv")
print(f"\n{df_comparacion.to_string(index=False)}")

# Visualización comparativa
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(df_comparacion))
width = 0.15

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    offset = width * (i - 2)
    bars = ax.bar(x + offset, df_comparacion[metric], width, label=metric, color=color, alpha=0.8)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Desempeño de Modelos', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_comparacion['Modelo'], rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0.7, 1.05])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/fig4_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico de comparación guardado: results/fig4_comparacion_modelos.png")
plt.close()

# ============================================================================
# 7. SELECCIÓN DEL MEJOR MODELO
# ============================================================================
print("\n[7/8] SELECCIONANDO MEJOR MODELO...")

# Seleccionar basado en F1-Score
mejor_idx = df_resultados['F1-Score'].idxmax()
mejor_resultado = df_resultados.iloc[mejor_idx]
mejor_modelo = mejor_resultado['modelo_obj']
mejor_nombre = mejor_resultado['Modelo']

print(f"✓ Mejor modelo: {mejor_nombre}")
print(f"  F1-Score: {mejor_resultado['F1-Score']:.4f}")
print(f"  Accuracy: {mejor_resultado['Accuracy']:.4f}")
print(f"  Recall:   {mejor_resultado['Recall']:.4f}")

# Guardar mejor modelo
joblib.dump(mejor_modelo, 'models/best_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("✓ Mejor modelo guardado: models/best_model.pkl")
print("✓ LabelEncoder guardado: models/label_encoder.pkl")

# Feature Importance (si es Random Forest o Gradient Boosting)
if hasattr(mejor_modelo, 'feature_importances_'):
    print("\n[8/8] CALCULANDO IMPORTANCIA DE VARIABLES...")
    
    importances = mejor_modelo.feature_importances_
    feature_names = feature_cols
    
    # Crear DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    feature_importance_df.to_csv('results/feature_importance.csv', index=False)
    print("✓ Importancia guardada: results/feature_importance.csv")
    print(f"\n{feature_importance_df.to_string(index=False)}")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
    bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    
    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, feature_importance_df['Importance'])):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_title(f'Importancia de Variables - {mejor_nombre}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fig5_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: results/fig5_feature_importance.png")
    plt.close()
else:
    print("\n[8/8] Modelo sin feature importances (Regresión Logística)")

# ============================================================================
# 9. REPORTE FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DEL ENTRENAMIENTO")
print("="*80)

print(f"\n📊 DATASET:")
print(f"   • Total registros: {len(X)}")
print(f"   • Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   • Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n🤖 MODELOS ENTRENADOS:")
for i, row in df_resultados.iterrows():
    print(f"   {i+1}. {row['Modelo']:25s} - F1: {row['F1-Score']:.4f}, Acc: {row['Accuracy']:.4f}")

print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
print(f"   • Accuracy:  {mejor_resultado['Accuracy']:.4f} ({mejor_resultado['Accuracy']*100:.2f}%)")
print(f"   • Precision: {mejor_resultado['Precision']:.4f}")
print(f"   • Recall:    {mejor_resultado['Recall']:.4f}")
print(f"   • F1-Score:  {mejor_resultado['F1-Score']:.4f}")
print(f"   • ROC-AUC:   {mejor_resultado['ROC-AUC']:.4f}")

print(f"\n📊 MATRIZ DE CONFUSIÓN (Test set, n={len(y_test)}):")
print(f"   • True Negatives:  {int(mejor_resultado['TN'])}")
print(f"   • False Positives: {int(mejor_resultado['FP'])}")
print(f"   • False Negatives: {int(mejor_resultado['FN'])}")
print(f"   • True Positives:  {int(mejor_resultado['TP'])}")
print(f"   • Total errores:   {int(mejor_resultado['FP'] + mejor_resultado['FN'])} de {len(y_test)}")

print(f"\n📂 ARCHIVOS GENERADOS:")
print(f"   1. models/best_model.pkl")
print(f"   2. models/label_encoder.pkl")
print(f"   3. results/comparacion_modelos.csv")
print(f"   4. results/feature_importance.csv (si aplica)")
print(f"   5. results/confusion_matrix_Logistic_Regression.png")
print(f"   6. results/confusion_matrix_Random_Forest.png")
print(f"   7. results/confusion_matrix_Gradient_Boosting.png")
print(f"   8. results/fig4_comparacion_modelos.png")
print(f"   9. results/fig5_feature_importance.png (si aplica)")

print(f"\n🔬 MLFLOW:")
print(f"   • Experimento: DRL_Angiografia_Prediction_976")
print(f"   • Runs: {len(modelos)}")
print(f"   • UI: mlflow ui --host 127.0.0.1 --port 5000")
print(f"   • URL: http://127.0.0.1:5000")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*80)
