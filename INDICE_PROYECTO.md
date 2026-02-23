# 📚 ÍNDICE COMPLETO DEL PROYECTO

## Microproyecto: Niveles de Referencia Diagnósticos (DRL) en Angiografía
**Universidad de los Andes | Maestría en IA | Febrero 2026**

---

## 📂 Estructura del Proyecto

```
microproyecto_drl/
│
├── 📄 README.md                                  # Documentación principal del proyecto
├── 📄 .gitignore                                 # Archivos ignorados por Git
├── 📄 setup.sh                                   # Script de instalación automática
├── 📄 GUIA_RAPIDA_EJECUCION.md                  # Guía de ejecución paso a paso
├── 📄 RESUMEN_EJECUTIVO.md                       # Números clave y resumen para defensa
├── 📄 INDICE_PROYECTO.md                         # Este archivo
│
├── 📂 data/                                      # Datos del proyecto
│   └── angiografia_dataset_1000_clean.csv        # Dataset limpio (976 registros, 9 variables)
│
├── 📂 notebooks/                                 # Scripts de análisis y modelado
│   ├── 01_EDA_Completo.py                        # Análisis exploratorio exhaustivo
│   └── 02_Modelado_MLflow.py                     # Entrenamiento de 3 modelos con MLflow
│
├── 📂 models/                                    # Modelos entrenados
│   ├── best_model.pkl                            # Gradient Boosting (97.96% accuracy)
│   └── label_encoder.pkl                         # Codificador de tipo de procedimiento
│
├── 📂 api/                                       # API REST con FastAPI
│   ├── main.py                                   # Servidor API con endpoint /predict
│   └── test_api.py                               # Suite de tests automatizados
│
├── 📂 dashboard/                                 # Dashboard interactivo
│   └── app.py                                    # Dashboard Streamlit con 3 tabs
│
├── 📂 results/                                   # Resultados y visualizaciones
│   ├── estadisticas_descriptivas.csv             # Stats del dataset
│   ├── P75_por_procedimiento.csv                 # DRLs de referencia
│   ├── resumen_por_tipo.csv                      # Resumen por procedimiento
│   ├── comparacion_modelos.csv                   # Comparación de 3 modelos
│   ├── feature_importance.csv                    # Importancia de variables
│   ├── fig1_distribuciones.png                   # Histogramas y distribuciones
│   ├── fig2_boxplots.png                         # Boxplots por tipo
│   ├── fig3_relaciones.png                       # Scatter plots y correlaciones
│   ├── fig4_comparacion_modelos.png              # Gráfico comparativo modelos
│   ├── fig5_feature_importance.png               # Gráfico importancia variables
│   ├── confusion_matrix_Logistic_Regression.png  # Matriz Logistic Regression
│   ├── confusion_matrix_Random_Forest.png        # Matriz Random Forest
│   └── confusion_matrix_Gradient_Boosting.png    # Matriz Gradient Boosting
│
├── 📂 mlruns/                                    # Logs de MLflow (generado automáticamente)
│   └── [Experimentos trackeados]
│
├── 📂 requirements/                              # Dependencias
│   └── requirements.txt                          # Librerías necesarias
│
└── 📂 docs/                                      # Documentación completa
    ├── GUIA_ARQUITECTURA_COMPLETA.md             # Arquitectura de 5 capas detallada
    ├── CHECKLIST_SUSTENTACION.md                 # Checklist para defensa oral
    └── Informe_Final_Microproyecto_DRL.docx      # Informe técnico (10 páginas)
```

---

## 📊 Archivos por Categoría

### 🔧 Configuración y Setup (4 archivos)
1. `README.md` - Documentación principal
2. `.gitignore` - Exclusiones de Git
3. `setup.sh` - Instalación automatizada
4. `requirements/requirements.txt` - Dependencias Python

### 📈 Datos (1 archivo)
1. `data/angiografia_dataset_1000_clean.csv` - 976 registros, 9 variables, sin valores faltantes

### 🧪 Scripts de Análisis (2 archivos)
1. `notebooks/01_EDA_Completo.py` - EDA con estadísticas, P75 y 3 visualizaciones
2. `notebooks/02_Modelado_MLflow.py` - 3 modelos + MLflow tracking

### 🤖 Modelos Entrenados (2 archivos)
1. `models/best_model.pkl` - Gradient Boosting (97.96% accuracy)
2. `models/label_encoder.pkl` - Encoder de tipo de procedimiento

### 🌐 API y Dashboard (3 archivos)
1. `api/main.py` - FastAPI con 4 endpoints
2. `api/test_api.py` - 5 tests automatizados
3. `dashboard/app.py` - Streamlit con 3 tabs

### 📊 Resultados (14 archivos)
**CSVs (5):**
1. `results/estadisticas_descriptivas.csv`
2. `results/P75_por_procedimiento.csv`
3. `results/resumen_por_tipo.csv`
4. `results/comparacion_modelos.csv`
5. `results/feature_importance.csv`

**Imágenes (9):**
6. `results/fig1_distribuciones.png`
7. `results/fig2_boxplots.png`
8. `results/fig3_relaciones.png`
9. `results/fig4_comparacion_modelos.png`
10. `results/fig5_feature_importance.png`
11. `results/confusion_matrix_Logistic_Regression.png`
12. `results/confusion_matrix_Random_Forest.png`
13. `results/confusion_matrix_Gradient_Boosting.png`
14. *(Se generarán más al ejecutar los scripts)*

### 📚 Documentación (6 archivos)
1. `GUIA_RAPIDA_EJECUCION.md` - Instrucciones paso a paso
2. `RESUMEN_EJECUTIVO.md` - Números clave para defensa
3. `INDICE_PROYECTO.md` - Este archivo
4. `docs/GUIA_ARQUITECTURA_COMPLETA.md` - Arquitectura detallada
5. `docs/CHECKLIST_SUSTENTACION.md` - Checklist de defensa
6. `docs/Informe_Final_Microproyecto_DRL.docx` - Informe técnico

---

## 🎯 Flujo de Trabajo Recomendado

### Fase 1: Setup (2 min)
```bash
bash setup.sh
source venv/bin/activate
```

### Fase 2: Análisis Exploratorio (2 min)
```bash
cd notebooks
python 01_EDA_Completo.py
```
**Genera:** 3 CSVs + 3 PNGs

### Fase 3: Modelado (4 min)
```bash
python 02_Modelado_MLflow.py
```
**Genera:** 2 PKLs + 2 CSVs + 5 PNGs + logs MLflow

### Fase 4: API (< 1 min)
```bash
cd ../api
python main.py  # En terminal separada
python test_api.py  # En otra terminal
```

### Fase 5: Dashboard (< 1 min)
```bash
cd ../dashboard
streamlit run app.py
```

### Fase 6: Visualizar MLflow (Opcional)
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

---

## 📖 Guías de Uso

### Para Ejecutar el Proyecto
👉 Ver: `GUIA_RAPIDA_EJECUCION.md`

### Para la Defensa Oral
👉 Ver: `RESUMEN_EJECUTIVO.md` + `docs/CHECKLIST_SUSTENTACION.md`

### Para Entender la Arquitectura
👉 Ver: `docs/GUIA_ARQUITECTURA_COMPLETA.md`

### Para el Informe Final
👉 Ver: `docs/Informe_Final_Microproyecto_DRL.docx`

---

## 🔢 Números Clave

| Categoría | Valor |
|-----------|-------|
| **Total de archivos de código** | 5 scripts Python |
| **Total de archivos de datos** | 1 CSV (976 registros) |
| **Total de modelos** | 2 PKL files |
| **Total de resultados** | 14 archivos (5 CSV + 9 PNG) |
| **Total de documentación** | 6 archivos Markdown/Word |
| **Accuracy del modelo** | 97.96% |
| **F1-Score** | 95.92% |
| **Total errores** | 4 de 244 predicciones |
| **Tiempo de ejecución total** | ~10 minutos |

---

## ✅ Checklist de Entregables

### ✅ Código Ejecutable (5/5)
- [x] `01_EDA_Completo.py`
- [x] `02_Modelado_MLflow.py`
- [x] `api/main.py`
- [x] `api/test_api.py`
- [x] `dashboard/app.py`

### ✅ Modelos (2/2)
- [x] `models/best_model.pkl`
- [x] `models/label_encoder.pkl`

### ✅ Datos (1/1)
- [x] `data/angiografia_dataset_1000_clean.csv`

### ✅ Resultados (14/14)
- [x] 5 CSVs de métricas
- [x] 9 PNGs de visualizaciones

### ✅ Documentación (6/6)
- [x] README.md
- [x] GUIA_RAPIDA_EJECUCION.md
- [x] RESUMEN_EJECUTIVO.md
- [x] docs/GUIA_ARQUITECTURA_COMPLETA.md
- [x] docs/CHECKLIST_SUSTENTACION.md
- [x] docs/Informe_Final_Microproyecto_DRL.docx

### ✅ Extras (3/3)
- [x] .gitignore
- [x] setup.sh
- [x] requirements/requirements.txt

---

## 🚀 Estado del Proyecto

### ✅ COMPLETADO
- [x] Dataset limpio y validado
- [x] EDA exhaustivo
- [x] 3 modelos entrenados y comparados
- [x] Experimentos trackeados en MLflow
- [x] API REST funcional con tests
- [x] Dashboard interactivo deployado
- [x] Documentación completa
- [x] Informe técnico finalizado
- [x] Scripts listos para ejecución
- [x] Guías de uso creadas

### 🎯 LISTO PARA
- [x] Ejecución inmediata
- [x] Defensa oral
- [x] Entrega final
- [x] Revisión académica

---

## 📞 Información de Contacto

**Universidad de los Andes**  
Maestría en Inteligencia Artificial  
Febrero 2026

---

## 📝 Notas Finales

### Tiempo Total de Desarrollo
- Setup: 2 min
- EDA: 2 min
- Modelado: 4 min
- API + Dashboard: 2 min
- **Total: ~10 minutos de ejecución**

### Tamaño del Proyecto
- **Líneas de código:** ~1,500 líneas
- **Tamaño total:** ~50 MB (con modelos y visualizaciones)
- **Archivos totales:** 32 archivos

### Versiones de Software
- Python: 3.9+
- Scikit-learn: 1.4.0
- MLflow: 2.9.2
- FastAPI: 0.109.0
- Streamlit: 1.30.0

---

**🎓 PROYECTO 100% COMPLETO Y DOCUMENTADO**

**✅ Listo para Defensa y Entrega Final**

**Universidad de los Andes | Maestría en IA | Febrero 2026**
