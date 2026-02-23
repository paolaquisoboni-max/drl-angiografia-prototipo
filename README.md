# Microproyecto: Niveles de Referencia Diagnósticos (DRL) en Angiografía

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)
![MLflow](https://img.shields.io/badge/MLflow-2.9.2-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-yellow)

## 📋 Descripción

Sistema de Machine Learning para predecir si procedimientos angiográficos exceden los **Niveles de Referencia Diagnósticos (DRL)** establecidos por organismos internacionales como la IAEA y el ACR.

El DRL corresponde al **percentil 75 (P75)** de la distribución de dosis de radiación para un tipo específico de procedimiento en pacientes estándar.

## 🎯 Objetivo

Desarrollar un modelo de clasificación supervisada que:
- Prediga si un procedimiento angiográfico excede el DRL (P75)
- Proporcione probabilidades de excedencia
- Ayude a optimizar protocolos radiológicos
- Mejore la protección radiológica del paciente

## 📊 Dataset

- **Registros:** 976 procedimientos angiográficos limpios
- **Variables:** 9 (ID, Fecha, Tipo_Procedimiento, Edad, Peso, PKA, Ka,r, Tiempo_Fluoroscopia, excede_DRL)
- **Target:** `excede_DRL` (0 = No excede, 1 = Excede)
- **Balance:** 731 No excede (74.9%), 245 Excede (25.1%)
- **Tipos de procedimientos:** 5 categorías
  - Coronariografía Diagnóstica (439 casos) - P75: 66.60 Gy·cm²
  - Angiografía Cerebral (197 casos) - P75: 135.09 Gy·cm²
  - Angiografía Aorta Abdominal (126 casos) - P75: 150.53 Gy·cm²
  - Angiografía Periférica (117 casos) - P75: 79.14 Gy·cm²
  - Angiografía Renal (97 casos) - P75: 55.66 Gy·cm²

## 🚀 Tecnologías

- **Python 3.9+**
- **Scikit-learn:** Modelos de ML (Logistic Regression, Random Forest, Gradient Boosting)
- **MLflow:** Tracking de experimentos y gestión de modelos
- **FastAPI:** API REST para inferencia
- **Streamlit:** Dashboard interactivo
- **Pandas & NumPy:** Procesamiento de datos
- **Matplotlib & Seaborn & Plotly:** Visualizaciones

## 📁 Estructura del Proyecto

```
microproyecto_drl/
│
├── data/
│   ├── angiografia_dataset_1000_clean.csv    # Dataset limpio (976 registros)
│   └── angiografia_dataset_1000.csv          # Dataset original (1000 registros)
│
├── notebooks/
│   ├── 01_EDA_Completo.py                    # Análisis exploratorio exhaustivo
│   └── 02_Modelado_MLflow.py                 # Entrenamiento con 3 modelos + MLflow
│
├── models/
│   ├── best_model.pkl                        # Mejor modelo entrenado (Gradient Boosting)
│   └── label_encoder.pkl                     # Codificador de tipo de procedimiento
│
├── api/
│   ├── main.py                               # API FastAPI
│   └── test_api.py                           # Tests de la API
│
├── dashboard/
│   └── app.py                                # Dashboard Streamlit
│
├── results/
│   ├── estadisticas_descriptivas.csv         # Estadísticas del dataset
│   ├── P75_por_procedimiento.csv             # DRLs (P75) por tipo de procedimiento
│   ├── resumen_por_tipo.csv                  # Resumen agregado por tipo
│   ├── comparacion_modelos.csv               # Comparación de desempeño
│   ├── feature_importance.csv                # Importancia de variables
│   ├── fig1_distribuciones.png               # Histogramas y distribuciones
│   ├── fig2_boxplots.png                     # Boxplots por tipo de procedimiento
│   ├── fig3_relaciones.png                   # Scatter plots y correlaciones
│   ├── fig4_comparacion_modelos.png          # Gráfico comparativo de modelos
│   ├── fig5_feature_importance.png           # Gráfico de importancia de variables
│   ├── confusion_matrix_Logistic_Regression.png
│   ├── confusion_matrix_Random_Forest.png
│   └── confusion_matrix_Gradient_Boosting.png
│
├── mlruns/                                   # Directorio de MLflow (generado automáticamente)
│
├── requirements/
│   └── requirements.txt                      # Dependencias del proyecto
│
├── docs/
│   ├── GUIA_ARQUITECTURA_COMPLETA.md
│   ├── CHECKLIST_SUSTENTACION.md
│   └── Informe_Final_Microproyecto_DRL.docx
│
└── README.md                                 # Este archivo
```

## 🔧 Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd microproyecto_drl
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements/requirements.txt
```

## 📖 Uso

### 1️⃣ Análisis Exploratorio de Datos (EDA)

```bash
cd notebooks
python 01_EDA_Completo.py
```

**Genera:**
- `results/estadisticas_descriptivas.csv`
- `results/P75_por_procedimiento.csv`
- `results/resumen_por_tipo.csv`
- `results/fig1_distribuciones.png`
- `results/fig2_boxplots.png`
- `results/fig3_relaciones.png`

### 2️⃣ Entrenamiento de Modelos con MLflow

```bash
cd notebooks
python 02_Modelado_MLflow.py
```

**Entrena 3 modelos:**
1. Logistic Regression
2. Random Forest
3. Gradient Boosting (mejor modelo)

**Genera:**
- `models/best_model.pkl`
- `models/label_encoder.pkl`
- `results/comparacion_modelos.csv`
- `results/feature_importance.csv`
- `results/fig4_comparacion_modelos.png`
- `results/fig5_feature_importance.png`
- Matrices de confusión para cada modelo
- Logs en MLflow (`mlruns/`)

### 3️⃣ Visualizar Experimentos en MLflow

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Abre: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 4️⃣ Iniciar API REST (FastAPI)

```bash
cd api
python main.py
```

**Endpoints disponibles:**
- `GET /` - Información de la API
- `GET /health` - Health check
- `GET /tipos-procedimiento` - Lista de tipos válidos
- `POST /predict` - Predicción de excedencia DRL

**Documentación interactiva:**
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5️⃣ Probar la API

```bash
cd api
python test_api.py
```

### 6️⃣ Iniciar Dashboard (Streamlit)

```bash
cd dashboard
streamlit run app.py
```

Abre: [http://localhost:8501](http://localhost:8501)

**Funcionalidades del Dashboard:**
- Formulario de predicción con validación
- Medidor de probabilidad (gauge)
- Comparación con DRL de referencia
- Tabla de P75 por tipo de procedimiento
- Estadísticas del dataset

## 📊 Resultados del Modelo

### Mejor Modelo: Gradient Boosting Classifier

| Métrica    | Valor   |
|------------|---------|
| Accuracy   | 97.96%  |
| Precision  | 95.92%  |
| Recall     | 94.00%  |
| F1-Score   | 95.92%  |
| ROC-AUC    | 99.56%  |

### Matriz de Confusión (Test Set, n=244)

|                | Predicción: No excede | Predicción: Excede |
|----------------|----------------------|-------------------|
| **Real: No excede** | 183 (TN)          | 1 (FP)            |
| **Real: Excede**    | 3 (FN)            | 57 (TP)           |

**Total errores:** 4 de 244 predicciones (1.64%)

### Importancia de Variables

| Feature                    | Importancia |
|----------------------------|-------------|
| PKA_Gycm2                  | 39.6%       |
| Tipo_Procedimiento_encoded | 23.9%       |
| Kar_mGy                    | 15.9%       |
| Tiempo_Fluoroscopia_min    | 11.8%       |
| Peso                       | 4.7%        |
| Edad                       | 4.1%        |

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                        │
│                    (Interfaz de Usuario)                        │
│                  http://localhost:8501                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       FASTAPI REST API                          │
│                    (Capa de Lógica)                             │
│                  http://127.0.0.1:8000                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ joblib.load()
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  GRADIENT BOOSTING MODEL                        │
│                  (Scikit-learn .pkl)                            │
│                    Accuracy: 97.96%                             │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ Tracked by
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                          MLFLOW                                 │
│               (Experimentos y Registro)                         │
│                  http://127.0.0.1:5000                          │
└─────────────────────────────────────────────────────────────────┘
```

## 🧪 Ejemplo de Uso de la API

### Request (JSON)

```json
{
  "Tipo_Procedimiento": "Coronariografía Diagnóstica",
  "PKA_Gycm2": 75.5,
  "Kar_mGy": 550.0,
  "Tiempo_Fluoroscopia_min": 12.5,
  "Edad": 65,
  "Peso": 75.0
}
```

### Response (JSON)

```json
{
  "excede_DRL": 1,
  "probabilidad": 0.87,
  "clasificacion": "Excede DRL",
  "inputs": {
    "Tipo_Procedimiento": "Coronariografía Diagnóstica",
    "Tipo_Procedimiento_encoded": 0,
    "PKA_Gycm2": 75.5,
    "Kar_mGy": 550.0,
    "Tiempo_Fluoroscopia_min": 12.5,
    "Edad": 65,
    "Peso": 75.0
  }
}
```

## 📚 Referencias

1. **IAEA (2014)** - Safety Reports Series No. 75: "Establishment of Diagnostic Reference Levels"
2. **ACR (2018)** - Practice Parameter for Diagnostic Reference Levels and Achievable Doses in Medical X-Ray Imaging
3. **Balter et al. (2008)** - "Methods to Estimate Radiation Dose for Interventional Procedures" - Radiology 254(2):326-341

## 👥 Equipo

- **Universidad de los Andes**
- **Maestría en Inteligencia Artificial**
- **Febrero 2026**

## 📄 Licencia

Este proyecto es parte de un trabajo académico para la Universidad de los Andes.

## 🔗 Enlaces Útiles

- [Documentación FastAPI](https://fastapi.tiangolo.com/)
- [Documentación Streamlit](https://docs.streamlit.io/)
- [Documentación MLflow](https://www.mlflow.org/docs/latest/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**🏥 Sistema de Predicción de DRL en Angiografía**  
*Optimizando la protección radiológica mediante Machine Learning*
