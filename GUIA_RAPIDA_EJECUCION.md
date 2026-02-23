# 🚀 GUÍA RÁPIDA DE EJECUCIÓN

## ⚡ Setup Rápido (5 minutos)

```bash
# 1. Navegar al directorio del proyecto
cd microproyecto_drl

# 2. Ejecutar script de setup automático
bash setup.sh

# 3. Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

## 📊 Pipeline Completo

### Paso 1: Análisis Exploratorio (2-3 minutos)

```bash
cd notebooks
python 01_EDA_Completo.py
```

**Archivos generados:**
- ✅ `results/estadisticas_descriptivas.csv`
- ✅ `results/P75_por_procedimiento.csv`
- ✅ `results/resumen_por_tipo.csv`
- ✅ `results/fig1_distribuciones.png`
- ✅ `results/fig2_boxplots.png`
- ✅ `results/fig3_relaciones.png`

### Paso 2: Entrenamiento de Modelos (3-5 minutos)

```bash
cd notebooks
python 02_Modelado_MLflow.py
```

**Archivos generados:**
- ✅ `models/best_model.pkl`
- ✅ `models/label_encoder.pkl`
- ✅ `results/comparacion_modelos.csv`
- ✅ `results/feature_importance.csv`
- ✅ `results/fig4_comparacion_modelos.png`
- ✅ `results/fig5_feature_importance.png`
- ✅ `results/confusion_matrix_*.png` (3 archivos)
- ✅ `mlruns/` (logs de MLflow)

**Resultado esperado:**
```
🏆 MEJOR MODELO: Gradient_Boosting
   • Accuracy:  0.9796 (97.96%)
   • Precision: 0.9592
   • Recall:    0.9400
   • F1-Score:  0.9592
   • ROC-AUC:   0.9956
```

### Paso 3: Visualizar MLflow (Opcional)

En una **nueva terminal**:

```bash
cd microproyecto_drl
source venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5000
```

Abrir navegador: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Paso 4: Iniciar API REST

En una **nueva terminal**:

```bash
cd microproyecto_drl
source venv/bin/activate
cd api
python main.py
```

**Verificar API:**
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

**Probar API:**

```bash
# En otra terminal
cd api
python test_api.py
```

### Paso 5: Iniciar Dashboard

En una **nueva terminal**:

```bash
cd microproyecto_drl
source venv/bin/activate
cd dashboard
streamlit run app.py
```

Abrir navegador: [http://localhost:8501](http://localhost:8501)

## 🔥 Ejecución One-Line (Todo en Segundo Plano)

**Linux/Mac:**

```bash
cd notebooks && python 01_EDA_Completo.py && python 02_Modelado_MLflow.py && cd ../api && python main.py &
cd ../dashboard && streamlit run app.py
```

## 🧪 Prueba Rápida de Predicción

### Usando curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Tipo_Procedimiento": "Coronariografía Diagnóstica",
    "PKA_Gycm2": 45.0,
    "Kar_mGy": 350.0,
    "Tiempo_Fluoroscopia_min": 8.0,
    "Edad": 55,
    "Peso": 70.0
  }'
```

### Usando Python:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "Tipo_Procedimiento": "Coronariografía Diagnóstica",
        "PKA_Gycm2": 75.0,
        "Kar_mGy": 550.0,
        "Tiempo_Fluoroscopia_min": 12.0,
        "Edad": 60,
        "Peso": 75.0
    }
)

print(response.json())
```

## 📁 Checklist de Entregables

### ✅ Código Ejecutable
- [x] `01_EDA_Completo.py`
- [x] `02_Modelado_MLflow.py`
- [x] `api/main.py`
- [x] `api/test_api.py`
- [x] `dashboard/app.py`

### ✅ Archivos de Datos
- [x] `data/angiografia_dataset_1000_clean.csv` (976 registros)
- [x] `results/P75_por_procedimiento.csv`
- [x] `results/estadisticas_descriptivas.csv`

### ✅ Modelos Entrenados
- [x] `models/best_model.pkl`
- [x] `models/label_encoder.pkl`

### ✅ Resultados y Visualizaciones
- [x] `results/fig1_distribuciones.png`
- [x] `results/fig2_boxplots.png`
- [x] `results/fig3_relaciones.png`
- [x] `results/fig4_comparacion_modelos.png`
- [x] `results/fig5_feature_importance.png`
- [x] `results/confusion_matrix_*.png` (3 archivos)
- [x] `results/comparacion_modelos.csv`
- [x] `results/feature_importance.csv`

### ✅ Documentación
- [x] `README.md`
- [x] `.gitignore`
- [x] `requirements/requirements.txt`
- [x] `setup.sh`
- [x] `GUIA_RAPIDA_EJECUCION.md`
- [x] `docs/GUIA_ARQUITECTURA_COMPLETA.md`
- [x] `docs/CHECKLIST_SUSTENTACION.md`
- [x] `docs/Informe_Final_Microproyecto_DRL.docx`

### ✅ Experimentos MLflow
- [x] `mlruns/` (generado automáticamente)

## ⚙️ Solución de Problemas

### API no responde
```bash
# Verificar si el puerto 8000 está en uso
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Matar proceso si es necesario
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Dashboard no carga
```bash
# Verificar si el puerto 8501 está en uso
lsof -i :8501  # Linux/Mac

# Matar proceso si es necesario
kill -9 <PID>
```

### Error de dependencias
```bash
pip install --upgrade pip
pip install -r requirements/requirements.txt --force-reinstall
```

### Modelos no encontrados
```bash
# Ejecutar entrenamiento
cd notebooks
python 02_Modelado_MLflow.py
```

## 📊 Métricas Esperadas

| Modelo               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~87%     | ~82%      | ~76%   | ~79%     | ~93%    |
| Random Forest       | ~96%     | ~94%      | ~91%   | ~92%     | ~98%    |
| **Gradient Boosting** | **97.96%** | **95.92%** | **94.00%** | **95.92%** | **99.56%** |

## 🎯 Estructura Final del Proyecto

```
microproyecto_drl/
├── 📂 data/
│   └── angiografia_dataset_1000_clean.csv
├── 📂 notebooks/
│   ├── 01_EDA_Completo.py
│   └── 02_Modelado_MLflow.py
├── 📂 models/
│   ├── best_model.pkl
│   └── label_encoder.pkl
├── 📂 api/
│   ├── main.py
│   └── test_api.py
├── 📂 dashboard/
│   └── app.py
├── 📂 results/ (11 archivos)
├── 📂 mlruns/ (generado)
├── 📂 requirements/
│   └── requirements.txt
├── 📂 docs/ (3 archivos)
├── README.md
├── .gitignore
├── setup.sh
└── GUIA_RAPIDA_EJECUCION.md
```

## 📞 Contacto

Universidad de los Andes - Maestría en IA - Febrero 2026

---

**¡Todo listo para la ejecución y sustentación! 🚀**
