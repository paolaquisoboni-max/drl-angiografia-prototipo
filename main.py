#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
API REST CON FASTAPI - PREDICCIÓN DE EXCEDENCIA DRL
Microproyecto: Niveles de Referencia Diagnósticos en Angiografía
========================================================================
Universidad de los Andes - Maestría en IA
Febrero 2026
========================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict
import uvicorn
import os

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================
app = FastAPI(
    title="API Predicción DRL Angiografía",
    description="API REST para predecir si un procedimiento angiográfico excede el Nivel de Referencia Diagnóstico (DRL - P75)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CARGA DE MODELOS
# ============================================================================
model_path = "../models/best_model.pkl"
encoder_path = "../models/label_encoder.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")
if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"❌ LabelEncoder no encontrado: {encoder_path}")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

print("✓ Modelo cargado correctamente")
print("✓ LabelEncoder cargado correctamente")

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================
class ProcedimientoInput(BaseModel):
    """Esquema de entrada para la predicción"""
    Tipo_Procedimiento: str = Field(
        ..., 
        description="Tipo de procedimiento angiográfico",
        example="Coronariografía Diagnóstica"
    )
    PKA_Gycm2: float = Field(
        ..., 
        gt=0, 
        description="Producto Kerma-Área (Gy·cm²)",
        example=75.5
    )
    Kar_mGy: float = Field(
        ..., 
        gt=0, 
        description="Kerma en aire de referencia (mGy)",
        example=550.0
    )
    Tiempo_Fluoroscopia_min: float = Field(
        ..., 
        gt=0, 
        description="Tiempo de fluoroscopia (minutos)",
        example=12.5
    )
    Edad: int = Field(
        ..., 
        ge=18, 
        le=120, 
        description="Edad del paciente (años)",
        example=65
    )
    Peso: float = Field(
        ..., 
        gt=30, 
        lt=200, 
        description="Peso del paciente (kg)",
        example=75.0
    )

    class Config:
        schema_extra = {
            "example": {
                "Tipo_Procedimiento": "Coronariografía Diagnóstica",
                "PKA_Gycm2": 75.5,
                "Kar_mGy": 550.0,
                "Tiempo_Fluoroscopia_min": 12.5,
                "Edad": 65,
                "Peso": 75.0
            }
        }


class ProcedimientoOutput(BaseModel):
    """Esquema de salida de la predicción"""
    excede_DRL: int = Field(..., description="0: No excede DRL, 1: Excede DRL")
    probabilidad: float = Field(..., description="Probabilidad de exceder DRL (0-1)")
    clasificacion: str = Field(..., description="Dentro de DRL / Excede DRL")
    inputs: Dict = Field(..., description="Valores de entrada procesados")


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/", tags=["Health"])
async def root():
    """Endpoint raíz - Verificación de estado"""
    return {
        "mensaje": "API Predicción DRL Angiografía",
        "version": "1.0.0",
        "status": "activo",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Verificación de salud de la API"""
    return {
        "status": "healthy",
        "modelo_cargado": model is not None,
        "encoder_cargado": label_encoder is not None
    }


@app.get("/tipos-procedimiento", tags=["Información"])
async def get_tipos_procedimiento():
    """Obtiene los tipos de procedimiento válidos"""
    tipos = label_encoder.classes_.tolist()
    return {
        "tipos_disponibles": tipos,
        "cantidad": len(tipos)
    }


@app.post("/predict", response_model=ProcedimientoOutput, tags=["Predicción"])
async def predict(procedimiento: ProcedimientoInput):
    """
    Predice si un procedimiento angiográfico excede el Nivel de Referencia Diagnóstico (DRL).
    
    - **Tipo_Procedimiento**: Tipo de procedimiento (ej: "Coronariografía Diagnóstica")
    - **PKA_Gycm2**: Producto Kerma-Área en Gy·cm²
    - **Kar_mGy**: Kerma en aire de referencia en mGy
    - **Tiempo_Fluoroscopia_min**: Tiempo de fluoroscopia en minutos
    - **Edad**: Edad del paciente en años
    - **Peso**: Peso del paciente en kg
    
    Retorna:
    - **excede_DRL**: 0 (no excede) o 1 (excede)
    - **probabilidad**: Probabilidad de exceder DRL (0-1)
    - **clasificacion**: Clasificación textual
    - **inputs**: Valores procesados de entrada
    """
    try:
        # Validar tipo de procedimiento
        if procedimiento.Tipo_Procedimiento not in label_encoder.classes_:
            tipos_validos = label_encoder.classes_.tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de procedimiento inválido. Opciones válidas: {tipos_validos}"
            )
        
        # Codificar tipo de procedimiento
        tipo_encoded = label_encoder.transform([procedimiento.Tipo_Procedimiento])[0]
        
        # Crear array de features
        features = np.array([[
            procedimiento.PKA_Gycm2,
            procedimiento.Kar_mGy,
            procedimiento.Tiempo_Fluoroscopia_min,
            procedimiento.Edad,
            procedimiento.Peso,
            tipo_encoded
        ]])
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Probabilidad de clase 1 (excede)
        
        # Clasificación textual
        clasificacion = "Excede DRL" if prediction == 1 else "Dentro de DRL"
        
        # Construir respuesta
        response = {
            "excede_DRL": int(prediction),
            "probabilidad": float(probability),
            "clasificacion": clasificacion,
            "inputs": {
                "Tipo_Procedimiento": procedimiento.Tipo_Procedimiento,
                "Tipo_Procedimiento_encoded": int(tipo_encoded),
                "PKA_Gycm2": procedimiento.PKA_Gycm2,
                "Kar_mGy": procedimiento.Kar_mGy,
                "Tiempo_Fluoroscopia_min": procedimiento.Tiempo_Fluoroscopia_min,
                "Edad": procedimiento.Edad,
                "Peso": procedimiento.Peso
            }
        }
        
        return JSONResponse(content=response, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("API PREDICCIÓN DRL ANGIOGRAFÍA")
    print("="*80)
    print("✓ Servidor iniciando...")
    print("✓ Documentación: http://127.0.0.1:8000/docs")
    print("✓ Redoc: http://127.0.0.1:8000/redoc")
    print("="*80)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
