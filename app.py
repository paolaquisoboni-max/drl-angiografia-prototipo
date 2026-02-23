#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
DASHBOARD STREAMLIT - PREDICCIÓN DE EXCEDENCIA DRL
Microproyecto: Niveles de Referencia Diagnósticos en Angiografía
========================================================================
Universidad de los Andes - Maestría en IA
Febrero 2026
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="DRL Predicción - Angiografía",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS
# ============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TÍTULO
# ============================================================================
st.markdown('<h1 class="main-title">🏥 Sistema de Predicción de DRL en Angiografía</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR - INFORMACIÓN
# ============================================================================
with st.sidebar:
    st.header("ℹ️ Información del Sistema")
    st.markdown("""
    **Niveles de Referencia Diagnósticos (DRL)**
    
    Los DRL son valores de referencia (percentil 75) de dosis de radiación 
    para procedimientos médicos estándar, establecidos para optimizar la 
    protección radiológica.
    
    **¿Qué hace este sistema?**
    - Predice si un procedimiento angiográfico excede el DRL
    - Proporciona probabilidad de excedencia
    - Ayuda en la optimización de protocolos radiológicos
    
    **Modelo:** Gradient Boosting Classifier  
    **Accuracy:** 97.96%  
    **F1-Score:** 95.92%  
    **Registros de entrenamiento:** 976
    """)
    
    st.markdown("---")
    st.markdown("**📚 Universidad de los Andes**")
    st.markdown("Maestría en Inteligencia Artificial")
    st.markdown("Febrero 2026")

# ============================================================================
# CARGA DE DATOS P75
# ============================================================================
@st.cache_data
def load_p75_data():
    try:
        df_p75 = pd.read_csv('../results/P75_por_procedimiento.csv')
        return df_p75
    except:
        # Datos de respaldo
        return pd.DataFrame({
            'Tipo_Procedimiento': [
                'Angiografía Aorta Abdominal',
                'Angiografía Cerebral',
                'Angiografía Periférica',
                'Angiografía Renal',
                'Coronariografía Diagnóstica'
            ],
            'n': [126, 197, 117, 97, 439],
            'P75_PKA_Gycm2': [150.53, 135.09, 79.14, 55.66, 66.60]
        })

df_p75 = load_p75_data()

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================
def predict_drl(tipo_proc, pka, kar, tiempo, edad, peso):
    """Realiza la predicción mediante la API REST"""
    url = "http://127.0.0.1:8000/predict"
    
    payload = {
        "Tipo_Procedimiento": tipo_proc,
        "PKA_Gycm2": float(pka),
        "Kar_mGy": float(kar),
        "Tiempo_Fluoroscopia_min": float(tiempo),
        "Edad": int(edad),
        "Peso": float(peso)
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la API: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con la API. Asegúrate de que el servidor esté corriendo en http://127.0.0.1:8000")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predicción", "📊 DRLs de Referencia", "📈 Estadísticas"])

# ----------------------------------------------------------------------------
# TAB 1: PREDICCIÓN
# ----------------------------------------------------------------------------
with tab1:
    st.header("Predicción de Excedencia de DRL")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Datos del Procedimiento")
        
        # Formulario de entrada
        with st.form("prediction_form"):
            # Tipo de procedimiento
            tipos_disponibles = df_p75['Tipo_Procedimiento'].tolist()
            tipo_proc = st.selectbox(
                "Tipo de Procedimiento Angiográfico",
                options=tipos_disponibles,
                index=4  # Coronariografía por defecto
            )
            
            # Mostrar P75 de referencia
            p75_ref = df_p75[df_p75['Tipo_Procedimiento'] == tipo_proc]['P75_PKA_Gycm2'].values[0]
            st.info(f"📌 **DRL (P75) de referencia:** {p75_ref:.2f} Gy·cm²")
            
            # Variables dosimétricas
            col_a, col_b = st.columns(2)
            with col_a:
                pka = st.number_input(
                    "PKA (Gy·cm²)",
                    min_value=5.0,
                    max_value=350.0,
                    value=75.0,
                    step=5.0,
                    help="Producto Kerma-Área"
                )
                
                kar = st.number_input(
                    "Ka,r (mGy)",
                    min_value=50.0,
                    max_value=3000.0,
                    value=550.0,
                    step=50.0,
                    help="Kerma en aire de referencia"
                )
                
                tiempo = st.number_input(
                    "Tiempo de Fluoroscopia (min)",
                    min_value=1.0,
                    max_value=60.0,
                    value=12.0,
                    step=1.0
                )
            
            with col_b:
                edad = st.number_input(
                    "Edad del Paciente (años)",
                    min_value=18,
                    max_value=120,
                    value=60,
                    step=1
                )
                
                peso = st.number_input(
                    "Peso del Paciente (kg)",
                    min_value=30.0,
                    max_value=200.0,
                    value=75.0,
                    step=1.0
                )
            
            # Botón de predicción
            submit_button = st.form_submit_button(
                "🔍 Realizar Predicción",
                use_container_width=True,
                type="primary"
            )
        
        # Realizar predicción
        if submit_button:
            with st.spinner("Analizando datos..."):
                result = predict_drl(tipo_proc, pka, kar, tiempo, edad, peso)
                
                if result:
                    st.session_state['last_prediction'] = result
    
    # Mostrar resultados
    with col2:
        st.subheader("🎯 Resultado de la Predicción")
        
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']
            
            # Medidor de probabilidad
            prob = result['probabilidad'] * 100
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidad de Excedencia", 'font': {'size': 16}},
                delta={'reference': 50, 'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                    'bar': {'color': "red" if prob > 50 else "green"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgreen'},
                        {'range': [50, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Clasificación
            if result['excede_DRL'] == 1:
                st.markdown(f"""
                <div class="danger-box">
                    <h3 style="color: #721c24; margin: 0;">⚠️ EXCEDE DRL</h3>
                    <p style="margin: 5px 0;">El procedimiento supera el nivel de referencia diagnóstico.</p>
                    <p style="margin: 0;"><strong>Probabilidad:</strong> {prob:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3 style="color: #155724; margin: 0;">✅ DENTRO DE DRL</h3>
                    <p style="margin: 5px 0;">El procedimiento está dentro del nivel de referencia.</p>
                    <p style="margin: 0;"><strong>Probabilidad de exceder:</strong> {prob:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Comparación con P75
            st.markdown("---")
            st.subheader("📊 Comparación con P75")
            
            pka_input = result['inputs']['PKA_Gycm2']
            ratio = (pka_input / p75_ref) * 100
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                st.metric("PKA del Procedimiento", f"{pka_input:.2f} Gy·cm²")
            with col_comp2:
                st.metric("DRL (P75) Referencia", f"{p75_ref:.2f} Gy·cm²", 
                         delta=f"{ratio-100:.1f}%" if ratio > 100 else f"{ratio-100:.1f}%",
                         delta_color="inverse")

# ----------------------------------------------------------------------------
# TAB 2: DRLs DE REFERENCIA
# ----------------------------------------------------------------------------
with tab2:
    st.header("📊 Niveles de Referencia Diagnósticos (P75)")
    
    # Tabla de P75
    st.dataframe(
        df_p75.style.format({'P75_PKA_Gycm2': '{:.2f}'}),
        use_container_width=True,
        height=250
    )
    
    # Gráfico de barras
    fig_p75 = px.bar(
        df_p75,
        x='Tipo_Procedimiento',
        y='P75_PKA_Gycm2',
        title='Niveles de Referencia Diagnósticos (P75) por Tipo de Procedimiento',
        labels={'P75_PKA_Gycm2': 'PKA P75 (Gy·cm²)', 'Tipo_Procedimiento': 'Tipo de Procedimiento'},
        color='P75_PKA_Gycm2',
        color_continuous_scale='RdYlGn_r',
        text='P75_PKA_Gycm2'
    )
    
    fig_p75.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_p75.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
    
    st.plotly_chart(fig_p75, use_container_width=True)
    
    # Información adicional
    st.info("""
    **Interpretación:**
    - El P75 (percentil 75) representa el valor por debajo del cual se encuentra el 75% de las observaciones.
    - Valores por encima del P75 indican dosis más altas que el estándar de referencia.
    - Los DRL se utilizan para optimizar los protocolos radiológicos y garantizar la protección del paciente.
    """)

# ----------------------------------------------------------------------------
# TAB 3: ESTADÍSTICAS
# ----------------------------------------------------------------------------
with tab3:
    st.header("📈 Estadísticas del Dataset")
    
    # Cargar estadísticas si existen
    try:
        df_stats = pd.read_csv('../results/estadisticas_descriptivas.csv')
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df_stats, use_container_width=True)
    except:
        st.warning("No se pudieron cargar las estadísticas descriptivas.")
    
    # Resumen por tipo
    try:
        df_resumen = pd.read_csv('../results/resumen_por_tipo.csv')
        st.subheader("Resumen por Tipo de Procedimiento")
        st.dataframe(df_resumen, use_container_width=True)
    except:
        st.warning("No se pudo cargar el resumen por tipo de procedimiento.")
    
    # Distribución de casos
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total de Registros", "976")
    with col_stat2:
        st.metric("Casos Dentro de DRL", "731 (74.9%)")
    with col_stat3:
        st.metric("Casos que Exceden DRL", "245 (25.1%)")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>🏥 Sistema de Predicción de DRL en Angiografía | Universidad de los Andes | Maestría en IA | Febrero 2026</p>
    <p>Modelo: Gradient Boosting Classifier | Accuracy: 97.96% | F1-Score: 95.92%</p>
</div>
""", unsafe_allow_html=True)
