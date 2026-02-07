# Prototipo basado en aprendizaje automático para el análisis de Niveles de Referencia Diagnósticos en Angiografía

## Descripción
Este repositorio contiene el desarrollo de un microproyecto académico orientado al análisis de registros dosimétricos retrospectivos de procedimientos de angiografía, con el fin de establecer Niveles de Referencia Diagnósticos (DRL) locales y explorar el uso de modelos de aprendizaje automático como apoyo a la optimización de la protección radiológica.

El proyecto se enmarca en las recomendaciones de la ICRP Publicación 135 y sigue un enfoque de prototipo analítico.

## Objetivo
Desarrollar un prototipo que permita:
- Analizar indicadores dosimétricos (DAP/PKA, Ka,r, tiempo de fluoroscopia).
- Establecer DRL locales usando el percentil 75 (P75).
- Explorar un modelo supervisado para predecir el riesgo de exceder el DRL.

## Estructura del repositorio
- data/: conjuntos de datos (gestionados con DVC).
- src/: scripts de procesamiento y modelado.
- notebooks/: análisis exploratorio y visualización.
- docs/: documentación y reportes del proyecto.

## Estado del proyecto
Proyecto en desarrollo – etapa de exploración y diseño del prototipo.
