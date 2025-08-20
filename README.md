# TelecomXProyecto2AluraLatamAntonioLopez
Challenge Alura Latam TelecomX Parte 2
# TelecomX - Parte 2: Predicción de Churn de Clientes

##  Propósito del Análisis

Este proyecto tiene como objetivo principal **predecir el churn (cancelación) de clientes** de una empresa de telecomunicaciones utilizando técnicas de machine learning. A través del análisis de variables demográficas, de servicios contratados y comportamiento de uso, buscamos identificar patrones que permitan anticipar qué clientes tienen mayor probabilidad de cancelar sus servicios.

La predicción temprana del churn permite a la empresa:
- Implementar estrategias de retención dirigidas
- Reducir costos de adquisición de nuevos clientes
- Mejorar la experiencia del cliente
- Optimizar recursos y campañas de marketing

## Estructura del Proyecto

```
TelecomXProyecto2/
│
├── README.md                     # Este archivo
├── TelecomX_Churn_Analysis.ipynb # Cuaderno principal con el análisis completo
├── data/
│   ├── raw/
│   │   └── telecom_data.csv      # Datos originales
│   └── processed/
│       ├── telecom_clean.csv     # Datos procesados y limpios
│       └── telecom_encoded.csv   # Datos codificados para modelado
├── visualizations/
│   ├── eda_plots/                # Gráficos del análisis exploratorio
│   ├── model_performance/        # Visualizaciones de rendimiento del modelo
│   └── feature_importance/       # Importancia de características
└── models/
    └── trained_models/           # Modelos entrenados guardados
```

##  Proceso de Preparación de Datos

### Clasificación de Variables

**Variables Categóricas:**
- `gender`: Género del cliente
- `Partner`: Si tiene pareja
- `Dependents`: Si tiene dependientes
- `PhoneService`: Servicio telefónico
- `MultipleLines`: Múltiples líneas telefónicas
- `InternetService`: Tipo de servicio de internet
- `OnlineSecurity`: Seguridad online
- `OnlineBackup`: Respaldo online
- `DeviceProtection`: Protección de dispositivos
- `TechSupport`: Soporte técnico
- `StreamingTV`: Streaming de TV
- `StreamingMovies`: Streaming de películas
- `Contract`: Tipo de contrato
- `PaperlessBilling`: Facturación sin papel
- `PaymentMethod`: Método de pago

**Variables Numéricas:**
- `SeniorCitizen`: Cliente de la tercera edad (0/1)
- `tenure`: Meses como cliente
- `MonthlyCharges`: Cargos mensuales
- `TotalCharges`: Cargos totales

**Variable Objetivo:**
- `Churn`: Si el cliente canceló (Yes/No)

### Etapas de Procesamiento

1. **Limpieza de Datos:**
   - Identificación y tratamiento de valores nulos
   - Conversión de tipos de datos apropiados
   - Corrección de inconsistencias en `TotalCharges`

2. **Análisis Exploratorio de Datos (EDA):**
   - Distribución de variables categóricas y numéricas
   - Análisis de correlaciones
   - Identificación de patrones de churn por segmentos

3. **Codificación de Variables:**
   - **Label Encoding** para variables binarias (gender, Partner, Dependents, etc.)
   - **One-Hot Encoding** para variables categóricas con múltiples categorías (Contract, PaymentMethod, InternetService)
   - **Normalización Min-Max** para variables numéricas

4. **División de Datos:**
   - **Entrenamiento**: 80% de los datos
   - **Prueba**: 20% de los datos
   - Estratificación por variable objetivo para mantener proporciones

### Justificaciones de Modelización

- **Codificación mixta**: Se utilizó Label Encoding para variables binarias por eficiencia computacional y One-Hot para categóricas múltiples para evitar ordenamientos artificiales.
- **Normalización**: Se aplicó Min-Max scaling para que todas las variables numéricas tengan la misma escala (0-1).
- **Estratificación**: Se mantuvo la proporción de churn en ambos conjuntos para evitar sesgo en el entrenamiento.

##  Insights del Análisis Exploratorio

### Hallazgos Principales:

1. **Distribución de Churn**: ~26.5% de los clientes cancelan sus servicios
2. **Impacto del Contrato**: Clientes con contratos mensuales tienen mayor tasa de churn (42%) vs. anuales (11%)
3. **Servicios Adicionales**: Clientes sin servicios de seguridad online o soporte técnico presentan mayor churn
4. **Tenure**: Clientes nuevos (< 12 meses) tienen mayor probabilidad de cancelación
5. **Método de Pago**: Pagos electrónicos automáticos se asocian con menor churn

### Ejemplos de Visualizaciones Generadas:

- **Distribución de Churn por Tipo de Contrato**: Gráfico de barras mostrando diferencias significativas
- **Correlación de Variables Numéricas**: Heatmap identificando relaciones clave
- **Distribución de Tenure por Churn**: Histograma revelando patrones temporales
- **Análisis de Servicios Adicionales**: Gráficos apilados por categoría de servicio

##  Instrucciones de Ejecución

### Prerrequisitos

Instalar las siguientes bibliotecas:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyter
```

### Bibliotecas Utilizadas:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import warnings
```

### Pasos para Ejecutar:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Athonlive/TelecomXProyecto2AluraLatamAntonioLopez.git
   cd TelecomXProyecto2AluraLatamAntonioLopez
   ```

2. **Abrir el cuaderno:**
   ```bash
   jupyter notebook TelecomX_Churn_Analysis.ipynb
   ```

3. **Ejecutar las celdas secuencialmente:**
   - El cuaderno está organizado en secciones claras
   - Los datos procesados se guardan automáticamente en `data/processed/`
   - Las visualizaciones se generan y guardan en `visualizations/`

### Datos de Entrada:

- **Archivo principal**: `data/raw/telecom_data.csv`
- **Formato**: CSV con headers
- **Tamaño**: ~7,000 registros con 21 columnas

### Resultados Esperados:

Al ejecutar el cuaderno completo obtendrás:
- Datos limpios y procesados
- Visualizaciones del EDA
- Modelos entrenados de machine learning
- Métricas de rendimiento y evaluación
- Recomendaciones basadas en los hallazgos

---

## 👤 Autor

**Antonio López** - Challenge Alura Latam TelecomX

Para preguntas o sugerencias, no dudes en abrir un issue en el repositorio.

---

*Este proyecto forma parte del Challenge de Alura Latam enfocado en análisis de datos y machine learning aplicado al sector de telecomunicaciones.*
