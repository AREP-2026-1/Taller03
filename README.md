# Taller 3: Análisis y Experimentación con Capas Convolucionales

## 📋 Contexto y Motivación

En este curso, las redes neuronales no se tratan como cajas negras sino como **componentes arquitectónicos** cuyas decisiones de diseño afectan el rendimiento, la escalabilidad y la interpretabilidad.

Este taller se enfoca en **capas convolucionales** como un ejemplo concreto de cómo el sesgo inductivo se introduce en los sistemas de aprendizaje.

En lugar de seguir una receta, **seleccionarás, analizarás y experimentarás** con una arquitectura convolucional usando un dataset real.

---

## 🎯 Objetivos de Aprendizaje

Al completar este taller, deberías ser capaz de:

1. **Entender** el rol y la intuición matemática detrás de las capas convolucionales
2. **Analizar** cómo las decisiones arquitectónicas (kernel size, depth, stride, padding) afectan el aprendizaje
3. **Comparar** capas convolucionales con capas completamente conectadas para datos tipo imagen
4. **Realizar** un análisis exploratorio de datos (EDA) mínimo pero significativo para tareas de redes neuronales
5. **Comunicar** decisiones arquitectónicas y experimentales de manera clara

---

## 🗂️ Estructura del Proyecto

```
Taller03/
├── README.md                 # Este archivo
├── data/                     # Datos del proyecto
│   ├── raw/                 # Datos originales
│   └── processed/           # Datos procesados
├── notebooks/               # Notebooks de Jupyter para análisis
│   ├── 01_eda.ipynb        # Análisis exploratorio de datos
│   ├── 02_baseline.ipynb   # Modelo baseline (fully connected)
│   └── 03_cnn.ipynb        # Modelo convolucional
├── src/                     # Código fuente
│   ├── models/             # Definiciones de modelos
│   ├── utils/              # Funciones auxiliares
│   └── training/           # Scripts de entrenamiento
├── results/                 # Resultados y visualizaciones
│   ├── figures/            # Gráficos y visualizaciones
│   └── models/             # Modelos entrenados
└── requirements.txt         # Dependencias del proyecto
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- pip o conda para gestión de paquetes

### Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd Taller03

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

- `torch` / `tensorflow` - Framework de deep learning
- `numpy` - Operaciones numéricas
- `pandas` - Manipulación de datos
- `matplotlib` / `seaborn` - Visualización
- `scikit-learn` - Métricas y preprocesamiento
- `jupyter` - Notebooks interactivos

---

## 📊 Dataset

### Descripción

[Describe aquí el dataset que utilizarás: nombre, fuente, número de clases, tamaño de las imágenes, número de ejemplos, etc.]

### Preparación de Datos

```bash
# Descargar y preparar el dataset
python src/data/prepare_data.py
```

---

## 🧪 Experimentación

### 1. Análisis Exploratorio de Datos (EDA)

Antes de construir cualquier modelo, es crucial entender los datos:

- Distribución de clases
- Dimensiones de las imágenes
- Estadísticas de píxeles (media, desviación estándar)
- Ejemplos de cada clase
- Posibles desbalances o sesgos

**Notebook**: `notebooks/01_eda.ipynb`

### 2. Modelo Baseline (Fully Connected)

Implementación de una red neuronal completamente conectada como línea base:

- Arquitectura simple: Input → Flatten → Dense Layers → Output
- Sirve como punto de comparación
- Analiza limitaciones para datos tipo imagen

**Notebook**: `notebooks/02_baseline.ipynb`

### 3. Modelo Convolucional (CNN)

Diseño y experimentación con capas convolucionales:

#### Parámetros a Experimentar:

- **Kernel Size**: 3x3, 5x5, 7x7
- **Depth**: Número de capas convolucionales
- **Stride**: 1, 2 (afecta el downsampling)
- **Padding**: 'same', 'valid'
- **Pooling**: MaxPooling, AveragePooling
- **Activation Functions**: ReLU, LeakyReLU, etc.

**Notebook**: `notebooks/03_cnn.ipynb`

### 4. Comparación de Resultados

| Modelo | Parámetros | Accuracy | F1-Score | Tiempo Entrenamiento |
|--------|-----------|----------|----------|---------------------|
| Baseline FC | X | X% | X | X min |
| CNN v1 | Y | Y% | Y | Y min |
| CNN v2 | Z | Z% | Z | Z min |

---

## 📈 Métricas de Evaluación

- **Accuracy**: Precisión general del modelo
- **Precision, Recall, F1-Score**: Métricas por clase
- **Confusion Matrix**: Visualización de errores
- **Training/Validation Loss**: Curvas de aprendizaje
- **Número de Parámetros**: Complejidad del modelo
- **Tiempo de Entrenamiento**: Eficiencia computacional

---

## 🧠 Conceptos Clave

### ¿Por qué Capas Convolucionales?

Las capas convolucionales introducen **sesgo inductivo** al asumir que:

1. **Localidad**: Los píxeles cercanos están relacionados
2. **Equivariancia Traslacional**: Un patrón es relevante sin importar su posición
3. **Compartición de Parámetros**: Reduce drásticamente el número de parámetros

### Intuición Matemática

Una capa convolucional aplica un filtro (kernel) sobre la imagen:

$$
(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

Donde:
- $I$ es la imagen de entrada
- $K$ es el kernel/filtro
- $*$ denota la operación de convolución

---

## 📝 Entregables

### Documentación Requerida

1. **Notebooks con análisis completo**:
   - EDA detallado con visualizaciones
   - Implementación y justificación del modelo baseline
   - Experimentación con arquitecturas convolucionales
   - Comparación de resultados

2. **Reporte de decisiones arquitectónicas**:
   - ¿Por qué elegiste ciertos valores de hiperparámetros?
   - ¿Qué observaste al cambiar kernel size, depth, stride?
   - ¿Cómo se compara la CNN con el baseline?

3. **Visualizaciones**:
   - Curvas de aprendizaje
   - Matriz de confusión
   - Ejemplos de predicciones correctas e incorrectas
   - Visualización de filtros aprendidos (opcional)

---

## 🔍 Preguntas Guía para el Análisis

### Sobre la Arquitectura

- ¿Cómo afecta el tamaño del kernel a los patrones que puede detectar la red?
- ¿Qué sucede con la dimensión espacial a medida que aumenta la profundidad?
- ¿Cuál es el trade-off entre profundidad y ancho de la red?

### Sobre el Rendimiento

- ¿Por qué la CNN supera (o no) al modelo fully connected?
- ¿Dónde falla el modelo? ¿Hay clases que confunde sistemáticamente?
- ¿El modelo está sobreajustando? ¿Cómo lo detectas?

### Sobre la Eficiencia

- ¿Cuántos parámetros tiene cada modelo?
- ¿Cómo se relaciona el número de parámetros con el rendimiento?
- ¿Vale la pena el costo computacional adicional?

---

## 🛠️ Uso

### Entrenar un Modelo

```bash
# Entrenar modelo baseline
python src/training/train_baseline.py --epochs 50 --batch_size 32

# Entrenar modelo CNN
python src/training/train_cnn.py --epochs 50 --batch_size 32 --kernel_size 3
```

### Evaluar un Modelo

```bash
python src/evaluation/evaluate.py --model_path results/models/best_cnn.pth
```

---

## 📚 Referencias y Recursos

### Papers Fundamentales

- LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
- Krizhevsky et al. (2012) - "ImageNet Classification with Deep CNNs (AlexNet)"
- Simonyan & Zisserman (2014) - "Very Deep CNNs (VGGNet)"

### Recursos Adicionales

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning Book - Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [Distill.pub - Feature Visualization](https://distill.pub/2017/feature-visualization/)

---

## 👥 Autor

[Tu nombre]

---

## 📄 Licencia

[Especifica la licencia si aplica]

---

## 🤝 Contribuciones

Si tienes sugerencias o encuentras errores, no dudes en abrir un issue o pull request.

---

**Última actualización**: Febrero 2026
