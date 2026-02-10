# Workshop 3: Analysis and Experimentation with Convolutional Layers

## 📋 Context and Motivation

In this course, neural networks are not treated as black boxes but as **architectural components** whose design choices affect performance, scalability, and interpretability.

This workshop focuses on **convolutional layers** as a concrete example of how inductive bias is introduced into learning systems.

Rather than following a recipe, students will **select, analyze, and experiment** with a convolutional architecture using a real dataset.

---

## 🎯 Learning Objectives

By completing this workshop, you should be able to:

1. **Understand** the role and mathematical intuition behind convolutional layers
2. **Analyze** how architectural decisions (kernel size, depth, stride, padding) affect learning
3. **Compare** convolutional layers with fully connected layers for image-like data
4. **Perform** a minimal but meaningful exploratory data analysis (EDA) for NN tasks
5. **Communicate** architectural and experimental decisions clearly

---

## 🗂️ Project Structure

```
Taller03/
├── README.md                 # This file
├── data/                     # Project data
│   ├── raw/                 # Original data
│   └── processed/           # Processed data
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_baseline.ipynb   # Baseline model (fully connected)
│   └── 03_cnn.ipynb        # Convolutional model
├── src/                     # Source code
│   ├── models/             # Model definitions
│   ├── utils/              # Utility functions
│   └── training/           # Training scripts
├── results/                 # Results and visualizations
│   ├── figures/            # Plots and visualizations
│   └── models/             # Trained models
└── requirements.txt         # Project dependencies
```

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Taller03

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

- `torch` / `tensorflow` - Deep learning framework
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` / `seaborn` - Visualization
- `scikit-learn` - Metrics and preprocessing
- `jupyter` - Interactive notebooks

---

## 📊 Dataset

### Description

**CIFAR-10** (Krizhevsky et al.):
- 60,000 RGB images of 32x32
- 10 balanced classes (50,000 train / 10,000 test)
- Suitable for CNNs due to spatial structure and local patterns

### Data Preparation

The dataset is downloaded automatically from the notebooks using `get_cifar10_loaders`.

---

## 🧪 Experimentation

### 1. Exploratory Data Analysis (EDA)

Before building any model, it's crucial to understand the data:

- Class distribution
- Image dimensions
- Pixel statistics (mean, standard deviation)
- Examples from each class
- Potential imbalances or biases

**Notebook**: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)

**Key results (EDA):**
- Class distribution: balanced
- Image size: 32x32x3
- Pixel statistics: see tables and plots in the notebook

### 2. Baseline Model (Fully Connected)

Implementation of a fully connected neural network as baseline:

- Simple architecture: Input → Flatten → Dense Layers → Output
- Serves as comparison point
- Analyze limitations for image-like data

**Notebook**: [notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb)

**Architecture (summary):**
```
Input (3x32x32) -> Flatten (3072)
   -> Dense(512) -> ReLU -> Dropout(0.2)
   -> Dense(256) -> ReLU -> Dropout(0.2)
   -> Dense(10)
```

**Key results (Baseline):**
- Parameters: ~1,707,274
- Accuracy/curves: see notebook outputs

### 3. Convolutional Model (CNN)

Design and experimentation with convolutional layers:

#### Parameters to Experiment With:

- **Kernel Size**: 3x3, 5x5, 7x7
- **Depth**: Number of convolutional layers
- **Stride**: 1, 2 (affects downsampling)
- **Padding**: 'same', 'valid'
- **Pooling**: MaxPooling, AveragePooling
- **Activation Functions**: ReLU, LeakyReLU, etc.

**Notebook**: [notebooks/03_cnn.ipynb](notebooks/03_cnn.ipynb)

**Base architecture (simple diagram):**
```
Input (3x32x32)
   -> Conv(32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
   -> Conv(64, 3x3) -> BN -> ReLU -> MaxPool(2x2)
   -> Flatten (4096)
   -> Dense(128) -> ReLU -> Dropout(0.5)
   -> Dense(10)
```

**Key results (CNN base):**
- Parameters: ~545,290
- Accuracy/curves: see notebook outputs

### 4. Results Comparison

| Model | Parameters | Accuracy | Notes |
|--------|-----------|----------|---------------|
| Baseline FC | ~1,707,274 | TBD | See baseline notebook |
| CNN base | ~545,290 | TBD | Better parameter efficiency |
| CNN (experiments) | Variable | TBD | Kernel, depth, pooling |

Figures saved in [results/figures/](results/figures/)

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall model accuracy
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: Error visualization
- **Training/Validation Loss**: Learning curves
- **Number of Parameters**: Model complexity
- **Training Time**: Computational efficiency

---

## 🧠 Key Concepts

### Why Convolutional Layers?

Convolutional layers introduce **inductive bias** by assuming:

1. **Locality**: Nearby pixels are related
2. **Translation Equivariance**: A pattern is relevant regardless of its position
3. **Parameter Sharing**: Dramatically reduces the number of parameters

### Mathematical Intuition

A convolutional layer applies a filter (kernel) over the image:

$$
(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

Where:
- $I$ is the input image
- $K$ is the kernel/filter
- $*$ denotes the convolution operation

---

## 📝 Deliverables

### Required Documentation

1. **Notebooks with complete analysis**:
   - Detailed EDA with visualizations
   - Baseline model implementation and justification
   - Experimentation with convolutional architectures
   - Results comparison

2. **Architectural decisions report**:
   - Why did you choose certain hyperparameter values?
   - What did you observe when changing kernel size, depth, stride?
   - How does the CNN compare with the baseline?

3. **Visualizations**:
   - Learning curves
   - Confusion matrix
   - Examples of correct and incorrect predictions
   - Visualization of learned filters (optional)

---

## 🔍 Guiding Questions for Analysis

### About Architecture

- How does kernel size affect the patterns the network can detect?
- What happens to spatial dimensions as depth increases?
- What is the trade-off between network depth and width?

### About Performance

- Why does the CNN outperform (or not) the fully connected model?
- Where does the model fail? Are there classes it systematically confuses?
- Is the model overfitting? How can you detect it?

### About Efficiency

- How many parameters does each model have?
- How does the number of parameters relate to performance?
- Is the additional computational cost worth it?

---

## 🛠️ Usage

### Train a Model

```bash
# Train baseline model
python src/training/train_baseline.py --epochs 50 --batch_size 32

# Train CNN model
python src/training/train_cnn.py --epochs 50 --batch_size 32 --kernel_size 3
```

### Evaluate a Model

```bash
python src/evaluation/evaluate.py --model_path results/models/best_cnn.pth
```

---

## 📚 References and Resources

---

## 🧠 Interpretation and Architectural Reasoning (In your own words)

**1) Why did convolutional layers outperform (or not) the baseline?**
CNNs exploit spatial structure: they detect local patterns (edges, textures) and reuse filters across the image. This reduces parameters, improves generalization, and learns feature hierarchies. The baseline flattens the image and loses spatial relationships, so it needs far more parameters to capture the same patterns.

**2) What inductive bias does convolution introduce?**
It introduces **locality** (nearby pixels are related) and **translation equivariance** (a pattern matters regardless of position). It also encourages **hierarchical** features from simple to complex.

**3) In what types of problems would convolution NOT be appropriate?**
In data without spatial structure (tabular), in tasks where absolute position is critical, or in irregular structures (graphs, point clouds). If global context or non-local relationships are needed, other models (transformers, GNNs) may fit better.

### Fundamental Papers

- LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
- Krizhevsky et al. (2012) - "ImageNet Classification with Deep CNNs (AlexNet)"
- Simonyan & Zisserman (2014) - "Very Deep CNNs (VGGNet)"

### Additional Resources

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning Book - Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [Distill.pub - Feature Visualization](https://distill.pub/2017/feature-visualization/)

---
