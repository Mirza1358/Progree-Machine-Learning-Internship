# Machine Learning Internship - Progree

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB1628?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

This repository contains the end-to-end projects, machine learning pipelines, and comprehensive technical documentation completed during the **One-Month Machine Learning Internship** at **Progree**.

Each task focuses on a core machine learning paradigm—ranging from regularized regression with leakage-free preprocessing, deep convolutional neural networks for computer vision, to clinical disease diagnosis on imbalanced data optimizing recall and sensitivity.

---

## 📋 Table of Contents
- [Repository Structure](#-repository-structure)
- [Task 2: Predictive Linear Regression Model Pipeline](#-task-2-predictive-linear-regression-model-pipeline)
  - [Objective & Dataset](#objective--dataset)
  - [Pipeline Architecture](#pipeline-architecture)
  - [Models & Results](#models--results)
  - [Key Takeaways](#key-takeaways)
- [Task 3: Multi-Class Image Classification using CNN](#-task-3-multi-class-image-classification-using-cnn)
  - [Objective & Dataset](#objective--dataset-1)
  - [CNN Architecture & Data Augmentation](#cnn-architecture--data-augmentation)
  - [Training & Performance](#training--performance)
  - [Key Takeaways](#key-takeaways-1)
- [Task 4: High-Imbalance Clinical Disease Predictor Pipeline](#-task-4-high-imbalance-clinical-disease-predictor-pipeline)
  - [Objective & Clinical Motivation](#objective--clinical-motivation)
  - [Methodology & Imbalance Handling](#methodology--imbalance-handling)
  - [Model Comparison & Evaluation](#model-comparison--evaluation)
  - [Decision Threshold Tuning](#decision-threshold-tuning)
  - [Key Takeaways](#key-takeaways-2)
- [🛠️ Tech Stack & Tools](#️-tech-stack--tools)
- [🚀 Getting Started](#-getting-started)
- [👤 Author](#-author)

---

## 📂 Repository Structure

```text
├── Task2/
│   ├── housing.csv                   # California Housing dataset
│   ├── task2_regression.ipynb         # End-to-end regression pipeline notebook
│   └── Task2_Report.docx             # Detailed technical report & documentation
├── Task3/
│   ├── task3_cnn_cifar10.ipynb       # Deep learning CNN classification notebook
│   └── Task3_Report.docx             # Detailed technical report & documentation
├── Task4/
│   ├── Task4-Disease-Predictor.ipynb   # Imbalanced clinical classification notebook
│   └── Task4_Report.docx             # Detailed technical report & documentation
├── .gitignore                        # Git ignore patterns for Python & Jupyter
├── requirements.txt                  # Reproducible project dependencies
└── README.md                         # Comprehensive repository documentation
```

---

## 🏠 Task 2: Predictive Linear Regression Model Pipeline

### Objective & Dataset
- **Objective:** Construct a complete, leakage-free supervised machine learning pipeline to predict median district housing values in California.
- **Dataset:** California Housing Prices (Census 1990 data, 20,640 records across 10 attributes).
  - **Features:** 8 numerical features (`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`) and 1 categorical feature (`ocean_proximity` with 5 categories).
  - **Target:** `median_house_value` (continuous numerical value).
  - **Data Challenges:** 207 missing entries in `total_bedrooms`, strong multicollinearity among room/population metrics, and a known cap on values at $500,001.

### Pipeline Architecture
To eliminate data leakage, all preprocessing steps were integrated into a unified Scikit-learn `Pipeline` via `ColumnTransformer`:
1. **Numerical Pipeline:**
   - **Imputation:** `SimpleImputer(strategy="mean")` learned strictly on training folds to fill missing values in `total_bedrooms`.
   - **Feature Scaling:** `StandardScaler()` zero-mean and unit-variance standardization.
2. **Categorical Pipeline:**
   - **Encoding:** `OneHotEncoder(handle_unknown="ignore")` converting `ocean_proximity` into binary indicator columns.
3. **Model Integration:** Preprocessor combined directly with regularized linear estimator.

### Models & Results
Evaluated on an 80/20 train/test split (16,512 training / 4,128 test samples) using $R^2$ and Root Mean Squared Error (RMSE):

| Model | Regularization ($lpha$) | $R^2$ Score | RMSE (USD) |
|---|:---:|:---:|:---:|
| **Linear Regression** | - | **0.6257** | **$70,031.49** |
| **Ridge Regression** | $lpha = 1.0$ | **0.6257** | **$70,038.41** |
| **Ridge Regression** | $lpha = 10.0$ | **0.6255** | **$70,055.89** |

### Key Takeaways
- **Strongest Predictor:** `median_income` exhibits the highest correlation ($pprox 0.688$) with median house values.
- **Location Premium:** Proximity to water (`<1H OCEAN`, `NEAR BAY`, `NEAR OCEAN`) corresponds to substantially higher valuations compared to `INLAND`.
- **Regularization Analysis:** Due to the large sample size relative to the feature count, Ridge regularization yielded comparable scores to standard OLS, effectively constraining coefficients without underfitting.

---

## 🖼️ Task 3: Multi-Class Image Classification using CNN

### Objective & Dataset
- **Objective:** Design, implement, and train a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify real-world color images into 10 distinct classes.
- **Dataset:** CIFAR-10 Benchmark Dataset.
  - **Scale:** 60,000 $32 	imes 32$ RGB images (50,000 training, 10,000 testing).
  - **Balance:** Perfectly balanced (6,000 images per class).
  - **Classes:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

### CNN Architecture & Data Augmentation
A deep convolutional architecture was engineered with **403,882 trainable parameters**:
1. **In-Graph Data Augmentation:** Built directly as input layers to prevent memorization and combat overfitting during training:
   - `RandomFlip("horizontal")`
   - `RandomRotation(0.1)`
   - `RandomZoom(0.1)`
   - `RandomTranslation(0.1, 0.1)`
2. **Feature Extraction Blocks:**
   - **Block 1:** `Conv2D(32, 3x3, ReLU)` + `BatchNormalization` $	o$ `Conv2D(32, 3x3, ReLU)` $	o$ `MaxPooling2D(2x2)` $	o$ `Dropout(0.25)`
   - **Block 2:** `Conv2D(64, 3x3, ReLU)` + `BatchNormalization` $	o$ `Conv2D(64, 3x3, ReLU)` $	o$ `MaxPooling2D(2x2)` $	o$ `Dropout(0.25)`
   - **Block 3:** `Conv2D(128, 3x3, ReLU)` + `BatchNormalization` $	o$ `MaxPooling2D(2x2)` $	o$ `Dropout(0.30)`
3. **Classification Head:**
   - `Flatten()` $	o$ `Dense(128, ReLU)` $	o$ `Dropout(0.50)` $	o$ `Dense(10, Softmax)`

### Training & Performance
- **Optimization:** Adam optimizer with `SparseCategoricalCrossentropy` loss over 30 epochs.
- **Evaluation on Unseen Test Set (10,000 images):**
  - **Test Accuracy:** **71.64%**
  - **Test Loss:** **0.8242**

| Metric | Overall Score |
|---|:---:|
| **Overall Accuracy** | **72%** |
| **Macro Average F1-Score** | **0.71** |
| **Weighted Average F1-Score** | **0.71** |

### Key Takeaways
- **Vehicle vs. Animal Discrimination:** The network achieved high precision and recall on structured vehicles (`truck`, `automobile`, `ship`, `airplane`), while exhibiting higher intra-class variance between visually similar quadrupeds (`cat` vs `dog`).
- **Stabilization:** Batch Normalization significantly accelerated convergence, and layered Dropout successfully prevented overfitting despite the complex 32x32 feature representation.

---

## 🩺 Task 4: High-Imbalance Clinical Disease Predictor Pipeline

### Objective & Clinical Motivation
- **Objective:** Develop an end-to-end, recall-focused diagnostic system to identify malignant breast cancer tumors from cytology measurements.
- **Clinical Imperative:** In clinical screening, a **False Negative (FN)**—failing to detect a malignant cancer—can be fatal. Hence, models must be explicitly optimized for **Recall** rather than raw accuracy.
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset.
  - **Size:** 569 instances, 30 real-valued cell nuclei features.
  - **Imbalance:** 357 Benign (62.7%), 212 Malignant (37.3%).

### Methodology & Imbalance Handling
1. **Leakage-Free SMOTE Pipeline:**
   - Synthetic Minority Over-sampling Technique (SMOTE) was encapsulated inside `imblearn.pipeline.Pipeline`.
   - Ensures synthetic samples are created strictly within training folds during cross-validation, completely preventing optimistic validation leakage.
2. **Alternative Approach - Class Weighting:**
   - Evaluated `class_weight="balanced"` (for SVM and Random Forest) and `scale_pos_weight` (for XGBoost) as an alternative to synthetic oversampling.
3. **Hyperparameter Tuning:**
   - 5-Fold Stratified K-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True)`) using `scoring="recall"`.
   - Grid search across kernels ($C, \gamma$), tree depth, and boosting learning rates.

### Model Comparison & Evaluation
Evaluated on an unseen stratified test set (114 patients: 72 benign, 42 malignant):

| Model | Imbalance Technique | 5-Fold CV Recall | Test Recall | Test Precision | Test F1 | Test ROC-AUC | Missed Cancers (FN) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Random Forest** | None (Default) | 0.941 | - | - | - | - | - |
| **SVM (RBF Kernel)** ★ | **SMOTE (Pipeline)** | **0.965** | **0.952** | **0.976** | **0.964** | **0.996** | **2 / 42** |
| **Random Forest** | SMOTE (Pipeline) | 0.953 | 0.929 | 1.000 | 0.963 | 0.998 | 3 / 42 |
| **XGBoost** | SMOTE (Pipeline) | 0.965 | 0.929 | 1.000 | 0.963 | 0.993 | 3 / 42 |
| **SVM (Class Weighted)** | `class_weight='balanced'` | 0.953 | - | - | - | - | - |
| **RF (Class Weighted)** | `class_weight='balanced'` | 0.947 | - | - | - | - | - |
| **XGBoost (Weighted)** | `scale_pos_weight` | 0.959 | - | - | - | - | - |

### Decision Threshold Tuning
By adjusting the classification decision threshold based on predicted probabilities from the tuned SVM model:

| Threshold | Test Recall | Test Precision | Clinical Implication |
|:---:|:---:|:---:|---|
| **0.50** | **0.976** | **0.976** | 41 / 42 Malignant cases identified (1 FN, 1 FP) |
| **0.40** | **0.976** | **0.953** | 41 / 42 Malignant cases identified (1 FN, 2 FP) |
| **0.30** | **0.976** | **0.953** | Highly conservative screening |
| **0.20** | **0.976** | **0.932** | Extreme risk-aversion, higher follow-up biopsy rate |

### Key Takeaways
- The tuned SVM model with SMOTE achieved **0.996 ROC-AUC** and **95.2% recall** (using standard decision function), correctly flagging 40 out of 42 malignant cases with only 1 false alarm.
- Utilizing calibrated probability thresholds elevated detection sensitivity to **97.6% (41 of 42 malignant cases caught)**, proving the clinical viability of threshold tuning in healthcare diagnostics.

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python 3.10+
- **Machine Learning & Preprocessing:** Scikit-Learn, Imbalanced-Learn (`imblearn`)
- **Deep Learning:** TensorFlow 2.x, Keras
- **Gradient Boosting:** XGBoost
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Environment & Execution:** Jupyter Notebook, Git

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Mirza1358/Progree-Machine-Learning-Internship.git
cd Progree-Machine-Learning-Internship
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to any of the task folders (`Task2/`, `Task3/`, or `Task4/`) and run the respective `.ipynb` file.

---

## 👤 Author

**Muhammad Bilal Mirza**
- **GitHub:** [@Mirza1358](https://github.com/Mirza1358)
- **Email:** [bilal.mirza0987@gmail.com](mailto:bilal.mirza0987@gmail.com)
- **Internship:** Machine Learning Intern at **Progree**
