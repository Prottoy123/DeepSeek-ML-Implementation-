# Thesis Machine Learning Implementation

This repository contains the machine learning codebase for analyzing user behavioral intention (BI) based on the UTAUT (Unified Theory of Acceptance and Use of Technology) framework. The analysis involves preprocessing survey data (Likert scales) and applying various classification algorithms to predict and understand the drivers of high behavioral intention (`High_BI`).

## 📁 Repository Structure

The project includes four main Jupyter Notebooks, each focusing on a distinct machine learning algorithm. Every notebook follows a comprehensive pipeline from data preprocessing to model evaluation and visualization.

### 1. Logistic Regression (`DeepSeek_LogisticRegression_Fixed.ipynb`)
- **Focus**: Interpretable probabilistic classification and statistical significance.
- **Key Features**:
  - Converts Likert scale responses to numeric values.
  - Implements Logistic Regression using `scikit-learn` and `statsmodels`.
  - Analyzes Odds Ratios and conducts statistical significance testing for UTAUT constructs.
  - Generates visualizations: Coefficient charts, Odds Ratios, ROC Curve, and Confusion Matrix.

### 2. Random Forest (`DeepSeek_RandomForest_Fixed.ipynb`)
- **Focus**: Ensemble learning and non-linear feature importance.
- **Key Features**:
  - Incorporates demographic data analysis.
  - Trains a Random Forest Classifier.
  - Extracts and visualizes Gini-based Feature Importance and aggregated Construct Importance.
  - Generates class distribution charts and comprehensive performance metrics.

### 3. Support Vector Classifier (`DeepSeek_SVG_Fixed.ipynb`)
- **Focus**: Maximum margin classification for robust decision boundaries.
- **Key Features**:
  - Utilizes `LinearSVC` with `CalibratedClassifierCV` to provide probability estimates.
  - Employs data scaling pipelines (`StandardScaler` + `make_pipeline`) for optimal SVM performance.
  - Evaluates model metrics using cross-validation and standard classification reports.

### 4. XGBoost (`DeepSeek_XGBoost_Final.ipynb`)
- **Focus**: Advanced gradient boosting for maximum predictive performance.
- **Key Features**:
  - End-to-end XGBoost model training handling unscaled categorical variables efficiently.
  - Hyperparameter tuning utilizing `GridSearchCV` and `StratifiedKFold`.
  - Detailed visualization of XGBoost-specific feature importance.
  - Comparative analysis of UTAUT construct importance.

## 🛠️ Pipeline Overview

All notebooks generally follow a standardized analytical pipeline:
1. **Data Preprocessing**: Mapping columns, handling missing values, and converting categorical Likert scale data into numerical formats.
2. **Target Engineering**: Creating the binary target variable `High_BI` (High Behavioral Intention).
3. **Feature Engineering**: Encoding categorical variables and scaling numerical features where algorithmically necessary (e.g., Logistic Regression, SVC).
4. **Model Training & Tuning**: Applying cross-validation and grid search techniques to optimize model hyperparameters.
5. **Evaluation**: Assessing models utilizing Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
6. **Professional Visualizations**: Dedicated, reproducible cells for generating publication-ready charts (e.g., Feature Importance, ROC curves).
7. **Export**: Saving the trained models, results, and generated insights.

## 💻 Tech Stack & Libraries

- **Language**: Python 3.x
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `statsmodels`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistics**: `scipy`

## 🚀 How to Use

1. Clone this repository.
2. Ensure you have the required dependencies installed (e.g., `pip install pandas numpy scikit-learn xgboost statsmodels matplotlib seaborn`).
3. Place your dataset in the corresponding directory as expected by the notebooks.
4. Run the Jupyter Notebooks sequentially or explore the specific model of interest. Each visualization is logically separated into its own cell for easy modification and execution.
