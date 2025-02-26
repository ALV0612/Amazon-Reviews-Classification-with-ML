# Amazon Reviews Classification Project

## Overview

This project aims to classify Amazon product reviews as either "spam" or "ham" (legitimate) using machine learning techniques. The dataset, `amazon_reviews.csv`, contains 5002 reviews labeled as spam (1) or ham (0). The project involves data cleaning, exploratory data analysis (EDA), text preprocessing, feature extraction, and model training with hyperparameter tuning to achieve the best classification performance.

## Dataset

- **File**: `amazon_reviews.csv`
- **Size**: 5002 rows, 2 columns (`Reviews`, `label`)
- **Labels**: 
  - `0`: Ham (legitimate reviews) - 3356 instances
  - `1`: Spam - 1646 instances
- **Format**: CSV

## Dependencies

- Python 3.11.9
- Libraries:
  - `numpy`
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `scikit-learn`
  - `spacy` (`en_core_web_sm` model)
  - `imblearn` (for SMOTE)
  - `lazypredict`
  - `optuna`

Install dependencies using:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn spacy imblearn lazypredict optuna
python -m spacy download en_core_web_sm

## Project Workflow

### 1. Data Cleaning
- Imported necessary libraries: `numpy`, `pandas`, `seaborn`, etc.
- Loaded the dataset and checked for missing values (none found).
- Removed duplicates (if any) using `drop_duplicates`.
- Encoded labels using `LabelEncoder`: "ham" → 0, "spam" → 1.

### 2. Exploratory Data Analysis (EDA)
- Analyzed label distribution: 67.09% ham, 32.91% spam.
- Visualized the distribution using a pie chart with `matplotlib`.

### 3. Text Preprocessing
- Used `spacy` (`en_core_web_sm`) for natural language processing.
- Preprocessing steps:
  - Converted text to lowercase.
  - Removed punctuation using regex.
  - Lemmatized words and removed stop words.
- Created a new column `clean_review` with processed text.

### 4. Feature Extraction
- Applied `TfidfVectorizer` to convert text into numerical features (max 5000 features).
- Resulting shape after vectorization: `(5002, 4863)`.

### 5. Data Preparation
- Split data into training (80%) and testing (20%) sets using `train_test_split`.
- Standardized features using `StandardScaler` (with `with_mean=False` for sparse matrices).
- Addressed class imbalance using `SMOTE` oversampling on the training set.

### 6. Model Selection
- Used `LazyClassifier` to evaluate 30 classification models.
- Top performer: `RandomForestClassifier` with an accuracy of 0.72.

### 7. Hyperparameter Tuning
#### Using Optuna
- Optimized `RandomForestClassifier` with 20 trials.
- Best parameters: 
  - `n_estimators`: 329
  - `max_depth`: 45
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
- Accuracy on test set: 0.7173

#### Using GridSearchCV
- Searched parameter grid: 
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [10, 20, None]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
- Best parameters:
  - `n_estimators`: 100
  - `max_depth`: None
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
- Results on test set:
  - Accuracy: 0.7183
  - F1-score: 0.7095

