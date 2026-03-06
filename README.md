# Retail Feature Engineering & Market Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A comprehensive retail analytics solution for feature engineering, temporal data processing, and sales forecasting with robust leakage prevention mechanisms

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Feature Engineering Pipeline](#feature-engineering-pipeline)
- [Data Splits & Leakage Prevention](#data-splits--leakage-prevention)
- [Transformations](#transformations)
- [Outputs](#outputs)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates advanced feature engineering techniques for retail sales analysis, focusing on:

- **Temporal data handling** with strict train-validation-test splits
- **Leakage prevention** in predictive modeling
- **Feature scaling and encoding** best practices
- **Time-series transformations** including lag features and adstock modeling

The analysis is built for real-world retail scenarios where accurate sales forecasting depends on proper feature engineering and robust validation strategies.

## ✨ Features

### Core Capabilities

- 🔧 **Advanced Feature Engineering**:log transformations, sales-per-employee metrics, categorical binning
- 📊 **Statistical Transformations**: Z-score standardization, one-hot encoding
- 🛡️ **Leakage Control**: Temporal splits with strict preprocessing boundaries
- 📈 **Time-Series Features**: Lag variables, adstock decay modeling
- 🎯 **Model-Ready Outputs**: Clean datasets optimized for ML pipelines

### Key Engineered Features

| Feature                 | Description                           | Type               |
| ----------------------- | ------------------------------------- | ------------------ |
| `log_weekly_sales`      | Log-transformed sales (analysis only) | Continuous         |
| `sales_per_employee`    | Revenue per employee (analysis only)  | Continuous         |
| `month`                 | Extracted month from date             | Categorical (1-12) |
| `day`                   | Day of week                           | Categorical (0-6)  |
| `store_rating_category` | Low/Medium/High store rating buckets  | Categorical        |
| `temperature_scaled`    | Z-score normalized temperature        | Continuous         |
| `fuel_price_scaled`     | Z-score normalized fuel price         | Continuous         |
| `cpi_scaled`            | Z-score normalized CPI                | Continuous         |
| `unemployment_scaled`   | Z-score normalized unemployment       | Continuous         |

## 📊 Dataset Description

### Source Data

- **Total Records**: 1,000 observations
- **Stores**: 20 unique retail locations
- **Time Period**: January 1, 2023 to December 29, 2024 (104 weeks)
- **Granularity**: Weekly aggregated sales data

### Raw Features

```
store_id            : String identifier for each store (Store_1 to Store_20)
week_start_date     : Start date of the week (DD-MM-YYYY format)
weekly_sales        : Target variable - total sales for the week
store_area          : Store floor area in square feet
num_employees       : Number of employees at the store
store_rating        : Store quality rating (1.0 to 10.0)
holiday_flag        : Binary indicator for holiday week (0/1)
temperature         : Average weekly temperature (°F)
fuel_price          : Average fuel price for the week
cpi                 : Consumer Price Index
unemployment        : Regional unemployment rate
```

### Data Quality

- ✅ **No missing values** in core numeric fields
- ✅ **No zero-employee rows** detected
- ✅ **Complete temporal coverage** across all stores
- ✅ **Balanced distribution** across 20 stores

## 📁 Project Structure

```
retail-feature-engineering/
│
├── retail_feature_engineering_submission.ipynb   # Main analysis notebook
├── retail_feature_engineering_assignment.csv     # Raw input data
├── retail_processed_full.csv                     # Full processed dataset
├── retail_model_ready.csv                        # ML-ready dataset
├── submission_summary.md                         # One-page project summary
├── README.md                                     # This file
└── .gitignore                                    # Git ignore rules
```

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/KishoreMuruganantham/retail-feature-engineering-analysis.git
cd retail-feature-engineering-analysis
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install pandas numpy jupyter matplotlib seaborn scikit-learn
```

Alternatively, create a `requirements.txt`:

```txt
pandas>=1.5.0
numpy>=1.23.0
jupyter>=1.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

Then install:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Analysis

1. **Launch Jupyter Notebook**

```bash
jupyter notebook retail_feature_engineering_submission.ipynb
```

2. **Execute all cells** in sequence to:
   - Load and explore the dataset
   - Apply feature engineering transformations
   - Create temporal data splits
   - Generate model-ready outputs

3. **Review outputs**:
   - `retail_processed_full.csv` - Complete processed dataset with all features
   - `retail_model_ready.csv` - Model-ready dataset (leakage features removed)

### Quick Start Example

```python
import pandas as pd

# Load processed data
df = pd.read_csv('retail_model_ready.csv')

# Inspect features
print(df.columns.tolist())
print(df.dataset_split.value_counts())

# Separate into splits
train_data = df[df['dataset_split'] == 'train']
validation_data = df[df['dataset_split'] == 'validation']
test_data = df[df['dataset_split'] == 'test']
```

## 🔧 Feature Engineering Pipeline

### 1. **Temporal Feature Extraction**

```python
# Month extraction (1-12)
df['month'] = pd.to_datetime(df['week_start_date']).dt.month

# Day of week extraction (0=Monday, 6=Sunday)
df['day'] = pd.to_datetime(df['week_start_date']).dt.dayofweek
```

### 2. **Derived Metrics**

```python
# Log transformation of target (analysis only - LEAKAGE RISK)
df['log_weekly_sales'] = np.log1p(df['weekly_sales'])

# Efficiency metric (analysis only - LEAKAGE RISK)
df['sales_per_employee'] = df['weekly_sales'] / df['num_employees']
```

### 3. **Categorical Binning**

```python
# Store rating categories
bins = [0, 3.33, 6.67, 10]
labels = ['Low', 'Medium', 'High']
df['store_rating_category'] = pd.cut(df['store_rating'], bins=bins, labels=labels)
```

### 4. **Standardization (Fit on Training Only)**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_to_scale = ['temperature', 'fuel_price', 'cpi', 'unemployment']

# Fit ONLY on training data
scaler.fit(train_data[features_to_scale])

# Transform all splits
train_scaled = scaler.transform(train_data[features_to_scale])
val_scaled = scaler.transform(validation_data[features_to_scale])
test_scaled = scaler.transform(test_data[features_to_scale])
```

### 5. **One-Hot Encoding (Categories Learned from Training)**

```python
# Encode categorical variables
categorical_features = ['holiday_flag', 'month', 'store_id', 'store_rating_category']

# Get categories from training data only
train_encoded = pd.get_dummies(train_data, columns=categorical_features)

# Apply same encoding to validation/test (align columns)
val_encoded = pd.get_dummies(validation_data, columns=categorical_features)
val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
```

### 6. **Collinearity Removal**

```python
# Remove features with |correlation| > 0.8
correlation_matrix = train_data[numeric_features].corr()
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((correlation_matrix.columns[i],
                                   correlation_matrix.columns[j]))
```

## 🛡️ Data Splits & Leakage Prevention

### Temporal Splitting Strategy

The project uses a strict **chronological split** to prevent data leakage:

```
┌───────────────────────────────────────────┬──────────────────┐
│     Development Window (2023)             │   Test (2024)    │
│  ┌─────────────────┬──────────────────┐   │                  │
│  │  Train (80%)    │  Validation(20%) │   │  Test Window     │
│  │  ~42 weeks      │  ~10 weeks       │   │  ~52 weeks       │
└──┴─────────────────┴──────────────────┴───┴──────────────────┘
   2023-01-01         2023-10-15      2023-12-31  2024-12-29
```

### Implementation

```python
# Define temporal boundaries
development_mask = df['week_start_date'] < pd.Timestamp('2024-01-01')
test_mask = df['week_start_date'] >= pd.Timestamp('2024-01-01')

# Split development into train/validation (80/20 by unique weeks)
unique_weeks_dev = sorted(df[development_mask]['week_start_date'].unique())
train_weeks = unique_weeks_dev[:int(0.8 * len(unique_weeks_dev))]

train_df = df[df['week_start_date'].isin(train_weeks)]
validation_df = df[development_mask & ~df['week_start_date'].isin(train_weeks)]
test_df = df[test_mask]
```

### Leakage Features (Excluded from Modeling)

⚠️ **These features are derived from the target variable and must NOT be used for prediction:**

1. **`log_weekly_sales`** - Direct transformation of the target
2. **`sales_per_employee`** - Directly uses `weekly_sales` in calculation

These features are:

- ✅ Included in `retail_processed_full.csv` for exploratory analysis
- ❌ **Excluded** from `retail_model_ready.csv` for predictive modeling

## 🔄 Transformations

### Adstock Transformation

Models the **decaying effect** of past advertising or events on current sales.

```python
def adstock_transform(series, decay_rate=0.5):
    """
    Apply adstock transformation with exponential decay.

    Parameters:
    - series: Input time series
    - decay_rate: Decay factor (0-1), where 0.5 = 50% retention

    Returns:
    - Transformed series with cumulative decayed effects
    """
    adstocked = [series[0]]
    for i in range(1, len(series)):
        adstocked.append(series[i] + decay_rate * adstocked[i-1])
    return adstocked
```

**Use Case**: Calculating the lasting impact of marketing campaigns on sales.

### Lag Features

Creates **shifted time-series features** for forecasting models.

```python
def create_lag_features(df, column, lags=[1, 2, 7]):
    """
    Create lagged features for time-series analysis.

    Parameters:
    - df: DataFrame with time-sorted data
    - column: Column to create lags for
    - lags: List of lag periods (e.g., [1, 2, 7] for 1, 2, and 7 weeks)

    Returns:
    - DataFrame with new lag columns
    """
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df.groupby('store_id')[column].shift(lag)
    return df
```

**Example Application**:

```python
# Create temperature lag features
df = create_lag_features(df, 'temperature', lags=[1, 2, 7])

# Now you have: temperature_lag_1, temperature_lag_2, temperature_lag_7
```

## 📤 Outputs

### 1. `retail_processed_full.csv`

**Purpose**: Complete dataset for exploratory analysis and validation

**Contents**:

- All original features
- All engineered features (including leakage features)
- Scaled continuous variables
- One-hot encoded categorical variables
- `dataset_split` column ('train', 'validation', 'test')

**Use Cases**:

- ✅ Exploratory data analysis
- ✅ Visualization and reporting
- ✅ Understanding feature relationships
- ❌ Direct use in predictive models (contains leakage features)

### 2. `retail_model_ready.csv`

**Purpose**: ML-ready dataset for predictive modeling

**Contents**:

- Safe predictor features only
- Scaled continuous variables
- One-hot encoded categorical variables
- `weekly_sales` target variable
- `dataset_split` column

**Excluded Features**:

- ❌ `log_weekly_sales` (leakage)
- ❌ `sales_per_employee` (leakage)

**Use Cases**:

- ✅ Training forecasting models
- ✅ Cross-validation experiments
- ✅ Production deployment
- ✅ Model benchmarking

### Usage Example

```python
# Load model-ready data
model_data = pd.read_csv('retail_model_ready.csv')

# Separate features and target
X = model_data.drop(['weekly_sales', 'dataset_split', 'week_start_date'], axis=1)
y = model_data['weekly_sales']

# Split by dataset_split column
X_train = X[model_data['dataset_split'] == 'train']
y_train = y[model_data['dataset_split'] == 'train']

X_val = X[model_data['dataset_split'] == 'validation']
y_val = y[model_data['dataset_split'] == 'validation']

X_test = X[model_data['dataset_split'] == 'test']
y_test = y[model_data['dataset_split'] == 'test']
```

## 🔬 Technical Implementation

### Key Libraries

- **pandas**: Data manipulation and feature engineering
- **numpy**: Numerical computations
- **scikit-learn**: Scaling, encoding, and preprocessing
- **matplotlib/seaborn**: Visualization (optional)

### Design Principles

1. **Temporal Integrity**: All preprocessing steps respect chronological order
2. **No Lookahead Bias**: Validation/test data never influences training transformations
3. **Reproducibility**: All transformations are deterministic and documented
4. **Modularity**: Each transformation step is independent and reusable
5. **Scalability**: Pipeline can handle additional stores or time periods

### Performance Considerations

- ⚡ Efficient pandas operations (vectorized where possible)
- 💾 Memory-optimized data types
- 🔄 Streamlined preprocessing pipeline
- 📊 Minimal redundant calculations

## 📈 Results

### Dataset Statistics

| Split          | Records | Weeks | Date Range               | Percentage |
| -------------- | ------- | ----- | ------------------------ | ---------- |
| **Train**      | ~620    | ~42   | 2023-01-01 to 2023-10-15 | 62%        |
| **Validation** | ~180    | ~10   | 2023-10-15 to 2023-12-31 | 18%        |
| **Test**       | ~200    | ~52   | 2024-01-01 to 2024-12-29 | 20%        |

### Feature Summary

- **Total Engineered Features**: 15+
- **One-Hot Encoded Features**: 40+ (depending on categories)
- **Scaled Continuous Features**: 4
- **Temporal Features**: 2 (month, day)
- **Model-Ready Features**: ~50-60 columns

### Validation Insights

- ✅ No missing values after preprocessing
- ✅ All categorical encodings properly aligned across splits
- ✅ Scaling parameters learned exclusively from training data
- ✅ Temporal ordering strictly preserved
- ✅ No data leakage detected in validation framework

## 🚀 Future Enhancements

### Potential Extensions

1. **Additional Features**:
   - Rolling average sales (7-day, 14-day, 30-day windows)
   - Store-specific seasonal patterns
   - Holiday proximity indicators
   - Weather patterns (categorical: cold/warm/hot)
   - Regional economic indicators

2. **Advanced Transformations**:
   - Box-Cox transformations for non-normal distributions
   - Interaction features (e.g., holiday × temperature)
   - Polynomial features for non-linear relationships
   - Principal Component Analysis (PCA) for dimensionality reduction

3. **Time-Series Enhancements**:
   - ARIMA/SARIMA features
   - Exponential smoothing components
   - Fourier terms for seasonality
   - Change-point detection

4. **Model Integration**:
   - Automated ML pipeline with scikit-learn Pipeline
   - Hyperparameter tuning framework
   - Cross-validation strategies (TimeSeriesSplit)
   - Model performance monitoring

5. **Deployment**:
   - API endpoint for real-time predictions
   - Batch prediction pipelines
   - Model retraining automation
   - Drift detection and alerts

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

### Contribution Areas

- 🐛 Bug fixes
- ✨ New feature engineering techniques
- 📚 Documentation improvements
- 🧪 Additional test cases
- 🎨 Visualization enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kishore Muruganantham**

- GitHub: [@KishoreMuruganantham](https://github.com/KishoreMuruganantham)
- Repository: [retail-feature-engineering-analysis](https://github.com/KishoreMuruganantham/retail-feature-engineering-analysis)

## 🙏 Acknowledgments

- **Springboard** - for the comprehensive data science curriculum
- **Retail Industry Partners** - for dataset structure inspiration
- **Open Source Community** - for amazing tools and libraries

## 📚 References

### Key Concepts

- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Avoiding Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)

### Tools & Libraries

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Jupyter Notebook](https://jupyter.org/)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for data science enthusiasts

</div>
