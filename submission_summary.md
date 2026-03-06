# One-Page Summary

## Dataset Overview

The retail dataset contains 1,000 rows across 20 stores, with weekly observations spanning 2023-01-01 through 2024-12-29. The core numeric fields are complete, and the sample inspection showed no zero-employee rows, which reduces risk for the sales-per-employee calculation.

## Feature Engineering

The notebook creates the required engineered fields: `log_weekly_sales`, `sales_per_employee`, `month`, `day`, and the categorical `store_rating_category` buckets (`Low`, `Medium`, `High`). It also applies z-score scaling to `temperature`, `fuel_price`, `cpi`, and `unemployment`, and one-hot encodes `holiday_flag` and `month`. For a more practical modeling dataset, the notebook also one-hot encodes `store_id` and `store_rating_category` using categories learned only from the training split.

## Leakage Control and Validation Strategy

The pipeline uses a strict temporal boundary to avoid leakage: all rows before 2024 form the development window, while rows from 2024 onward are reserved for testing. The development window is further split chronologically into an 80% training subset and a 20% validation subset based on unique weeks. Any learned preprocessing step, including missing-value imputation, z-score scaling, and one-hot category discovery, is fit only on the training subset and then applied unchanged to validation and test data.

Two engineered variables are explicitly treated as leakage features for predictive modeling: `log_weekly_sales` and `sales_per_employee`. Both are derived directly from the target `weekly_sales`, so they are useful for analysis but must not be used as predictors in a real sales forecasting model.

## Collinearity and Outputs

The notebook computes a correlation matrix on the continuous model-candidate features and drops any feature whose absolute correlation exceeds 0.8 according to the assignment threshold. It saves two outputs when run: a full processed dataset for traceability and a model-ready dataset that excludes direct leakage features and keeps the leakage-safe encodings and scaled variables.

## Additional Transformations

The notebook also implements and tests two standalone transformation utilities requested in the brief. The adstock transformation demonstrates how past advertising spend retains a decaying influence over time, while the lag transformation creates shifted temperature features such as `Lag_1`, `Lag_2`, and `Lag_7` for time-series forecasting use cases.
