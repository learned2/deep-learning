# deep-learning
# Alphabet Soup Charity Success Predictor

## Overview

This project leverages supervised machine learning to help the nonprofit foundation Alphabet Soup identify which funding applicants are most likely to succeed. Using historical data on over 34,000 previously funded organizations, the goal was to create a binary classification model that predicts whether an applicant will be successful (`IS_SUCCESSFUL = 1`) or not (`IS_SUCCESSFUL = 0`).

The initial approach used a deep neural network (DNN) built with TensorFlow and Keras. However, after experimentation and evaluation, the project pivoted to using XGBoost, a gradient-boosted decision tree algorithm, to improve performance on this structured dataset.

---

## Data Preprocessing

- **Target Variable**: `IS_SUCCESSFUL`
- **Features**: All other columns excluding `EIN` and `NAME`
- **Dropped Columns**: 
  - `EIN`, `NAME` (identifiers with no predictive value)
- **Categorical Grouping**: 
  - Rare values in `APPLICATION_TYPE` and `CLASSIFICATION` columns were grouped under `'Other'` to reduce dimensionality
- **Encoding**: All categorical features were one-hot encoded using `pd.get_dummies()`
- **Scaling**: Numerical features were scaled using `StandardScaler`

---

## Model Development

### Deep Learning Approach

A baseline neural network was created with:

- 2 hidden layers using ReLU activation
- 1 output layer using sigmoid activation
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Epochs: 50–100

Despite tuning the model architecture and increasing training epochs, the highest accuracy achieved was **~60%**, far below the target threshold of **75%**.

### XGBoost Approach

Given the limited performance of the DNN on this tabular dataset, XGBoost was implemented:

- XGBoost is highly optimized for structured/tabular data
- The model required minimal hyperparameter tuning
- Accuracy improved significantly, achieving over **75%** consistently on the test set

This experience demonstrated that traditional ensemble methods like XGBoost can outperform deep learning on structured datasets where nonlinear relationships and decision boundaries can be captured more efficiently by tree-based models.

---

## Model Evaluation

### Final XGBoost Results

- **Accuracy**: >75%
- **Loss Function**: log loss
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score

The XGBoost model provided more stable and interpretable results compared to the neural network, with less overfitting and higher generalization on unseen data.

---

## Conclusion

This project was a strong reminder that **the best algorithm depends on the data**. While deep learning is powerful, simpler models like XGBoost often excel with structured, tabular datasets. XGBoost delivered not only a higher accuracy score but also faster training and more consistent results.

---

## Technologies Used

- Python
- Pandas, NumPy, scikit-learn
- TensorFlow / Keras
- XGBoost


