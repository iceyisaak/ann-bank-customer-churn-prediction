# ANN Bank Customer Churn Prediction

##### Dataset: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

##### Repo: https://github.com/iceyisaak/ann-bank-customer-churn-prediction

##### Streamlit Dashboard: https://ann-bank-customer-churn-prediction.streamlit.app/

---

### Project Overview

This project aims to predict whether a bank customer will leave the bank (churn) or stay, based on various geo-demographical and transactional attributes. The predictive model is built using an Artificial Neural Network (ANN), a deep learning architecture that excels at finding non-linear patterns in customer behavior.

Customer retention is a critical challenge for the banking industry. It is significantly more expensive to acquire a new customer than to retain an existing one. By identifying "at-risk" customers through data analysis and predictive modeling, banks can take proactive measures (special offers, personalized outreach) to improve retention rates.

### Dataset
The project typically utilizes the Churn_Modelling.csv dataset, which includes 10,000 records with features such as:

- Credit Score: The customer's creditworthiness.
- Geography: Location (e.g., France, Spain, Germany).
- Gender: Male or Female.
- Age: Customer's age.
- Tenure: Number of years with the bank.
- Balance: Amount in the account.
- NumOfProducts: Number of bank products the customer uses.
- HasCrCard: Whether the customer has a credit card (1 = Yes, 0 = No).
- IsActiveMember: Engagement status.
- EstimatedSalary: Annual income.
- Exited: The target variable (1 = Left, 0 = Stayed).

### Tech Stack
- Language: Python 3.x
- Deep Learning: TensorFlow / Keras
- Data Manipulation: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Pre-processing: Scikit-Learn (LabelEncoder, OneHotEncoder, StandardScaler)

### Implementation Steps
- Data Preprocessing:
  - Handling categorical data (Geography and Gender) using One-Hot Encoding and Label Encoding.
  - Splitting the dataset into Training (80%) and Test (20%) sets.
  - Feature Scaling using StandardScaler to ensure the neural network converges efficiently.

- Building the ANN:
  - Initializing a Sequential model.
  - Adding Input and Hidden layers with ReLU activation.
  - Adding an Output layer with a Sigmoid activation function for binary classification.

- Training:
  - Compiling the model with the Adam optimizer and binary_crossentropy loss function.
  - Training over multiple epochs with a specified batch size.

- Evaluation:
  - Predicting results on the test set.
  - Analyzing performance via a Confusion Matrix and Accuracy Score.

### Results
The model achieves a predictive accuracy of approximately 86%, providing a robust tool for identifying potential churners based on historical data.

---
### Try on Streamlit: 
Streamlit Dashboard: https://ann-bank-customer-churn-prediction.streamlit.app/

---