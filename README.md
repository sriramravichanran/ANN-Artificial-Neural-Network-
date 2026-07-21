# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project demonstrates how an **Artificial Neural Network (ANN)** can be used to predict whether a customer is likely to leave a bank (Customer Churn Prediction).

The project covers the complete deep learning workflow, including data preprocessing, feature engineering, model building, training, evaluation, and saving the trained model for future predictions.

---

## 🚀 Features

* Data preprocessing
* Label Encoding
* One-Hot Encoding
* Train-Test Split
* Feature Scaling using StandardScaler
* ANN Model using TensorFlow & Keras
* Binary Classification
* Model Saving (.h5)
* Scaler Saving (.pkl)

---

## 📂 Project Structure

```
customer-churn-ann-proj/
│
├── data/
│   └── Churn_Modelling.csv
│
├── models/
│   ├── ann_model.h5
│   └── scaler.pkl
│
├── notebook/
│   └── experimentation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary

Target Variable:

* **Exited**

  * 1 → Customer Left
  * 0 → Customer Stayed

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns
* Label Encoding for Gender
* One-Hot Encoding for Geography
* Train-Test Split
* StandardScaler for feature scaling

---

## 🧠 ANN Architecture

The neural network consists of:

* Input Layer
* Hidden Layer (16 neurons, ReLU)
* Hidden Layer (8 neurons, ReLU)
* Output Layer (1 neuron, Sigmoid)

---

## 🔧 Model Configuration

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Evaluation Metric: Accuracy
* Epochs: 10
* Batch Size: 32

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow
* Keras
* Joblib

---

## ▶️ How to Run

### Clone the Repository

```bash
git clone <repository-link>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

### Make Predictions

```bash
python src/predict.py
```

---

## 📈 Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning Fundamentals
* Artificial Neural Networks (ANN)
* Data Preprocessing
* Feature Engineering
* Feature Scaling
* TensorFlow & Keras
* Binary Classification
* Model Deployment Preparation

---

## 👨‍💻 Author

**Shri Ram**

Aspiring AI/ML Engineer | Data Science | Deep Learning | Generative AI

If you found this project helpful, feel free to ⭐ the repository and connect with me on LinkedIn.
