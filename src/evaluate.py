from src.data_preprocessing import load_data, preprocessing_the_data
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


def evaluate_model():

    # Load Dataset
    df = load_data(r"data\Churn_Modelling.csv")

    X_train, X_test, y_train, y_test = preprocessing_the_data(df)

    # Load Scaler
    scaler = joblib.load("models/scaler.pkl")

    # Scale Test Data
    X_test_scaled = scaler.transform(X_test)

    # Load Model
    model = load_model("models/ann_model.h5")

    # Predictions
    y_pred_prob = model.predict(X_test_scaled)

    y_pred = (y_pred_prob > 0.5).astype(int)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()