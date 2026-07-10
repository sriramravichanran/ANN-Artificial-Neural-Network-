import joblib
from tensorflow.keras.models import load_model

def predict_churn(customer_data):

    # Load Scaler
    scaler = joblib.load("models/scaler.pkl")

    # Load Model
    model = load_model("models/ann_model.h5")

    # Scale Input
    customer_data = scaler.transform([customer_data])

    # Prediction
    prediction = model.predict(customer_data)

    if prediction[0][0] > 0.5:
        return "Customer will Leave"
    else:
        return "Customer will Stay"


if __name__ == "__main__":

    customer_data = [
        650,      # CreditScore
        1,        # Gender
        42,       # Age
        3,        # Tenure
        60000,    # Balance
        2,        # NumOfProducts
        1,        # HasCrCard
        1,        # IsActiveMember
        50000,    # EstimatedSalary
        0,        # Geography_Germany
        1         # Geography_Spain
    ]

    result = predict_churn(customer_data)

    print(result)