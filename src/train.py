from src.data_preprocessing import load_data, preprocessing_the_data
from src.model import build_ann
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():

    # Loading Dataset
    df = load_data(r"data\Churn_Modelling.csv")

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocessing_the_data(df)

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save FITTED scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Build Model
    model = build_ann(X_train_scaled.shape[1])

    # Train Model
    model.fit(
        X_train_scaled,
        y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    # Save Model
    model.save("models/ann_model.h5")

    print("Model trained and saved successfully...!")


if __name__ == "__main__":
    train_model()
    # 