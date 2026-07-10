import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocessing_the_data(df):

    # Removing unnecessary Columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encoding Gender
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    # One Hot Encoding Geography
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    # Features and Target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data("data/Churn_Modelling.csv")

    X_train, X_test, y_train, y_test = preprocessing_the_data(df)

    print("Preprocessing completed Successfully")



    # Changes removed
    import joblib
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Save Scaler
    joblib.dump(scaler,"models/scaler.pkl")