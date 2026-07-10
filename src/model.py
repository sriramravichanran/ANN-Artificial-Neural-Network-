from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_ann(input_dim):

    model = Sequential()

    # Input + Hidden Layer
    model.add(Dense(
        units=16,
        activation='relu',
        input_dim = input_dim
    ))

    # Hidden Layer 2
    model.add(Dense(
        units=8,
        activation='relu'
    ))

    # Output Layer
    model.add(Dense(
        units=1,
        activation='sigmoid'
    ))

    # Compiling
    model.compile(
        optimizer = "adam", # To Update weights 
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    
    # 12 input Columns
    # Building the Model
    model = build_ann(12)
    
    # Model Architecture
    model.summary()