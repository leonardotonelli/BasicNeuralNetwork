import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.model import NeuralNetwork

def main():
    # Generate a simple 3-class dataset
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Initialize the neural network
    # - 2 neurons for input (2 features), 
    # - 5 neurons in a hidden layer, 
    # - 3 neurons in the output layer (3 classes)
    nn = NeuralNetwork(num_layers=3, neurons=[2, 5, 3], activation_function="sigmoid")

    # Train the neural network
    nn.fit(X_train.T, y_train.T, epochs=1000, alpha=0.01)

    # Predict on the test set
    y_pred, _, _ = nn.forward(X_test.T)

    # Convert the output probabilities back to class labels
    y_pred_labels = np.argmax(y_pred, axis=0)

    # Convert one-hot encoded test labels back to class labels
    y_test_labels = np.argmax(y_test, axis=1)

    # Accuracy
    accuracy = np.mean(y_pred_labels == y_test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()