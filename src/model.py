import numpy as np

class NeuralNetwork:
    def __init__(self, num_layers: int, neurons: list, activation_function: str):
        self.num_layers = num_layers
        self.neurons = neurons
        # Initialize weights with small random values
        self.weights_matrices = [np.random.randn(neurons[i], neurons[i-1]) * 0.01 for i in range(1, len(neurons))]
        self.biases = [np.zeros((neurons[i], 1)) for i in range(1, len(neurons))]
        self.activation_function = activation_function

    def forward(self, x):
        # Ensure input x is a 2D column vector
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        res = x
        activations = [x]
        pre_activations = []

        for i in range(len(self.weights_matrices)):
            W = self.weights_matrices[i]
            b = self.biases[i]
            z = W.dot(res) + b  
            pre_activations.append(z)
            if i == len(self.weights_matrices) - 1:
                res = self.softmax(z)
            else:
                res = self.sigmoid(z)
            activations.append(res)

        return res, activations, pre_activations
    
    def backward(self, x, y, activations, pre_activations):
        gradients_w = [np.zeros_like(w) for w in self.weights_matrices]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        m = y.shape[1]  
        delta = activations[-1] - y  

        for i in reversed(range(len(self.weights_matrices))):
            gradients_w[i] = np.dot(delta, activations[i].T) / m
            gradients_b[i] = np.sum(delta, axis=1, keepdims=True) / m
            if i > 0:  
                delta = np.dot(self.weights_matrices[i].T, delta) * self.activation_derivative(pre_activations[i-1])

        return gradients_w, gradients_b
    
    def activation_derivative(self, z):
        if self.activation_function == "sigmoid":
            sigmoid = self.sigmoid(z)
            return sigmoid * (1 - sigmoid)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[1]
        return -np.sum(y_true * np.log(y_pred + 1e-10)) / m  # Adding a small epsilon for numerical stability

    def fit(self, X, y, epochs=1000, alpha=0.01):
        for epoch in range(epochs):
            # forward pass
            y_pred, activations, pre_activations = self.forward(X)  
            loss = self.compute_loss(y, y_pred)

            gradients_w, gradients_b = self.backward(X, y, activations, pre_activations)

            # step
            for i in range(len(self.weights_matrices)):
                self.weights_matrices[i] -= alpha * gradients_w[i]
                self.biases[i] -= alpha * gradients_b[i]

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

