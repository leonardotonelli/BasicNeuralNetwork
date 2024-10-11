import numpy as np



class NeuralNetwork:
    def __init__(self, num_layers: int, neurons: list, activation_function: str):
        self.num_layers = num_layers
        self.neurons = neurons
        self.weights_matrices = np.array( 
            [ np.zeros(shape=(neurons[i], neurons[i-1])) for i in range(1, len(neurons)) ]
        )
        self.biases = np.array([ np.zeros(x) for x in neurons ])
        self.activation_function = activation_function


    def forward(self, x, y):
        np.insert(self.weights_matrices, 0, np.zeros(len(x)))
        np.insert(self.weights_matrices, -1, np.zeros(len(np.unique(y))))

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def softmax(x):
            exp_x = np.exp(x - np.max(x))  
            return exp_x / np.sum(exp_x)

        if self.activation_function == "sigmoid":
            self.activation_function = lambda x: sigmoid(x)

        res = x
        activations = [x]  
        pre_activations = []  

        for i in range(len(self.weights_matrices)):
            W = self.weights_matrices[i]
            b = self.biases[i]
            f = self.activation_function
            z = W.dot(res) + b
            pre_activations.append(z)
            if i == len(self.weights_matrices) - 1:
                res = softmax(z)
            else:
                res = self.activation_function(z)
            activations.append(res)
        
        return res, activations, pre_activations
    
    def backward(self, x, y, activations, pre_activations):
        gradients_w = [np.zeros_like(w) for w in self.weights_matrices]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        m = y.shape[0]
        delta = activations[-1] - y  

        for i in reversed(range(len(self.weights_matrices))):
            gradients_w[i] = np.dot(delta, activations[i].T) / m
            gradients_b[i] = np.sum(delta, axis=1, keepdims=True) / m
            if i > 0:  
                delta = np.dot(self.weights_matrices[i].T, delta) * self.activation_derivative(pre_activations[i-1])
        
        return gradients_w, gradients_b
    
    def activation_derivative(self, z):
        if self.activation_function == "sigmoid":
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid * (1 - sigmoid)
        
    def fit(self, X, Y, epochs=1000, alpha=0.01):
        def grad_f(params):
            self.weights_matrices, self.biases = params
            _, activations, pre_activations = self.forward(X)
            grad_w, grad_b = self.backward(X, Y, activations, pre_activations)
            return grad_w, grad_b
        
        # cross entropy loss (for classification)
        def compute_loss(self, y_true, y_pred):
            m = y_true.shape[0]
            return -np.sum(y_true * np.log(y_pred)) / m

        for epoch in range(epochs):
            # forward pass
            y_pred, activations, pre_activations = self.forward(X)
            loss = self.compute_loss(Y, y_pred)
            gradients_w, gradients_b = self.backward(X, Y, activations, pre_activations)

            # step
            for i in range(len(self.weights_matrices)):
                self.weights_matrices[i] -= alpha * gradients_w[i]
                self.biases[i] -= alpha * gradients_b[i]

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

