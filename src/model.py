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
        activations = [x]  # Store activations for backpropagation
        pre_activations = []  # Store z values for backpropagation

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
        # Backpropagation to calculate gradients
        gradients_w = [np.zeros_like(w) for w in self.weights_matrices]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        m = y.shape[0]
        
        # Compute the derivative of the loss with respect to the softmax output
        delta = activations[-1] - y  # y_pred - y_true

        # Backpropagate through layers
        for i in reversed(range(len(self.weights_matrices))):
            gradients_w[i] = np.dot(delta, activations[i].T) / m
            gradients_b[i] = np.sum(delta, axis=1, keepdims=True) / m
            if i > 0:  # Skip backpropagating to the input layer
                delta = np.dot(self.weights_matrices[i].T, delta) * self.activation_derivative(pre_activations[i-1])
        
        return gradients_w, gradients_b

