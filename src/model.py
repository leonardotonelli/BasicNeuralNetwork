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
            exp_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
            return exp_x / np.sum(exp_x)

        if self.activation_function == "sigmoid":
            self.activation_function = lambda x: sigmoid(x)

        res = x
        for i in range(len(self.weights_matrices)):
            W = self.weights_matrices[i]
            b = self.biases[i]
            f = self.activation_function
            if i == len(self.weights_matrices):
                res = softmax( W.T.dot(res) + b )
            else:
                res = f( W.T.dot(res) + b )
        
        return res
        

