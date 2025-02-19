import numpy as np
class Flatten:
    def __init__(self):
        self.input_shape = None  # Stores the input shape for backpropagation
    
    def get_input_shape(self):
        return self.input_shape

    def forward_pass(self, X):
        """
        Flattens the input tensor into a 1D array (no batch size preservation).
        """
        X = np.array(X)
        self.input_shape = X.shape  # Save the original shape for backward pass
        return X.flatten()

    def backward_pass(self, delta):
        return delta.reshape(self.input_shape)