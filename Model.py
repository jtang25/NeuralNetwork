import numpy as np
import matplotlib as plt
from Dense_Layer import *
from Output_Layer import *
from Neuron import *

class Model:
    def __init__(self, Layers):
        self.Layers = Layers
        
    def get_Layers(self):
        return self.Layers
    
    def forward_pass(self, X):
        output = X
        for layer in self.get_Layers():
            output = layer.forward_pass(output)
        return output
    
    def relu(input):
        if input>0:
            return input
        else:
            return 0
        
    def drelu(input):
        if input>0:
            return 1
        else:
            return 0

    def sigmoid(input):
        return 1/(1+np.e**(-input))

    def dsigmoid(input):
        return (np.e**(-input))/((1+(np.e**(-input)))**2)
    
    
    
    # def fit(self, X, y):
        
    
model = Model(Layers=[
    Dense_Layer(n_of_neurons=4,input_shape=2),
    Dense_Layer(n_of_neurons=4,input_shape=4),
    Output_Layer(input_shape=4,output_shape=1)
])

X = np.array([1,1,1])

print(model.forward_pass(X))