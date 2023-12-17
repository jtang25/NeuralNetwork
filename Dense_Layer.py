import numpy as np
import matplotlib as plt
from Neuron import *

class Dense_Layer:
    def __init__(self, n_of_neurons, input_shape, activation_func=none):
        neuron_layer = np.array([])
        for x in range(n_of_neurons):
            neuron_layer = np.append(neuron_layer,Perceptron(n_of_weights=input_shape, step=activation_func, activation=activation_func))
        self.neurons = neuron_layer
    
    def get_neuron(self, n):
        return self.neurons[n]
    
    def get_neurons(self):
        return self.neurons
    
    def forward_pass(self, X):
        output = np.array([])
        for neuron in self.neurons:
            output = np.append(output,neuron.step_pass(X))
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
    
    # def fit(self, x, y):
    #     neurons = self.get_neurons()
    #     for x in neurons:
    #         output = self.forward_pass(x)
            

layer = Dense_Layer(4, 3, activation_func="relu")
print(layer.fit(1,2))