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
    
    def layer_output(self, X, layer):
        output = X
        for layer in self.get_Layers()[:layer]:
            output = layer.forward_pass(output)
        return output
    
    def dactivation(self, neuron, input):
        activation = neuron.get_activation()
        if activation=='sigmoid':
            return dsigmoid(neuron.raw_pass(input))
        elif activation=='relu':
            return drelu(neuron.raw_pass(input))
        else:
            return 1
    
    def fit(self, X, y, learning_rate):
        for k in range(len(X)):
            # 2d array of gradients for each layer
            layer_change_weights = []
            # derivatives of the last layer
            for n in self.get_Layers()[-1].get_neurons():
                layer = self.get_Layers()[len(self.get_Layers())-1]
                weights = n.get_weights()[:-1]
                a = (self.forward_pass(X[k])-y[k])
                b = layer.dactivation(n, self.layer_output(X[k],len(self.get_Layers())-2))
                adj_weights = [weights[x] - (learning_rate*(self.forward_pass(X[k])-y[k])*layer.dactivation(n, self.layer_output(X[k],len(self.get_Layers())-2))*self.layer_output(X[k],len(self.get_Layers())-2)[x]) for x in range(len(weights))]
                adj_weights.append(n.get_weights()[-1:][0]-(learning_rate*(self.forward_pass(X[k])-y[k])*layer.dactivation(n, self.layer_output(X[k],len(self.get_Layers())-2)))[0])
                layer_change_weights.append(adj_weights)
            neurons = self.get_Layers()[-1].get_neurons()
            for n in range(len(self.get_Layers()[-1].get_neurons())):
                neurons[n].change_weights(layer_change_weights[n])
            self.get_Layers()[-1].set_neurons(neurons)
            for l in range(len(self.get_Layers())-1):
                layer = self.get_Layers()[len(self.get_Layers())-2-l]
                new_layer_change_weights = []
                for n in range(len(layer.get_neurons())):
                    neuron = layer.get_neurons()[n]
                    weights = neuron.get_weights()[:-1]
                    prev_neurons =  self.get_Layers()[len(self.get_Layers())-1-l].get_neurons()
                    prev_partial = layer_change_weights[l]
                    d = 0
                    for x in range(len(prev_neurons)):
                        d += prev_neurons[x].get_weights()[n]*prev_partial[x]
                    a = layer.dactivation(neuron,self.layer_output(X[k],len(self.get_Layers())-2-l))
                    b = [self.layer_output(X[k],len(self.get_Layers())-2-l)[x] for x in range(len(weights))]
                    adj_weights = [weights[x] - (learning_rate*layer.dactivation(neuron,self.layer_output(X[k],len(self.get_Layers())-2-l))*self.layer_output(X[k],len(self.get_Layers())-2-l)[x]*d) for x in range(len(weights))]
                    adj_weights.append(weights[x] - (learning_rate*layer.dactivation(neuron,self.layer_output(X[k],len(self.get_Layers())-2-l))*d))
                    new_layer_change_weights.append(adj_weights)
                neurons = self.get_Layers()[len(self.get_Layers())-2-l].get_neurons()
                c = len(self.get_Layers()[len(self.get_Layers())-2-l].get_neurons())
                print(neurons)
                for n in range(len(self.get_Layers()[len(self.get_Layers())-2-l].get_neurons())):
                    neurons[n].change_weights(new_layer_change_weights[l][n])
                self.get_Layers()[len(self.get_Layers())-2-l].set_neurons(neurons)
                layer_change_weights = new_layer_change_weights
                    

model = Model(Layers=[
    Dense_Layer(output_shape=4,input_shape=2),
    Dense_Layer(output_shape=4,input_shape=4),
    Output_Layer(input_shape=4,output_shape=1)
])

X = np.array([[1,1]])
y = [2]

print(model.fit(X,y,learning_rate=0.01))