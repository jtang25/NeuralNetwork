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
    
    def set_Layer(self, new_layer, layer_index):
        self.Layers[layer_index] = new_layer
    
    def forward_pass(self, X):
        output = X
        for layer in self.get_Layers():
            output = layer.forward_pass(output)
        return output
    
    def layer_output(self, X, layer_num):
        output = X
        for layer in self.get_Layers()[:layer_num]:
            output = layer.forward_pass(output)
        return output
    
    def dactivation(self, neuron, input):
        activation = neuron.get_activation()
        if activation=='sigmoid':
            return dsigmoid(neuron.raw_pass(input))
        elif activation=='relu':
            return drelu(neuron.raw_pass(input))
        elif activation=='lrelu':
            return dlrelu(neuron.raw_pass(input))
        else:
            return 1
        
    def summary(self):
        for l in self.get_Layers():
            layer_type = str(type(l))
            if 'Dense' in layer_type:
                print('Dense Layer  |',len(l.get_neurons()),'neurons |',len(l.get_neurons()[0].get_weights())-1,'weights | 1 bias')
            elif 'Output' in layer_type:
                print('Output Layer |',len(l.get_neurons()),'outputs |',len(l.get_neurons()[0].get_weights())-1,'weights | 1 bias')
        print('Input Shape:',len(self.get_Layers()[0].get_neurons()[0].get_weights())-1)
        print('Output Shape:',len(self.get_Layers()[-1].get_neurons()))
    
    def fit(self, X, y, learning_rate):
        error = []
        for x in range(len(X)):
            # 2d array of gradients for each layer
            layer_change_weights = []
            # derivatives of the last layer
            output_layer = self.get_Layers()[-1:][0]
            output_neurons = output_layer.get_neurons()
            prev_layer = self.get_Layers()[len(self.get_Layers())-2]
            for n in output_neurons:
                weights = n.get_weights()[:-1]
                prev_output = self.layer_output(X[x], len(self.get_Layers())-1)
                adj_weights = [weights[i]-(learning_rate*(self.forward_pass(X[x])-y[x])*output_layer.dactivation(n,X[x])*prev_output[i])[0] for i in range(len(weights))]
                adj_weights.append(n.get_weights()[-1:][0]-(learning_rate*(self.forward_pass(X[x])-y[x])*output_layer.dactivation(n,X[x]))[0])
                n.change_weights(adj_weights)
                layer_change_weights.append(adj_weights)
            output_layer.set_neurons(output_neurons)
            self.set_Layer(output_layer, len(self.get_Layers())-1)
            layers = self.get_Layers()
            for k in range(len(layers)-2):
                layer = layers[len(layers)-2-k]
                new_layer_change_weights = []
                neurons = layers[len(layers)-2-k].get_neurons()
                prev_neurons = layers[len(layers)-1-k].get_neurons()
                prev_output = self.layer_output(X[k], len(layers)-2-k)
                for j in range(len(neurons)):
                    weights = neurons[j].get_weights()[:-1]
                    d = 0
                    for l in range(len(prev_neurons)):
                        d += prev_neurons[l].get_weights()[:-1][j]*layer_change_weights[l][j]
                    d *= layers[k].dactivation(neurons[j], prev_output)
                    adj_weights = [weights[i] - (learning_rate*d*prev_output[i]) for i in range(len(weights))]
                    adj_weights.append(weights[-1:][0]-(learning_rate*d))
                    neurons[j].change_weights(adj_weights)
                    new_layer_change_weights.append(adj_weights)
                layers[len(layers)-2-k].set_neurons(neurons)
                self.set_Layer(layers[len(layers)-2-k], len(layers)-2-k)
                layer_change_weights = new_layer_change_weights
            input_layer = layers[0]
            neurons = input_layer.get_neurons()
            prev_neurons = layers[1].get_neurons()
            for j in range(len(neurons)):
                weights = neurons[j].get_weights()[:-1]
                d = 0
                for l in range(len(prev_neurons)):
                    prev_weights = prev_neurons[l].get_weights()[:-1]
                    d += prev_neurons[l].get_weights()[:-1][j]*layer_change_weights[l][j]
                d *= input_layer.dactivation(neurons[j], X[x])
                adj_weights = [weights[i] - (learning_rate*d*X[x][i]) for i in range(len(weights))]
                adj_weights.append(weights[-1:][0]-(learning_rate*d))
                neurons[j].change_weights(adj_weights)
            input_layer.set_neurons(neurons)
            self.set_Layer(input_layer, 0)
            error.append(abs(self.forward_pass(X[x])-y[x]))
        return error