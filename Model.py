import numpy as np
import matplotlib as plt
from Dense_Layer import *
from Output_Layer import *
from Neuron import *
import copy

def mse_loss(output, target):
    return np.mean((output - target) ** 2)

def mse_loss_derivative(output, target):
    return output - target

def cross_entropy_loss(output, target):
    epsilon = 1e-15  # For numerical stability
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.sum(target * np.log(output))

def cross_entropy_loss_derivative(output, target):
    return output - target  # Simplifies when using softmax activation

class Model:
    def __init__(self, Layers, loss_function='mse'):
        self.Layers = Layers
        self.loss_function = loss_function
        
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
        raw_output = input  # Raw output of the neuron
        if activation == 'sigmoid':
            return neuron.dsigmoid(raw_output)
        elif activation == 'relu':
            return neuron.drelu(raw_output)
        elif activation == 'lrelu':
            return neuron.dlrelu(raw_output)
        elif activation == 'softmax':
            return 1  # Derivative handled in loss derivative
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
    
    def fit(self, X, y, learning_rate, epochs=1):
        best_model = copy.deepcopy(self)
        best_err = float('inf')
        err = []
        for epoch in range(epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            epoch_error = []
            for x_idx in range(len(X)):
                print(f"Training sample {x_idx + 1}/{len(X)}", end='\r')
                activations = [X[x_idx]]
                raw_outputs = []
                input = X[x_idx]
                for layer in self.get_Layers():
                    if isinstance(layer, Output_Layer) and layer.activation == 'softmax':
                        raw_output = np.array([neuron.raw_pass(input) for neuron in layer.get_neurons()])
                        activation = layer.softmax(raw_output)
                        raw_outputs.append(raw_output)
                        activations.append(activation)
                        input = activation
                    else:
                        layer_activations = []
                        layer_raw_outputs = []
                        for neuron in layer.get_neurons():
                            raw_output = neuron.raw_pass(input)
                            activation = neuron.step_pass(input)
                            layer_raw_outputs.append(raw_output)
                            layer_activations.append(activation)
                        raw_outputs.append(layer_raw_outputs)
                        activations.append(layer_activations)
                        input = layer_activations

                output_activations = np.array(activations[-1])
                target = np.array(y[x_idx])

                if self.loss_function == 'mse':
                    error_output_layer = mse_loss_derivative(output_activations, target)
                    sample_loss = mse_loss(output_activations, target)
                elif self.loss_function == 'cross_entropy':
                    error_output_layer = cross_entropy_loss_derivative(output_activations, target)
                    sample_loss = cross_entropy_loss(output_activations, target)
                else:
                    raise ValueError("Unsupported loss function")

                epoch_error.append(sample_loss)

                deltas = [error_output_layer]

                for l in range(len(self.get_Layers()) - 2, -1, -1):
                    layer = self.get_Layers()[l]
                    next_layer = self.get_Layers()[l + 1]
                    delta = []
                    for i, neuron in enumerate(layer.get_neurons()):
                        d_activation = self.dactivation(neuron, raw_outputs[l][i])
                        error = sum([deltas[0][k] * next_layer.get_neurons()[k].get_weights()[i]
                                    for k in range(len(next_layer.get_neurons()))])
                        delta_i = error * d_activation
                        delta.append(delta_i)
                    deltas.insert(0, delta)
                for l in range(len(self.get_Layers())):
                    layer = self.get_Layers()[l]
                    input_activation = np.array(activations[l])
                    for i, neuron in enumerate(layer.get_neurons()):
                        delta_i = deltas[l][i]
                        weights = neuron.get_weights()
                        adj_weights = weights[:-1] - learning_rate * delta_i * input_activation
                        bias = weights[-1] - learning_rate * delta_i
                        neuron.change_weights(np.append(adj_weights, bias))
                if sample_loss < best_err:
                    best_err = sample_loss
                    best_model = copy.deepcopy(self)

            # Append loss errors for this epoch
            err.extend(epoch_error)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {np.mean(epoch_error)}")

        return err, best_model
