import numpy as np
import matplotlib as plt
from Neuron import *
import time
import matplotlib.pyplot as plt

class Output_Layer:
    def __init__(self, input_shape, output_shape, activation_func=none):
        neuron_layer = np.array([])
        for x in range(output_shape):
            neuron_layer = np.append(neuron_layer,Perceptron(n_of_weights=input_shape, step=activation_func, activation=activation_func))
        self.neurons = neuron_layer
        
    def get_neuron(self, n):
        return self.neurons[n]
    
    def get_neurons(self):
        return self.neurons
    
    def set_neurons(self, adj_neurons):
        self.neurons = adj_neurons
    
    def forward_pass(self, X):
        output = np.array([])
        for neuron in self.neurons:
            output = np.append(output,neuron.step_pass(X))
        return output
    
    def relu(self, input):
        if input>0:
            return input
        else:
            return 0
        
    def drelu(self, input):
        if input>0:
            return 1
        else:
            return 0

    def sigmoid(self, input):
        return 1/(1+np.e**(-input))

    def dsigmoid(self, input):
        return self.sigmoid(input)*(1-self.sigmoid(input))
    
    def dactivation(self, neuron, input):
        activation = neuron.get_activation()
        if activation=='sigmoid':
            return dsigmoid(neuron.raw_pass(input))
        elif activation=='relu':
            return drelu(neuron.raw_pass(input))
        else:
            return neuron.raw_pass(input)
    
    def fit(self, X, y, learning_rate):
        weight_change = [[w] for w in self.get_neurons()[0].get_weights()]
        error = [0]
        neurons = self.get_neurons()
        self.get_neurons()[0].change_weights([0.5, 0.5, -0.36])
        for k in range(len(X)):
            for n in neurons:
                weights = n.get_weights()[:-1]
                # a = (self.forward_pass(X[k])[0]-y[k])
                # b = self.forward_pass(X[k])[0]
                # c = self.dactivation(n, X[k])
                # d = X[k]
                # e = [-learning_rate*(self.forward_pass(X[k])[0]-y[k])*self.dactivation(n, X[k])*X[k][x] for x in range(len(weights))]
                adj_weights = [weights[x]-learning_rate*(self.forward_pass(X[k])[0]-y[k])*self.dactivation(n, X[k])*X[k][x] for x in range(len(weights))]
                adj_weights.append(n.get_weights()[-1:][0]-learning_rate*(self.forward_pass(X[k])[0]-y[k])*self.dactivation(n, X[k]))
                n.change_weights(adj_weights)
                for x in range(len(adj_weights)):
                    weight_change[x].append(adj_weights[x])
                # print("weight",weights)
                # print("input",X[k])
            error.append(y[k]-n.step_pass(X[k]))
            self.set_neurons(neurons)
        return weight_change, error
    
    
## testing

start = time.time()

layer = Output_Layer(2, 1)

X = np.array([np.random.uniform(-1.1,1.1,2) for x in range(10000)])
y = [30*x[0]+20*x[1]+30 for x in X]

change, err = layer.fit(X,y, learning_rate=0.0002)

print(layer.get_neurons()[0].get_weights())

end = time.time()
print(end-start)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 10))

ycoef = [30,20,30]

for x in range(len(change)):
    axes[x].plot(change[x])
    axes[x].axhline(y=ycoef[x],color='black')
axes[3].scatter(range(len(err)),err)
plt.show()