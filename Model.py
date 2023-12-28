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
    
    def fit(self, X, y, learning_rate):
        error = []
        for k in range(len(X)):
            for layer in self.get_Layers():
                neurons = layer.get_neurons()
                for n in neurons:
                    weights = n.get_weights()[:-1]
                    adj_weights = [weights[x]-(learning_rate*(layer.forward_pass(X[k])-y[k])*layer.dactivation(n,X[k])*X[k][x])[0] for x in range(len(weights))]
                    adj_weights.append(n.get_weights()[-1:][0]-(learning_rate*(layer.forward_pass(X[k])-y[k])*layer.dactivation(n,X[k]))[0])
                    n.change_weights(adj_weights)
                layer.set_neurons(neurons)
            self.set_Layers(self.get_Layers())
            error.append(self.forward_pass(X[k])-y[k])


model = Model(Layers=[
    Dense_Layer(output_shape=4,input_shape=2),
    Dense_Layer(output_shape=4,input_shape=4),
    Output_Layer(input_shape=4,output_shape=1)
])

a = np.random.uniform(-100,100)
b = np.random.uniform(-100,100)
c = np.random.uniform(-100,100)

X = np.array([np.random.uniform(-2,2,2) for x in range(1000)])
y = [a*x[0]+b*x[1]+c for x in X]

error = model.fit(X,y,0.005)

plt.plot(error)