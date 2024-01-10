import numpy as np
import matplotlib.pyplot as plt

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
    
def lrelu(input):
    if input>0:
        return input
    else:
        return 0.1*input

def dlrelu(input):
    if input>0:
        return 1
    else:
        return 0.1

def sigmoid(input):
    return 1/(1+np.e**(-input))

def dsigmoid(input):
    return (np.e**(-input))/((1+(np.e**(-input)))**2)

def none(input):
    return 1

class Perceptron:
    def __init__(self, n_of_weights, step, activation):
        self.weights = np.random.uniform(-1,1,n_of_weights+1)
        self.step = step
        self.activation = activation
    
    def get_weights(self):
        return self.weights
    
    def change_weights(self, new_weights):
        self.weights = new_weights
    
    def get_step(self):
        return self.step
    
    def get_activation(self):
        return self.activation
    
    def raw_pass(self, input):
        output = self.weights[len(self.weights)-1]
        for x in range(len(self.weights)-1):
            output = output + self.weights[x]*input[x]
        return output
    
    def step_pass(self, input):
        output = self.weights[len(self.weights)-1]
        for x in range(len(self.weights)-1):
            output = output + self.weights[x]*input[x]
        match self.get_step():
            case "relu":
                output = relu(output)
            case "sigmoid":
                output = sigmoid(output)
            case "lrelu": 
                output = lrelu(output)
            case _:
                output = output
        return output
    
    # 2w(y-wx+b)
    def fit_step(self, X, y, learning_rate):
        preds = []
        # weights
        weights = self.get_weights()
        # calculate gradient with respect to each weight (and bias)
        # and adjust them accordingly
        for j in range(len(X)):
            pred = self.step_pass(X[j])
            for i in range(len(weights)-1):
                gradient = -2*X[j][i]*self.activation(weights[i]*X[j][i])*(y[j]-pred)
                weights[i] = weights[i] - learning_rate*gradient
            weights[len(weights)-1] = weights[len(weights)-1] - learning_rate*-2*(y[j]-pred)
        return preds
    
# perc = Perceptron(n_of_weights=1, step="", activation=none)
# print("Weights ",perc.get_weights())

# X_test = np.linspace(-5, 5, 21)
# y_test = X_test
# X_test = np.array([np.array([x]) for x in X_test])
# noise = np.random.uniform(low=-0.5, high=0.5, size=(21,))
# y_test = y_test + noise


# plt.figure(figsize=(8, 6))
# n= 10
# rainbow_colors = plt.cm.rainbow(np.linspace(0.5, 1, n))

# x = np.linspace(-10, 10, 100)
# y = [perc.step_pass([x1]) for x1 in x]
# plt.plot(x, y, label=0, color=rainbow_colors[0])

# for i in range(n-1):
#     perc.fit_step(X_test, y_test, learning_rate=0.001)
#     y = [perc.step_pass([x1]) for x1 in x]
#     plt.plot(x, y, label=i+1, color=rainbow_colors[i+1])
    
# plt.plot(x, y, label=n, color=rainbow_colors[n-1])

# plt.scatter(X_test,y_test, label= "point")

# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Linear Equation Plot')
# plt.axhline(0, color='black', linewidth=1)
# plt.axvline(0, color='black', linewidth=1)
# plt.grid(True)
# plt.legend()
# plt.show()
