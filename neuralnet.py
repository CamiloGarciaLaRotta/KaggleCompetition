##### NEURAL NET OUTLINE #####
###+-------------------------+
###| x   s                   |
###| x   s   m               |
###| x   s   m  --- y        |
###| x   s   m               |
###| x   s                   |
###|                         |
###|     W   M               |
###+-------------------------+
## |x > is nx1 input layer
## |s > is mx1 sigmoid layer with mxn weight matrix W
## |m > is kx1 softmax layer with kxm weight matrix M
## y = argmax |m > is prediction of neural net
## L is loss function
## Create vectorized functions to apply them to matrices:
## --> sigmoid = 1/(1 + np.exp(-x))
## --> softmax = np.exp(x) / (np.sum(np.exp(x))
#### FORWARD PROPAGATION
### |s > = sigmoid(W|x >) or s = sigmoid(W.dot(x))
### |m > = softmax(M|s >) or m = softmwax(M.dot(s))
### y = argmax(|m >) or y = outputs[np.argmax(m)]
#### BACKPROPAGATION
### Must propagate derivatives wrt m,s,x globally
### Derivatives wrt W,M are not propagated, but used to update W,M

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetLayer:
    def __init__(self, height, prevHeight, bias=True):
        self.height = height
        self.vals = np.zeros((height,))
        self.weights = 2 * np.random.rand(height,prevHeight) - 1
        self.input = False
        self.output = False
        self.prevInput = None
        self.bias = 1 if bias else 0
        self.has_bias = True
        self.b = 2 * np.random.rand(height,) - 1

    def forward(self, x):
        raise NotImplementedError("Must define forward functions for layer")

    def backpropagate(self, gamma):
        ## TODO
        return 0

class SoftMaxOutputLayer(NeuralNetLayer):
    def __init__(self, height, prevHeight):
        NeuralNetLayer.__init__(self, height, prevHeight, False)
        self.output = True
        self.error = 0

    def forward(self, x, b):
        xp = self.weights.dot(x) + np.multiply(b,self.b)
        norm = np.sum(np.exp(xp))
        self.prevInput = x
        self.vals = (1.0/norm) * np.exp(xp)
        if b == 0:
            self.has_bias = False
        return self.vals

    def backpropagate(self, gamma, prop):
        deriv = np.diag(self.vals) - np.diag(prop)
        backprop = deriv.dot(self.weights)
        prev = np.tile(self.prevInput, (self.height, 1))
        self.weights -= np.multiply(gamma, deriv.dot(prev))
        if self.has_bias:
            self.b -= np.multiply(gamma, self.vals - prop)
        return backprop

class SigmoidLayer(NeuralNetLayer):
    def __init__(self, height, prevHeight, bias=True):
        NeuralNetLayer.__init__(self, height, prevHeight, bias)

    def forward(self, x, b):
        self.prevInput = x
        self.vals = sigmoid(self.weights.dot(x) + np.multiply(b, self.b))
        if b == 0:
            self.has_bias = False
        return self.vals

    def backpropagate(self, gamma, prop):
        sumcols = np.diag(np.sum(prop, axis=0))
        deriv = self.vals * (1-self.vals)
        partialgrad = sumcols.dot(np.diag(deriv))
        backprop = partialgrad.dot(self.weights)
        prev = np.tile(self.prevInput, (self.height, 1))
        self.weights -= np.multiply(gamma, partialgrad.dot(prev))
        if self.has_bias:
            self.b -= np.multiply(gamma, deriv)
        return backprop

class HyperbolicTangentLayer(NeuralNetLayer):
    def __init__(self, height, prevHeight, bias=True):
        NeuralNetLayer.__init__(self, height, prevHeight, bias)

    def forward(self, x, b):
        self.prevInput = x
        self.vals = np.tanh(self.weights.dot(x) + np.multiply(b, self.b))
        if b == 0:
            self.has_bias = False
        return self.vals

    def backpropagate(self, gamma, prop):
        sumcols = np.diag(np.sum(prop, axis=0))
        deriv = 1 - np.power(self.vals, 2)
        partialgrad = sumcols.dot(np.diag(deriv))
        backprop = partialgrad.dot(self.weights)
        prev = np.tile(self.prevInput, (self.height, 1))
        self.weights -= np.multiply(gamma, partialgrad.dot(prev))
        if self.has_bias:
            self.b -= np.multiply(gamma, deriv)
        return backprop

class ReLULayer(NeuralNetLayer):
    def __init__(self, height, prevHeight, bias=True):
        NeuralNetLayer.__init__(self, height, prevHeight, bias)

    def forward(self, x, b):
        self.prevInput = x
        preactivation = self.weights.dot(x) + np.multiply(b, self.b)
        self.vals = np.maximum(preactivation,0,preactivation)
        if b == 0:
            self.has_bias = False
        return self.vals

    def backpropagate(self, gamma, prop):
        sumcols = np.diag(np.sum(prop, axis=0))
        deriv = self.vals.copy()
        deriv[deriv > 0] = 1.0
        partialgrad = sumcols.dot(np.diag(deriv))
        backprop = partialgrad.dot(self.weights)
        prev = np.tile(self.prevInput, (self.height, 1))
        self.weights -= np.multiply(gamma, partialgrad.dot(prev))
        if self.has_bias:
            self.b -= np.multiply(gamma, deriv)
        return backprop

class SoftPlusRectifierLayer(NeuralNetLayer):
    def __init__(self, height, prevHeight, bias=True):
        NeuralNetLayer.__init__(self, height, prevHeight, bias)

    def forward(self, x, b):
        self.prevInput = x
        self.vals = np.log(1 + np.exp(self.weights.dot(x) + np.multiply(b, self.b)))
        if b == 0:
            self.has_bias = False
        return self.vals

    def backpropagate(self, gamma, prop):
        sumcols = np.diag(np.sum(prop, axis=0))
        deriv = sigmoid(self.vals)
        partialgrad = sumcols.dot(np.diag(deriv))
        backprop = partialgrad.dot(self.weights)
        prev = np.tile(self.prevInput, (self.height, 1))
        self.weights -= np.multiply(gamma, partialgrad.dot(prev))
        if self.has_bias:
            self.b -= np.multiply(gamma, deriv)
        return backprop

class InputLayer(NeuralNetLayer):
    def __init__(self, height, prevHeight, bias=True):
        NeuralNetLayer.__init__(self, height, prevHeight, bias)
        self.input = True
        self.weights = None

    def forward(self, x, b):
        self.vals = x
        return x

class NeuralNet:
    def __init__(self, n_inputs, n_classes):
        self.layers = [InputLayer(n_inputs, 1), SoftMaxOutputLayer(n_classes, n_inputs)]
        self.n_inputs = n_inputs
        self.n_classes = n_classes

    def fit(self, x, y, gamma=1, epochs=1000, debug=False, validation=None):
        (_, cols) = x.shape
        if not cols == self.n_inputs:
            raise Exception("Size mismatch: input layer takes " + str(self.height) + " inputs but data provides " + str(cols))
        (_, cols) = y.shape
        if not cols == self.n_classes:
            raise Exception("Labels should be one-hot encoded for " + str(n_classes) + " categorical values")
        error = 0
        dataset = list(zip(x,y))
        errors = []
        valids = []
        for e in range(epochs):
            error = 0
            for (data, label) in dataset:
                out = self.forward(data)
                error -= np.sum(label.T.dot(np.log(out)))
                self.backpropagate(gamma, label)
            if not (validation is None):
                correct = 0
                for (x,y) in validation:
                    if self.predict(x) == y:
                        correct += 1
                valids.append(100.0 * float(correct)/len(validation))
            if debug:
                print(str(e+1) + ". Error: " + str(error))
                if not (validation is None):
                    print("Validation accuracy: {:.4f}%".format(100.0 * float(correct)/len(validation)))
            errors.append(error)
        return (errors, valids)

    def backpropagate(self, gamma, target):
        for layer in reversed(self.layers[1:]):
            target = layer.backpropagate(gamma, target)

    def forward(self, x):
        x = self.layers[0].forward(x,0)
        b = self.layers[0].bias
        for layer in self.layers[1:]:
            x = layer.forward(x,b)
            b = layer.bias
        return x

    def predict(self, x):
        self.forward(x)
        if type(self.layers[-1]) == SoftMaxOutputLayer:
            return np.argmax(self.layers[-1].vals)
        elif type(self.layers[-1]) == SigmoidLayer:
            if self.layers[-1].height == 1:
                return 1 if self.layers[-1].vals[0] > 0.5 else 0
            else:
                return np.argmax(self.layers[-1].vals)
        else:
            raise NotImplementedError("Neural net currently cannot support this output stage!")

class NeuralNetBuilder:
    def __init__(self):
        self.layers = [None]
    
    def input_layer(self, height):
        self.layers[0] = InputLayer(height, 1)
        return self

    def output_layer(self, height, bias=True):
        if height == 1:
            self.layers.append(SigmoidLayer(1,self.layers[-1].height), bias)
            self.layers[-1].output = True
        else:
            self.layers.append(SoftMaxOutputLayer(height, self.layers[-1].height))
        return self

    def add_layer(self, layer_type, height, bias=True):
        prev = self.layers[-1]
        if layer_type == "sigmoid":
            self.layers.append(SigmoidLayer(height, prev.height, bias))
        elif layer_type == "tanh":
            self.layers.append(HyperbolicTangentLayer(height, prev.height, bias))
        elif layer_type == "rectifier":
            self.layers.append(SoftPlusRectifierLayer(height, prev.height, bias))
        elif layer_type == "output":
            self.layers.append(SoftMaxOutputLayer(height, prev.height, bias))
        else:
            raise Exception('Layer type "' + layer_type + '" not known')
        return self

    def build(self, out_layers=1):
        if not self.layers[-1].output:
            if out_layers == 1:
                self.layers.append(SigmoidLayer(1,self.layers[-1].height))
            else:
                self.layers.append(SoftMaxOutputLayer(out_layers, self.layers[-1].height))

        net = NeuralNet(self.layers[0].height, self.layers[-1].height)
        net.layers = self.layers
        return net
