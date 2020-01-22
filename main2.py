'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-15
'''


from errorFunctions import *
from activation import *
import numpy as np
import matplotlib.pyplot as plt



class Neuron():

    def __init__(self,inputLen, activationFun = "sigmoid", learningRate = .5, weights = None, bias = None):
        self.inputLen = inputLen
        self.learnR = learningRate
        self.activationFunction = activationFun

        self.output = None #for backprop.
        self.input = None   #saves the input to the neuron for backprop.
        self.newWeights = [] #saves new weight but it doesn't update until the end of backprop. (method: updateWeight)
        self.delta = None

        if weights is None:
            #set them to random:
            self.weights = [np.random.random_sample() for i in range(inputLen)]
            self.bias = np.random.random_sample()

        else:
            self.weights = weights
            self.bias = bias



    def updateWeight(self):
        self.weights = self. newWeights
        self.newWeights = []


    def activate(self,x):
        '''
        given a value returns its value after activation(depending on the activation function used).
        :return:
        '''
        if self.activationFunction == "sigmoid":
           return sigmoid(x)
        #else:
        #    return linear(x)


    def calculate(self, input):
        '''
        Given an input, it will calculate the output
        :return:
        '''
        self.input = input
        self.output = self.activate(np.dot(input,self.weights) + self.bias)
        return self.output


    def backpropagationLastLayer(self, target):
        self.delta = mse_prime(self.output, target) * sigmoid_prime(self.output)
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[index] - self.learnR * self.delta * PreviousNeuronOutput)

    def backpropagation(self, sumDelta):
        #sumDelta will be computed at the layer level. Since it requires weights from multiple neurons.
        self.delta = sumDelta * sigmoid_prime(self.output)
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[index] - self.learnR * self.delta * self.input[index])

    def mini_Delta(self, index):
        return self.delta * self.weights[index]





class FullyConnectedLayer():
    def __init__(self, inputLen, numOfNeurons = 5, activationFun = "sigmoid", learningRate = .1, weights = None, bias = None):
        self.inputLen = inputLen
        self.neuronsNum = numOfNeurons
        self.activationFun = activationFun
        self.learningRate = learningRate
        self.weights = weights
        self.bias = bias
        self.layerOutput = []

        if weights is None:
            self.neurons = [Neuron(inputLen=self.inputLen, activationFun=activationFun, learningRate=self.learningRate, weights=self.weights) for i in range(numOfNeurons)]
        else:
            self.neurons = [Neuron(inputLen=self.inputLen, activationFun=activationFun, learningRate=self.learningRate, weights=self.weights[i], bias= self.bias[i]) for i in range(numOfNeurons)]


    def calculate(self, input):
        '''
        Will calculate the output of all the neurons in the layer.
        :return:
        '''
        self.layerOutput = []
        for neuron in self.neurons:
            self.layerOutput.append(neuron.calculate(input))

        return self.layerOutput

    def backPropagateLast(self, target):
        for targetIndex, neuron in enumerate(self.neurons):
            #neuron.update_weight(   target[targetIndex], self.layerOutput)
            neuron.backpropagationLastLayer(target=target[targetIndex])

    def updateWeights(self):
        for neuron in self.neurons:
            neuron.updateWeight()

    def deltaSum(self):
        delta_sum  = 0
        for index, neuron in enumerate(self.neurons):
            delta_sum += neuron.mini_Delta(index)
        return delta_sum

    def backpropagation(self, delta):
        for neuron in self.neurons:
            neuron.backpropagation(delta)



class NeuralNetwork():
    def __init__(self, inputLen, layersNum = 2, neuronsNum = None, activationVector = 0, lossFunction = "MSE", learningRate = .1, weights = None, bias = None):
        self.inputLen   = inputLen
        self.layersNum  = layersNum
        self.activationVector = activationVector
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.weights = weights
        self.bias = bias

        if neuronsNum is None :  #By default, each layer will have 5 neurons, unless specified.
            self.neuronsNum = [3 for i in range(layersNum)]
        else:
            self.neuronsNum = neuronsNum

        if activationVector is None or activationVector != layersNum: #This is the default vector if a problem is encountered or the vector is not provided when the class is created.
            self.activationVector = ["sigmoid" for i in range(layersNum)]

        #Define the layers of the networks with the respective neurons:
        self.layers = []
        inputLenLayer = inputLen

        if weights is None:
            for i in range(self.layersNum):
                self.layers.append(
                    FullyConnectedLayer(numOfNeurons=self.neuronsNum[i], activationFun=self.activationVector[i], inputLen=inputLenLayer, learningRate=self.learningRate, weights=self.weights))
                # The number of weights in one layer depends on the number of neurons in the previous layer:
                inputLenLayer = self.neuronsNum[i]


        else:
            for i in range(self.layersNum):
                self.layers.append(
                    FullyConnectedLayer(numOfNeurons=self.neuronsNum[i], activationFun=self.activationVector[i], inputLen=inputLenLayer, learningRate=self.learningRate, weights=self.weights[i], bias=self.bias[i]))
                # The number of weights in one layer depends on the number of neurons in the previous layer:
                inputLenLayer = self.neuronsNum[i]



    def showWeights(self):
        #Function which just goes through each neuron in each layer and displays the weights.
        inputLenLayer = self.inputLen
        for i in range(self.layersNum):
            print(" ")
            for k in range(self.neuronsNum[i]):
                print(self.layers[i].neurons[k].weights)

            inputLenLayer = self.neuronsNum[i]

    def showBias(self):
        #Function which just goes through each neuron in each layer and displays the weights.
        inputLenLayer = self.inputLen
        for i in range(self.layersNum):
            print(" ")
            for k in range(self.neuronsNum[i]):
                print(self.layers[i].neurons[k].bias)

            inputLenLayer = self.neuronsNum[i]



    def calculate(self, input):
        '''
        given an input calculates the output of the network.
        input should be a list.
        :return:
        '''
        output = input
        for layer in self.layers:
            output = layer.calculate(output)

        return output

    def backPropagate(self, target):
        self.layers[-1].backPropagateLast(target)
        layersCounter = self.layersNum+1

        for i in range(2,layersCounter):
            delta = self.layers[-i + 1].deltaSum()
            self.layers[-i].backpropagation(delta)

        for layer in self.layers:
            layer.updateWeights()



    def calculateLoss(self,input,desired_output, function = "MSE"):
        '''
        Given an input and desired output, calculate the loss.
        Can be implemented with MSE and binary cross.
        :param input:
        :param output:
        :return:
        '''
        N = len(input)
        output = self.calculate(input)
        if function == "MSE":
            error = mse(output, desired_output)
        else:
            binary_cross_entropy_loss(output, desired_output)

        return error



    def train(self):
        '''
        Basically, do forward and backpropagation all together here. 
        Given a single input and desired output, it will take one step of gradient descent.
        :return:
        '''
        x=0









def main():

    #The number of neurons in the last layers should be equal to the number of classes.
    #there are easier ways to do this, but I'm just following the instructions.

    '''
    #Random weights:
    Newweights = None
    model = NeuralNetwork(inputLen=2, layersNum = 2, neuronsNum = [3,2], activationVector = 0, lossFunction = "MSE", learningRate = .1, weights = Newweights)
    model.showWeights()
    out= model.calculate([1,0])
    '''



    #Let's try the class example by setting the bias and weights:
    Newweights = [[[.15,.20], [.25, .30]], [[.40, .45], [.5, .55]]]
    newBias = [[.35,.35],[.6,.6]]
    model = NeuralNetwork(inputLen=2, layersNum=2, neuronsNum=[2, 2], activationVector=0, lossFunction="MSE",
                          learningRate=.5, weights=Newweights, bias = newBias)
    model.showWeights()
    model.showBias()

    print("forward pass:")
    print(model.calculate([.05,.1]))


    model.backPropagate(target= [.01, .99])
    print("after BackProp: ")
    model.showWeights()


if __name__ == "__main__":
    main()
