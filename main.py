'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-15
'''


from errorFunctions import * #I put the error functions and their derivatives in a different file.
from activation import *     #I put the activation functions and their derivatives in a different file.

import numpy as np
import sys
import matplotlib.pyplot as plt



class Neuron():

    def __init__(self,inputLen, activationFun = "sigmoid", learningRate = .5, weights = None, bias = None):
        self.inputLen = inputLen
        self.learnR = learningRate
        self.activationFunction = activationFun

        self.output = None #for backprop.
        self.input = None   #saves the input to the neuron for backprop.
        self.newWeights = [] #saves new weight but it doesn't update until the end of backprop. (method: updateWeight)
        self.newBias = None
        self.delta = None

        if weights is None:
            #set them to random:
            self.weights = [np.random.random_sample() for i in range(inputLen)]
            self.bias = np.random.random_sample()

        else:
            self.weights = weights
            self.bias = bias



    def updateWeight(self):
        self.weights = self.newWeights
        self.newWeights = []
        self.bias = self.newBias
        self.newBias = None
        #self.bias = self.bias - self.learnR*self.delta


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
        self.newBias = self.bias - self.learnR*self.delta
        for index, PreviousNeuronOutput in enumerate(self.input):
            self.newWeights.append(self.weights[index] - self.learnR * self.delta * PreviousNeuronOutput)

    def backpropagation(self, sumDelta):
        #sumDelta will be computed at the layer level. Since it requires weights from multiple neurons.
        self.delta = sumDelta * sigmoid_prime(self.output)
        x1= sumDelta
        x2= sigmoid_prime(self.output)
        self.newBias = self.bias - self.learnR * self.delta
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
        delta_sumArr  = []
        x=len(self.neurons[0].weights)
        for i in range(len(self.neurons[0].weights)): #Weights in the RightLayer = #neurons in the LeftLayer
            delta_sum = 0
            for index, neuron in enumerate(self.neurons):
                delta_sum += neuron.mini_Delta(i)
                delta_sumArr.append(delta_sum)
        return delta_sumArr

    def backpropagation(self, deltaArr):
        for index, neuron in enumerate(self.neurons):
            neuron.backpropagation(deltaArr[index])



class NeuralNetwork():
    def __init__(self, neuronsNum = None, activationVector = 0, lossFunction = "MSE", learningRate = .1, weights = None, bias = None):
        self.inputLen   = neuronsNum[0]
        self.layersNum  = len(neuronsNum)-1 #Don't count the first one (input).
        self.activationVector = activationVector
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.weights = weights
        self.bias = bias

        if neuronsNum is None :  #By default, each layer will have 5 neurons, unless specified.
            self.neuronsNum = [3 for i in range(layersNum)]
        else:
            self.neuronsNum = neuronsNum[1:len(neuronsNum)] #Don't count he first one.

        if activationVector is None or activationVector != self.layersNum: #This is the default vector if a problem is encountered or the vector is not provided when the class is created.
            self.activationVector = ["sigmoid" for i in range(self.layersNum)]

        #Define the layers of the networks with the respective neurons:
        self.layers = []
        inputLenLayer = self.inputLen

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
            #print(" ")
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
            deltaArr = self.layers[-i + 1].deltaSum()
            self.layers[-i].backpropagation(deltaArr)

        for layer in self.layers:
            layer.updateWeights()



    def calculateLoss(self,input,target, function = "MSE"):
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
            error = mse(output, target)
        else:
            binary_cross_entropy_loss(output, target)

        return error



    def train(self, input, target, showLoss = False):
        '''
        Basically, do forward and backpropagation all together here.
        Given a single input and desired output, it will take one step of gradient descent.
        :return:
        '''
        self.calculate(input)
        if showLoss is True:
            print("mse: ", self.calculateLoss(input=input, target=target))
        self.backPropagate(target)



def doExample():
    print( "--- Example ---")

    #Let's try the class example by setting the bias and weights:
    Newweights = [[[.15,.20], [.25, .30]], [[.40, .45], [.5, .55]]]
    newBias = [[.35,.35],[.6,.6]]
    model = NeuralNetwork(neuronsNum=[2, 2, 2], activationVector=['sigmoid', 'sigmoid'], lossFunction="MSE",
                          learningRate=.5, weights=Newweights, bias = newBias)



    print("Original weights and biases of the network: ")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()


    print("\nForward pass: ")
    print(model.calculate([.05,.1]))

    #model.train(input= [.05,.1], target=[.01, .99]) #you could use just this function to do all at once.
    model.backPropagate(target= [.01, .99])
    print("\nAfter BackProp, the updated weights are:")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()

def doAnd():
    print( "\n--- AND ---")
    x = [[1,1],[1,0],[0,1], [0,0]]
    y = [[1],[0], [0], [0]]

    model = NeuralNetwork(neuronsNum=[2, 1], activationVector=['sigmoid'], lossFunction="MSE",
                          learningRate=6, weights=None, bias=None)


    print("-------- Before training ---------")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()



    for i in range(10000):
        for index in range(len(x)):
            model.train(input=x[index],target=y[index])

    print("---------------------------------")
    print("\nPredictions: ")
    for index2 in range(len(x)):
        print("\nPredict: ", x[index2])
        print(model.calculate(x[index2]))

    print("-------- After training ---------")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()




def doXor():
    print( "\n--- XOR ---")
    x = [[1,1],[1,0],[0,1], [0,0]]
    y = [[0],[1], [1], [0]]

    print("First model: [2,1] (Single Perceptron) : \n")
    model = NeuralNetwork(neuronsNum=[2, 1], activationVector=['sigmoid'], lossFunction="MSE",
                          learningRate=6, weights=None, bias=None)

    print("-------- Before training ---------")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()

    for i in range(10000):
        for index in range(len(x)):
            model.train(input=x[index], target=y[index])

    print("---------------------------------")
    print("\nPredictions: ")
    for index2 in range(len(x)):
        print("\nPredict: ", x[index2])
        print(model.calculate(x[index2]))

    print("-------- After training ---------")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()


    print("\n\n ***************************************************\n\n")

    print("Second model: [2,2,1] (Single Perceptron) : \n")


    model = NeuralNetwork(neuronsNum=[2, 2, 1], activationVector=['sigmoid', 'sigmoid'], lossFunction="MSE",
                          learningRate=.5, weights=None, bias=None)

    print("-------- Before training ---------")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()



    for i in range(10000): #It works with 100000 and alpha = 1.5 but it takes a minute
        for index in range(len(x)):
            model.train(input=x[index],target=y[index])

    print("---------------------------------")
    print("\nPredictions: ")
    for index2 in range(len(x)):
        print("\nPredict: ", x[index2])
        print(model.calculate(x[index2]))

    print("-------- After training ---------")
    print("Model's Weights:")
    model.showWeights()
    print("\nModel's Bias:")
    model.showBias()


def showLoss(learningRate, data = "and"):
    if data is "and":
        print("Loss for AND")
        x = [[1,1],[1,0],[0,1], [0,0]]
        y = [[1],[0], [0], [0]]
        title = "LearningRateVsErrorAND_MLE.png"
        figTitle= "Learning Rate Vs Error - AND "

    else:
        print("Loss for XOR")
        x = [[1, 1], [1, 0], [0, 1], [0, 0]]
        y = [[0], [1], [1], [0]]
        title = "LearningRateVsErrorXOR_MLE.png"
        figTitle= "Learning Rate Vs Error - XOR "


    errorAvrage = [] #This will contain the Ys of the plot.
    for i in learningRate:
        model = NeuralNetwork(neuronsNum=[2, 1], activationVector=['sigmoid'], lossFunction="MSE",
                          learningRate=i, weights=None, bias=None)

        errorListPerLearningRate = []
        for i in range(10):
            errorList = []
            for index in range(len(x)): #Train the algorithm with entire dataset
                model.train(input=x[index], target=y[index])

            for index2 in range(len(x)):
                errorList.append(model.calculateLoss(input=x[index2], target=y[index2])) #Collect individual errors

            errorListPerLearningRate.append(np.average(errorList))

        errorAvrage.append(np.average(errorListPerLearningRate))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    ax.plot(learningRate, errorAvrage)
    ax.set(xlabel='Learning Rate', ylabel='Loss',title=figTitle)
    ax.grid()
    plt.savefig(title)
    plt.show()








def main():
    # program_name = sys.argv[0]
    # arguments = sys.argv[1:]
    # print(arguments)
    #
    # # Input validation:
    # if len(arguments) != 1:
    #     print("Input only one of these: example, and, or xor")
    #
    # input = ["example", "and", "xor"]
    # userInput = input[2]
    #
    # if userInput is "example":
    #     doExample()
    # elif userInput is "and":
    #     doAnd()
    # elif userInput is "xor":
    #     doXor()
    # else:
    #     # Input validation
    #     print("Input Options: example, and, or xor")
    #     return 0


    #learningRateArr = np.linspace(0.1, 12, num=50)
    #showLoss(learningRateArr, data="and")
    doExample()

if __name__ == "__main__":
    main()
