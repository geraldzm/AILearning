from random import choice



class Perceptron:


    def __init__(self) -> None:
        
        self.weights = []

        # init weights
        for i in range(3):
            self.weights.append(choice([1, -1]))


    # Activation funcion
    def sign(self, n):
        if n >= 0:
            return 1
        return -1


    def guess(self, inputs):
        
        sum = 0

        for i in range(len(self.weights)):
            sum += self.weights[i] * inputs[i]

        return self.sign(sum)


    # Target is the correct answer given that inputs
    # Target is a number
    def train(self, inputs, target):

        guess = self.guess(inputs)
        error = target - guess

        learning_rate = 0.1 # we don't want to steer all the way to the target at once

        # Tuning all the weights
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * learning_rate


    def guessed_line(self, x):
        return -(self.weights[2] / self.weights[1]) - (self.weights[0] / self.weights[1]) * x


    def __str__(self) -> str:
        return str(self.weights)