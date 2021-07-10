import numpy as np
from numpy.ma.core import array



def sigmoid(z):
    return 1/(1 + np.exp(-z))

# supervised learning
class NeuralNetwork:

    
    def __init__(self, input_nodes, hidden_nodes, output_nodes) -> None:
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = 0.3

        # ih = input -> hidden
        # ho = hidden -> output

        # we want to have a matrix that reprecents the weights of the hidden layer
        # and a matrix that reprecents the weights of the output layer
        # and a array/vector/etc that reprecents the input layer

        # in this way we can multiply the input layer by the hidden layer and performe
        # all the operations at once

        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) 
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)


        # the input layer must have the bias as the last value of its vector, which will
        # always be 1 (the bias is like b in y=mx+b)
        # here we separate the bias from the weights matrix to simplify the operations

        self.weight_bias_h = np.random.rand(self.hidden_nodes)
        self.weight_bias_o = np.random.rand(self.output_nodes)


    def feed_forward(self, inputs):

        # Guessing with the current weights

        # This is a fully connected neural network
        # this meains that every node of one layer is connected
        # to all the nodes of the next and previous layer
        
        # One perceptron (neuron) is the sum of all the inputs, and each
        # input is multiply by some weight, and sum by the bias vector
        #
        #           x0 * w0 + x1 * w1 .... xn-1 * wn-1 + 1 * bias_weight
        # x = inputs, w = weights 
        #
        # in theory the input will always have a 1 at the end representing
        # the bias. The bias is also multiply by some weight, but since the
        # bias is always 1, then we don't need to pass it as a input
        # we just sum its weight
        # Finally the result of this formula is passed to an 'activation function'
        # 
        # The activation function simply normalizes the result, for example the sigmoid function
        # we use this function to make sense of the result, for example all numbers above 100
        # are equal to 2, and all numbers between 50 and 100 are equal to 1, and all the others
        # are equal to 0

        # we created a matrix of the weights such as:
        # 
        # first hidden neuron   | w0  w1  w2  wn |
        # second hidden neuron  | w0  w1  w2  wn |
        #                           ........
        #
        # we will receive the inputs:
        # first input   | x0 |
        # second input  | x1 |
        #                ....

        # So to calculate all the perceptron's operations at once we    
        # have to say: weight matrix * inputs + its bias weights
        # and then we just have to put all the numbers of that matrix into
        # the activation fuction

        # this will give us the result of the inputs of one layer with its next layer
        # in this example we have three layers, so we have to do this twice
        # first the input from the 'user' with the hidden layer
        # second the result of the first process with the output layer



        hidden = np.matmul(self.weights_ih, inputs)
        np.add(hidden, self.weight_bias_h) # + b
        hidden = sigmoid(hidden) # sigmoid function to each element

        output = np.matmul(self.weights_ho, hidden)
        np.add(output, self.weight_bias_o)
        output = sigmoid(output)

        return output


    # stochastic gradient decent
    def train(self, inputs, targets):

        hidden = np.matmul(self.weights_ih, inputs)
        np.add(hidden, self.weight_bias_h) # + b
        hidden = sigmoid(hidden) # sigmoid function to each element

        output = np.matmul(self.weights_ho, hidden)
        np.add(output, self.weight_bias_o)
        output = sigmoid(output)

        # Calculate errors
        #
        # the error is equals to the correct answer minus the guess
        # this will give us how much it is off
        # knowing this error we can 'tune' or adjust the weights in
        # the correct direction, how much we adjust them is the learning
        # rate, if the error = 0.8, we can adjust all the error (0.8)
        # or just a percentage of this, this will behave (Gradient descent) defferently
        # depending on this learning rate where 1 = all the error (here 0.8)

        # Backpropagation
        # First, starting from the outputs, if the given answer is wrong
        # We want to adjust each weight proportionately, if the neuron A had 0.2 responsability
        # on the wrong answer and B had 0.7 then we want to punish or adjust B with more intensity than A
        # 
        # if there is more then one neuron on the output then the error of one neuron on previous layer is
        # equal to the sum of all the errors to which it contributed
        #
        # then we do this recursively for each layer 


        # Calculatin the output erros
        output_errors = np.subtract(targets, output)

        # gradient output layer
        gradient = output * (1 - output) # this is a formula called traning function
        gradient = np.multiply(gradient, output_errors)
        gradient = np.multiply(gradient, self.learning_rate)

        # to get this errors we did: 
        #
        # output_weights    inputs         
        # | w0  w1 ...|     | x0 |      | y0 |
        # | w0  w1 ...|  *  | x1 | =    | y1 |
        # | ........  |     | .. |      | .. |
        #
        # | y0 |    | target0 |   | error0 |
        # | y1 | -  | target1 | = | error1 |
        # | .. |    | ......  |   | .....  |


        # now we want to calculate how much responsability each weight/input/edge has on this errors
        # we say edge becouse we want to calcualte the impact that all the previous neurons had 
        # inputting information, the previous neuron A gave the same information to all the output neurons,
        # we want to mesure that impact
        #
        # error_neuron_0 =  w_0_0 * error0 + w_1_0 * error1 ...  // W_0_0 = weight_row_column
        #          --- becouse neuron 0 gave the same information on all the w_n_0
        # error_neuron_1 =  w_0_1 * error0 + w_1_1 * error1 ...  
        # error_neuron_2 =  w_0_2 * error0 + w_1_2 * error1 ...  
        #    ......................................
        #
        # we multiply by the weight so this can be proportional, big weights produces big changes 
        #
        # we can calculate all of this at once, transforming the columns of the matrix to be its rows (transpose)
        # and multiplying it to the error vector
        # each row correspondes to that neuron, row0 is the error of neuron 0, and so on 


        # Back to front

        # first we start with the hidden <- output
        # transpose
        hidden_T = np.transpose(np.vstack(hidden)) # this pice of shit needs vstack becouse when it has only one dimension ex: [3, 54]  the transpose funtion does nothing
        weight_ho_deltas = np.matmul(gradient, hidden_T)
        
        # adjust the weights by deltas
        self.weights_ho += weight_ho_deltas
        # adjust the bias by its deltas (gradient)
        self.weight_bias_o += gradient



        # calculate hiddeng layer input <- hidden
        # transpose
        wheights_ho_T = np.transpose(self.weights_ho)
        hidden_errors = np.multiply(wheights_ho_T, output_errors)

        # gradient for hiddeng layer
        hidden_gradient = hidden * (1 - hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient *= self.learning_rate

        # calculate input -> hidden deltas
        inputs_T = np.transpose(inputs)
        weight_ih_deltas = np.multiply(hidden_gradient, inputs_T)

        self.weights_ih += weight_ih_deltas
        # adjust the bias by its deltas (gradient)
        self.weight_bias_h = np.add(self.weight_bias_h, hidden_gradient)