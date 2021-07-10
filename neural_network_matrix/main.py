import numpy as np
from neural_network import NeuralNetwork


# Xor example, first input, second input, label
trainig_data = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1]])

def main():

    # generating the outputs
    nn = NeuralNetwork(2, 2, 1)

    # Traing
    for i in range(10000):
        choose = np.random.randint(0, high=4)
        # print('input:', trainig_data[choose][:2], 'label:', trainig_data[choose][2])
        nn.train(trainig_data[choose][:2], [ trainig_data[choose][2]])

    # test
    print(nn.feed_forward([0,1]))
    print(nn.feed_forward([0,0]))
    print(nn.feed_forward([1,1]))
    print(nn.feed_forward([1,0]))
    

if __name__ == '__main__':
    main()