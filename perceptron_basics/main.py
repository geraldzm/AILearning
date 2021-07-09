from matplotlib import pyplot as plt
from perceptron import Perceptron
from traing import Point, f



def draw(perceptron, points):
    
    x = []
    y = []
    c = []

    for point in points:
        x.append(point.x)
        y.append(point.y)
        c.append(point.c)

    plt.scatter(x, y, c=c)

    # Correct line
    x0 = -1
    y0 = f(x0)
    x1 = 1
    y1 = f(x1)
    plt.plot([x0, x1], [y0, y1], c='orange')

    # Guessed line
    x0 = -1
    y0 = perceptron.guessed_line(x0)
    x1 = 1
    y1 = perceptron.guessed_line(x1)
    plt.plot([x0, x1], [y0, y1], c='b')

    plt.show()


def train(perceptron, traning_data):

    for i in range(1):
        for td in traning_data:
            perceptron.train([td.x, td.y, td.bias], td.label)


def classify(perceptron, points):

    for p in points:
        p.set_color(perceptron.guess([p.x, p.y, p.bias]))


def main():
    
    # create points
    points = []
    for i in range(100):
        p = Point()
        points.append(p)

    # training
    percep = Perceptron()
    
    print('starting weights:', percep.weights)
    train(percep, points)
    print('new weights:', percep.weights)

    # classifying
    classify(percep, points)

    draw(percep, points)


if __name__ == "__main__":
    main()