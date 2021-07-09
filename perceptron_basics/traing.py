import random



# Correct line
def f(x):
    return 0.8 * x - 0.2


class Point:

    def __init__(self) -> None:
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.bias = 1
        self.c = 'b'
        
        if self.y > f(self.x):
            self.label = 1 # the point is over the line
        else:
            self.label = -1 # the point is below the line


    def set_color(self, guess):

        if self.label == guess:
            self.c = 'g'
        else:
            self.c = 'r'