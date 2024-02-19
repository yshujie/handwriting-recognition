import numpy as np
from number_recognition.index import init_network, forward

def main():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y) # [0.31682708 0.69627909]

if __name__ == "__main__":
    main()