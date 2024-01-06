class Network:
    def __init__(self, n_nodes: int, n_layers: int):
        self.n_nodes = n_nodes
        self.n_layers = n_layers
class Layer(Network):
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.n_layers = 1

class Node(Layer):
    def __init__(self):
        self.n_nodes = 1
        self.n_layers = 1

if __name__ == '__main__':
    teste = Network(5,7)
    print(teste.n_nodes)
    print(teste.n_layers)