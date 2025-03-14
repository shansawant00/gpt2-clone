class Block:
    def __init__(self, layer, activation):
        self.layer = layer
        self.activation = activation

    def forward(self, x):
        return self.activation(self.layer(x))

    def __repr__(self):
        return f"Block(layer={self.layer}, activation={self.activation})"