import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, dims, batchnorm=False):
        super(FCNN, self).__init__()
        self.dims = dims
        layers = []
        layers.append(nn.Linear(in_features=dims[0], out_features=dims[1]))
        for i in range(2, len(dims)-1):
            layers.append(nn.Linear(in_features=dims[i - 1], out_features=dims[i]))
            layers.append(nn.ReLU())
            if(batchnorm):
                layers.append(nn.BatchNorm1d(num_features=dims[i]))
        layers.append(nn.Linear(in_features=dims[-2], out_features=dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FCNN2(nn.Module):
    def __init__(self, dims):
        super(FCNN2, self).__init__()
        self.dims = dims
        layers = []
        layers.append(nn.Linear(in_features=dims[0], out_features=dims[1]))
        for i in range(2, len(dims)-2):
            layers.append(nn.Linear(in_features=dims[i - 1], out_features=dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=dims[-3], out_features=dims[-2]))
        layers.append(nn.Linear(in_features=dims[-2], out_features=dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class ReLU_MLP(nn.Module):
    def __init__(self, layer_dims, output="linear", bias=True, layernorm=False):
        '''
        A generic ReLU MLP network.

        Arguments:
            - layer_dims: a list [d1, d2, ..., dn] where d1 is the dimension of input vectors and d1, ..., dn
                        is the dimension of outputs of each of the intermediate layers.
            - output: output activation function, either "sigmoid" or "linear".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''
        super(ReLU_MLP, self).__init__()
        layers = []
        for i in range(1, len(layer_dims) - 1):
            if (layernorm and i != 1):
                layers.append(nn.LayerNorm(layer_dims[i - 1]))
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i], bias=bias))
            layers.append(nn.ReLU(layer_dims[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1], bias=bias))
            layers.append(nn.Sigmoid())
        if (output == "linear"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1], bias=bias))

        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp, *args):
        if (type(inp) == tuple):
            args = inp[1:]
            inp = inp[0]
        if (len(args) > 0):
            inp = torch.cat([inp] + list(args), dim=1)
        return self.out(inp)


class FCCritic(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, bias=True):
        super(FCCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.inp_projector = nn.Linear(in_features=input_dim,
                                       out_features=hidden_layer_dims[0],
                                       bias=bias)
        self.outp_projector = nn.Linear(in_features = hidden_layer_dims[-1], out_features=1, bias=bias)
        self.hidden = ReLU_MLP(layer_dims=hidden_layer_dims,  bias=bias)
    def forward(self, inp_image):
        return self.outp_projector(nn.functional.relu(self.hidden(self.inp_projector(inp_image.flatten(start_dim=1)))))
