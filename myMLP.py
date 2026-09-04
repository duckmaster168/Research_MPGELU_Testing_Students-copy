import torch
import torch.nn as nn
from typing import List, Callable

class MyMLP(nn.Module):
    '''Custom MLP model supporting customizable layer depths and activation functions.'''
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 activation: Callable):
        super(MyMLP, self).__init__()
        self.flatten = nn.Flatten()

        # Allow activation to be a class, factory or an nn.Module instance
        self.activation = activation

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            # create activation layer
            if isinstance(self.activation, type):
                layers.append(self.activation())
            elif callable(self.activation):
                try:
                    act = self.activation
                    layers.append(act() if not isinstance(act, nn.Module) else act)
                except Exception:
                    layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)
