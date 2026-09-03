import torch
import torch.nn as nn
from typing import Iterable, Callable, Tuple, Union

class MyCNN(nn.Module):
    '''Custom CNN model with flexible architecture.'''
    def __init__(self,
                 input_shape: Union[int, Tuple[int, int, int]],
                 output_shape: int,
                 activation: Callable,
                 params: Iterable):
        super(MyCNN, self).__init__()

        # input_shape can be either number of channels (int) or (C, H, W)
        if isinstance(input_shape, int):
            in_channels = input_shape
            H, W = (32, 32)  # default spatial size if not provided
        elif isinstance(input_shape, tuple) and len(input_shape) == 3:
            in_channels, H, W = input_shape
        else:
            raise ValueError("input_shape must be int (channels) or tuple (C, H, W)")

        self._in_channels = in_channels
        self._input_hw = (H, W)
        self.activation = activation

        self.conv_layers = self._build_conv_layers(in_channels, params)

        # Compute flattened feature size by running a dummy tensor through conv_layers
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            out = self.conv_layers(dummy)
            num_features = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            self._make_activation(),
            nn.Linear(256, 256),
            self._make_activation(),
            nn.Linear(256, output_shape)
        )

    def _make_activation(self):
        # Accept either an nn.Module class or a callable that returns a module instance
        if isinstance(self.activation, type):
            return self.activation()
        elif callable(self.activation):
            # If activation is already an instance, return it; else try calling it
            try:
                act = self.activation
                return act() if not isinstance(act, nn.Module) else act
            except Exception:
                # Last resort: wrap with identity
                return nn.Identity()
        else:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _build_conv_layers(self, in_channels: int, arch: Iterable) -> nn.Sequential:
        layers = []
        curr_in = in_channels
        for k in arch:
            if isinstance(k, int):
                layers.append(nn.Conv2d(curr_in, k, kernel_size=3, stride=1, padding=1))
                layers.append(self._make_activation())
                curr_in = k
            elif isinstance(k, str) and k.lower().startswith('maxpool'):
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                raise ValueError(f"Unsupported architecture element: {k}")
        return nn.Sequential(*layers)
