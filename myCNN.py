import torch
import torch.nn as nn
from typing import Iterable, Callable

class MyCNN(nn.Module):
    '''Custom CNN model with flexible architecture.'''
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 activation: Callable,
                 params: Iterable):
        super(MyCNN, self).__init__()
        # Expect input_shape to be number of channels
        self.in_channels = input_shape
        self.activation = activation
        self.conv_layers = self.convo_layers(params)

        # The final channel count after convs is inferred by scanning params
        final_channels = None
        for k in params:
            if isinstance(k, int):
                final_channels = k
            elif k == "MaxPool":
                continue
        if final_channels is None:
            final_channels = self.in_channels

        self.fc = nn.Sequential(
            nn.Linear(in_features=final_channels * 4 * 4, out_features=256),
            self.activation(),
            nn.Linear(in_features=256, out_features=256),
            self.activation(),
            nn.Linear(in_features=256, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def convo_layers(self, arch: Iterable) -> nn.Sequential:
        layers = []
        in_ch = self.in_channels
        for k in arch:
            if isinstance(k, int):
                out_ch = k
                layers += [
                    nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    self.activation()
                ]
                in_ch = out_ch
            elif k == "MaxPool":
                layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        # After convs, add an adaptive avg pool to ensure 4x4 feature map if needed
        layers += [nn.AdaptiveAvgPool2d((4, 4))]
        return nn.Sequential(*layers)
