import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PGELU(nn.Module):
    '''
    Parametric Gaussian Error Linear Unit (PGELU) Activation Function.
    Formulation: f(x) = x * (1 + tanh(alpha * x + beta * x^3))
    '''
    def __init__(self, alpha_param: float = 1.0, beta_param: float = 0.04):
        super(PGELU, self).__init__()
        self.alpha_param = nn.Parameter(torch.tensor(alpha_param, dtype=torch.float32))
        self.beta_param = nn.Parameter(torch.tensor(beta_param, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.mul(x, (1 + torch.tanh(torch.mul(self.alpha_param, x) + torch.mul(self.beta_param, torch.pow(x, 3)))))
        return output

class LambdaGELU(nn.Module):
    '''
    Lambda-GELU Activation Function.
    Formulation: f(x) = (x / 2) * (1 + erf(lambda * x / sqrt(2)))
    '''
    def __init__(self, lambda_param: float = 1.0):
        super(LambdaGELU, self).__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_param, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.mul(0.5 * x, (1.0 + torch.erf(torch.mul(self.lambda_param, x) / np.sqrt(2))))
        return output

class MPGELU(nn.Module):
    '''
    Modified Parametric Gaussian Error Linear Unit (MPGELU) Activation Function.
    Constraint: lambda = 1 + softplus(s)
    Formulation: f(x) = (x / 2) * (1 + erf(lambda * x / sqrt(2)))
    '''
    def __init__(self, s_param: float = 0.0, use_softplus: bool = True):
        super(MPGELU, self).__init__()
        self.s_param = nn.Parameter(torch.tensor(s_param, dtype=torch.float32))
        self.use_softplus = bool(use_softplus)

    def get_lambda(self) -> torch.Tensor:
        if self.use_softplus:
            return 1.0 + F.softplus(self.s_param)
        else:
            return 1.0 + torch.log(torch.tensor(1.0, device=self.s_param.device) + torch.exp(self.s_param))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = self.get_lambda()
        output = torch.mul(0.5 * x, (1.0 + torch.erf(torch.mul(lam, x) / np.sqrt(2))))
        return output

def get_activation(name: str):
    '''Factory function mapping identifier string to activation class.'''
    name = name.lower()
    mapping = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "pgelu": PGELU,
        "lambdagelu": LambdaGELU,
        "mpgelu": MPGELU
    }
    if name not in mapping:
        raise ValueError(f"Activation function '{name}' is not recognized. Options: {list(mapping.keys())}")
    return mapping[name]
